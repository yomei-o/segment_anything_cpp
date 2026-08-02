// Pure-C++ SAM ViT-B image encoder (inference). Input [1,3,1024,1024] -> [1,256,64,64].
// Plain ViT: patch_embed(conv16) + abs pos_embed + 12 blocks (windowed 14; GLOBAL at 2,5,8,11) with
// decomposed relative position embeddings (MViTv2), + neck. Weights in export_vitb.py order.
#pragma once
#include "autograd.hpp"
#include "sam_ops.hpp"       // layernorm2d
#include "depth_ops.hpp"     // layernorm, gelu, slice_cols, hcat
#include "ops2d.hpp"         // reshape, mul_scalar, softmax_rows
#include "linalg.hpp"        // matmul, transpose2d
#include "face_ops.hpp"      // add_rowvec
#include "net_tinyvit.hpp"   // tv_detach, tok2sp, sp2tok, TvitW loader pattern (reused helpers)
#include <fstream>
#include <cstdint>
#include <string>
#include <vector>
#include <cmath>

struct VitbW {
  std::vector<float> buf; size_t off = 0;
  Tensor take(std::vector<int64_t> shape) { int64_t n = 1; for (auto d : shape) n *= d;
    Tensor t = from_data(shape, std::vector<float>(buf.begin() + off, buf.begin() + off + n), false); off += n; return t; }
};
inline VitbW load_vitb(const std::string& dir, bool fp16 = false) {
  std::string D = dir; if (!D.empty() && D.back() != '/') D += '/'; VitbW w;
  std::string fn = D + (fp16 ? "vitb_weights_fp16.bin" : "vitb_weights.bin");
  std::ifstream f(fn, std::ios::binary); if (!f) { printf("missing %s\n", fn.c_str()); std::exit(1); }
  if (fp16) { f.seekg(0, std::ios::end); size_t n = f.tellg() / 2; f.seekg(0); std::vector<uint16_t> h(n); f.read((char*)h.data(), n * 2);
    w.buf.resize(n); for (size_t i = 0; i < n; ++i) w.buf[i] = tvit_h2f(h[i]); return w; }
  f.seekg(0, std::ios::end); size_t n = f.tellg() / 4; f.seekg(0); w.buf.resize(n); f.read((char*)w.buf.data(), n * 4);
  return w;
}

// decomposed relative position bias [N,N] for one head: bias[qh*W+qw, kh*W+kw] =
//   sum_c q[qh*W+qw,c]*rel_h[qh-kh+H-1,c] + sum_c q[..,c]*rel_w[qw-kw+W-1,c]   (q_size==k_size)
inline Tensor decomposed_rel_pos(const Tensor& q, const Tensor& rph, const Tensor& rpw, int64_t H, int64_t W) {
  int64_t hd = q->shape[1], N = H * W;
  std::vector<float> rel_h(H * W * H), rel_w(H * W * W);           // rel_h[(qh,qw),kh], rel_w[(qh,qw),kw]
  for (int64_t qh = 0; qh < H; ++qh) for (int64_t qw = 0; qw < W; ++qw) {
    const float* qv = &q->data[(qh * W + qw) * hd];
    for (int64_t kh = 0; kh < H; ++kh) { const float* r = &rph->data[(qh - kh + H - 1) * hd];
      double s = 0; for (int64_t c = 0; c < hd; ++c) s += qv[c] * r[c]; rel_h[(qh * W + qw) * H + kh] = (float)s; }
    for (int64_t kw = 0; kw < W; ++kw) { const float* r = &rpw->data[(qw - kw + W - 1) * hd];
      double s = 0; for (int64_t c = 0; c < hd; ++c) s += qv[c] * r[c]; rel_w[(qw + qh * W) * W + kw] = (float)s; }
  }
  std::vector<float> bias((size_t)N * N);
  for (int64_t qi = 0; qi < N; ++qi) { int64_t qh = qi / W, qw = qi % W;
    for (int64_t kh = 0; kh < H; ++kh) for (int64_t kw = 0; kw < W; ++kw)
      bias[qi * N + kh * W + kw] = rel_h[qi * H + kh] + rel_w[qi * W + kw]; }
  return from_data({N, N}, bias, false);
}

// multi-head self-attention with decomposed rel-pos on [N,C] (N=H*W, standard q/k/v-block qkv)
inline Tensor vitb_attn(const Tensor& x, const Tensor& qkvW, const Tensor& qkvB, const Tensor& projW, const Tensor& projB,
                        const Tensor& rph, const Tensor& rpw, int heads, int64_t H, int64_t W) {
  int64_t C = x->shape[1], hd = C / heads;
  Tensor qkv = add_rowvec(matmul(x, qkvW), qkvB);                 // [N, 3C]; q=[0,C) k=[C,2C) v=[2C,3C), head-major
  float scale = 1.f / std::sqrt((float)hd);
  std::vector<Tensor> outs;
  for (int h = 0; h < heads; ++h) {
    Tensor q = slice_cols(qkv, h * hd, h * hd + hd);
    Tensor k = slice_cols(qkv, C + h * hd, C + h * hd + hd);
    Tensor v = slice_cols(qkv, 2 * C + h * hd, 2 * C + h * hd + hd);
    Tensor att = add(mul_scalar(matmul(q, transpose2d(k)), scale), decomposed_rel_pos(q, rph, rpw, H, W));
    outs.push_back(matmul(softmax_rows(att), v));
  }
  return add_rowvec(matmul(hcat(outs), projW), projB);
}

// ViT-B block: x + attn(norm1(x)) [windowed unless global], x + mlp(norm2(x))
inline Tensor vitb_block(const Tensor& x, VitbW& w, int64_t H, int64_t W, int64_t C, int heads, int ws) {
  Tensor n1w = w.take({C}), n1b = w.take({C});
  Tensor qkvW = w.take({C, 3 * C}), qkvB = w.take({3 * C});
  Tensor projW = w.take({C, C}), projB = w.take({C});
  int64_t Lh = ws > 0 ? 2 * ws - 1 : 2 * H - 1, Lw = ws > 0 ? 2 * ws - 1 : 2 * W - 1;
  Tensor rph = w.take({Lh, C / heads}), rpw = w.take({Lw, C / heads});
  Tensor n2w = w.take({C}), n2b = w.take({C});
  Tensor f1w = w.take({C, 4 * C}), f1b = w.take({4 * C}), f2w = w.take({4 * C, C}), f2b = w.take({C});

  Tensor xn = layernorm(x, n1w, n1b, 1e-6f);
  Tensor a;
  if (ws > 0 && !(H == ws && W == ws)) {
    Windows wd = to_windows(xn, H, W, C, ws); std::vector<Tensor> outs;
    for (auto& win : wd.win) outs.push_back(vitb_attn(win, qkvW, qkvB, projW, projB, rph, rpw, heads, ws, ws));
    a = from_windows(outs, H, W, C, ws, wd.nH, wd.nW);
  } else {
    a = vitb_attn(xn, qkvW, qkvB, projW, projB, rph, rpw, heads, H, W);
  }
  Tensor h = add(x, a);
  Tensor m = add_rowvec(matmul(gelu(add_rowvec(matmul(layernorm(h, n2w, n2b, 1e-6f), f1w), f1b)), f2w), f2b);
  return tv_detach(add(h, m));
}

inline Tensor g_vb_pe = nullptr, g_vb_b0 = nullptr, g_vb_b2 = nullptr, g_vb_b11 = nullptr;
inline Tensor vitb_forward(const Tensor& img, VitbW& w) {
  const int64_t C = 768, S = 64; const int heads = 12, ws = 14;
  Tensor pew = w.take({C, 3, 16, 16}), peb = w.take({C});
  Tensor pos = w.take({S * S, C});                                // pos_embed [1,64,64,768] as [4096,768]
  Tensor x = add(sp2tok(conv2d(img, pew, peb, 16, 0, 1)), pos);   // patch_embed -> tokens [4096,768] + pos
  g_vb_pe = x;
  const bool global[12] = {0,0,1,0,0,1,0,0,1,0,0,1};
  for (int i = 0; i < 12; ++i) {
    x = vitb_block(x, w, S, S, C, heads, global[i] ? 0 : ws);
    if (i == 0) g_vb_b0 = x; if (i == 2) g_vb_b2 = x; if (i == 11) g_vb_b11 = x;
  }
  // neck: [1,768,64,64] -> conv(768->256,1x1)->LN2d->conv(256->256,3x3)->LN2d
  Tensor s = tok2sp(x, C, S, S);
  Tensor nc0 = w.take({256, C, 1, 1}), ln0w = w.take({256}), ln0b = w.take({256});
  Tensor nc1 = w.take({256, 256, 3, 3}), ln1w = w.take({256}), ln1b = w.take({256});
  Tensor nb = nullptr;
  s = layernorm2d(conv2d(s, nc0, nb, 1, 0, 1), ln0w, ln0b, 1e-6f);
  s = layernorm2d(conv2d(s, nc1, nb, 1, 1, 1), ln1w, ln1b, 1e-6f);
  return s;
}

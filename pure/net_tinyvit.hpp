// Pure-C++ MobileSAM TinyViT image encoder (inference). Input [1,3,1024,1024] -> [1,256,64,64].
// Hierarchical: patch_embed -> ConvLayer(MBConv) -> 3x BasicLayer(windowed TinyViTBlock) -> neck.
// Conv2d_BN layers are BN-fused at export (biased conv). Weights read in export_tinyvit.py order.
#pragma once
#include "autograd.hpp"
#include "sam_ops.hpp"       // layernorm2d
#include "depth_ops.hpp"     // layernorm, gelu, slice_cols, hcat, slice_rows, vcat
#include "ops2d.hpp"         // reshape, mul_scalar, softmax_rows
#include "linalg.hpp"        // matmul, transpose2d
#include "face_ops.hpp"      // relu, add_rowvec
#include <fstream>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>

struct TvitW {
  std::vector<float> buf; size_t off = 0;
  Tensor take(std::vector<int64_t> shape) { int64_t n = 1; for (auto d : shape) n *= d;
    Tensor t = from_data(shape, std::vector<float>(buf.begin() + off, buf.begin() + off + n), false); off += n; return t; }
};
inline float tvit_h2f(uint16_t h) {
  uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF, f;
  if (e == 0) { if (m == 0) f = s; else { int k = -1; do { ++k; m <<= 1; } while (!(m & 0x400)); m &= 0x3FF; f = s | ((uint32_t)(127 - 15 - k) << 23) | (m << 13); } }
  else if (e == 0x1F) f = s | 0x7F800000u | (m << 13); else f = s | ((e - 15 + 127) << 23) | (m << 13);
  float o; std::memcpy(&o, &f, 4); return o;
}
inline TvitW load_tinyvit(const std::string& dir, bool fp16 = false) {
  std::string D = dir; if (!D.empty() && D.back() != '/') D += '/';
  TvitW w;
  if (fp16) { std::ifstream f(D + "tvit_weights_fp16.bin", std::ios::binary); if (!f) { printf("missing %stvit_weights_fp16.bin\n", D.c_str()); std::exit(1); }
    f.seekg(0, std::ios::end); size_t n = f.tellg() / 2; f.seekg(0); std::vector<uint16_t> h(n); f.read((char*)h.data(), n * 2);
    w.buf.resize(n); for (size_t i = 0; i < n; ++i) w.buf[i] = tvit_h2f(h[i]); return w; }
  std::ifstream f(D + "tvit_weights.bin", std::ios::binary); if (!f) { printf("missing %stvit_weights.bin\n", D.c_str()); std::exit(1); }
  f.seekg(0, std::ios::end); size_t n = f.tellg() / 4; f.seekg(0); w.buf.resize(n); f.read((char*)w.buf.data(), n * 4);
  return w;
}

// inference: break the autograd graph between steps so huge intermediates get freed (no backward here)
inline Tensor tv_detach(const Tensor& x) { return from_data(x->shape, x->data, false); }

// ---- layout helpers (B=1) ----
inline Tensor tok2sp(const Tensor& x, int64_t C, int64_t H, int64_t W) {   // [H*W,C] -> [1,C,H,W]
  return reshape(transpose2d(x), {1, C, H, W});
}
inline Tensor sp2tok(const Tensor& x) {                                    // [1,C,H,W] -> [H*W,C]
  int64_t C = x->shape[1], HW = x->shape[2] * x->shape[3];
  return transpose2d(reshape(x, {C, HW}));
}

// ---- MBConv on spatial [1,C,H,W]: c1(1x1)->gelu->c2(dw3x3)->gelu->c3(1x1)-> +shortcut ->gelu ----
inline Tensor mbconv(const Tensor& x, TvitW& w, int64_t Cin, int64_t hidden) {
  Tensor c1w = w.take({hidden, Cin, 1, 1}), c1b = w.take({hidden});
  Tensor c2w = w.take({hidden, 1, 3, 3}), c2b = w.take({hidden});
  Tensor c3w = w.take({Cin, hidden, 1, 1}), c3b = w.take({Cin});
  Tensor h = gelu(conv2d(x, c1w, c1b, 1, 0, 1));
  h = gelu(conv2d(h, c2w, c2b, 1, 1, hidden));
  h = conv2d(h, c3w, c3b, 1, 0, 1);
  return tv_detach(gelu(add(x, h)));
}

// ---- PatchMerging: spatial [1,C,H,W] -> conv1(1x1)->act->conv2(dw3x3,stride)->act->conv3(1x1) -> tokens ----
inline Tensor patch_merging(const Tensor& x, TvitW& w, int64_t Cin, int64_t Cout, int64_t stride) {
  Tensor c1w = w.take({Cout, Cin, 1, 1}), c1b = w.take({Cout});
  Tensor c2w = w.take({Cout, 1, 3, 3}), c2b = w.take({Cout});
  Tensor c3w = w.take({Cout, Cout, 1, 1}), c3b = w.take({Cout});
  Tensor h = gelu(conv2d(x, c1w, c1b, 1, 0, 1));
  h = gelu(conv2d(h, c2w, c2b, stride, 1, Cout));
  h = conv2d(h, c3w, c3b, 1, 0, 1);
  return tv_detach(sp2tok(h));
}

// ---- windowed attention on a single window [N,C] with pre-expanded bias ab[heads,N,N], key_dim=32 ----
inline Tensor tvit_attn_window(const Tensor& x, const Tensor& normw, const Tensor& normb,
                               const Tensor& qkvW, const Tensor& qkvB, const Tensor& projW, const Tensor& projB,
                               const Tensor& ab, int heads, int key_dim) {
  int64_t N = x->shape[0]; int per = key_dim * 3;                 // per-head qkv chunk (attn_ratio=1 -> d=key_dim)
  Tensor xn = layernorm(x, normw, normb, 1e-5f);
  Tensor qkv = add_rowvec(matmul(xn, qkvW), qkvB);               // [N, heads*96]
  float scale = 1.f / std::sqrt((float)key_dim);
  std::vector<Tensor> outs;
  for (int h = 0; h < heads; ++h) {
    Tensor q = slice_cols(qkv, h * per, h * per + key_dim);
    Tensor k = slice_cols(qkv, h * per + key_dim, h * per + 2 * key_dim);
    Tensor v = slice_cols(qkv, h * per + 2 * key_dim, h * per + 3 * key_dim);
    Tensor att = add(mul_scalar(matmul(q, transpose2d(k)), scale), slice_rows(ab, h * N, h * N + N)); // [N,N] + bias_h
    att = softmax_rows(att);
    outs.push_back(matmul(att, v));                              // [N,key_dim]
  }
  return add_rowvec(matmul(hcat(outs), projW), projB);           // [N, dim]
}

// window partition of tokens [H*W,C] with reflect-free zero pad -> list of [ws*ws,C] windows
struct Windows { std::vector<Tensor> win; int64_t nH, nW, pH, pW; };
inline Windows to_windows(const Tensor& x, int64_t H, int64_t W, int64_t C, int64_t ws) {
  int64_t pb = (ws - H % ws) % ws, pr = (ws - W % ws) % ws, pH = H + pb, pW = W + pr;
  int64_t nH = pH / ws, nW = pW / ws;
  Windows out; out.nH = nH; out.nW = nW; out.pH = pH; out.pW = pW;
  for (int64_t bh = 0; bh < nH; ++bh) for (int64_t bw = 0; bw < nW; ++bw) {
    std::vector<float> wd((size_t)ws * ws * C, 0.f);
    for (int64_t iy = 0; iy < ws; ++iy) for (int64_t ix = 0; ix < ws; ++ix) {
      int64_t gy = bh * ws + iy, gx = bw * ws + ix;
      if (gy < H && gx < W) { const float* s = &x->data[(gy * W + gx) * C]; std::copy(s, s + C, &wd[(iy * ws + ix) * C]); }
    }
    out.win.push_back(from_data({ws * ws, C}, wd, false));
  }
  return out;
}
inline Tensor from_windows(const std::vector<Tensor>& win, int64_t H, int64_t W, int64_t C, int64_t ws, int64_t nH, int64_t nW) {
  std::vector<float> od((size_t)H * W * C, 0.f);
  for (int64_t bh = 0; bh < nH; ++bh) for (int64_t bw = 0; bw < nW; ++bw) {
    const Tensor& wt = win[bh * nW + bw];
    for (int64_t iy = 0; iy < ws; ++iy) for (int64_t ix = 0; ix < ws; ++ix) {
      int64_t gy = bh * ws + iy, gx = bw * ws + ix;
      if (gy < H && gx < W) { const float* s = &wt->data[(iy * ws + ix) * C]; std::copy(s, s + C, &od[(gy * W + gx) * C]); }
    }
  }
  return from_data({H * W, C}, od, false);
}

// ---- TinyViTBlock: tokens [L,C], resolution H,W ----
inline Tensor tvit_block(const Tensor& x, TvitW& w, int64_t H, int64_t W, int64_t C, int heads, int ws) {
  int64_t N = (int64_t)ws * ws;
  Tensor normw = w.take({C}), normb = w.take({C});
  Tensor qkvW = w.take({C, heads * 96}), qkvB = w.take({heads * 96});
  Tensor projW = w.take({heads * 32, C}), projB = w.take({C});
  Tensor ab = w.take({heads * N, N});                            // [heads,N,N] flattened rows
  Tensor lcw = w.take({C, 1, 3, 3}), lcb = w.take({C});
  Tensor mnw = w.take({C}), mnb = w.take({C});
  Tensor f1w = w.take({C, 4 * C}), f1b = w.take({4 * C}), f2w = w.take({4 * C, C}), f2b = w.take({C});

  // windowed attention (+ residual)
  Tensor attn_out;
  if (H == ws && W == ws) {
    attn_out = tvit_attn_window(x, normw, normb, qkvW, qkvB, projW, projB, ab, heads, 32);
  } else {
    Windows wd = to_windows(x, H, W, C, ws);
    std::vector<Tensor> outs;
    for (auto& win : wd.win) outs.push_back(tvit_attn_window(win, normw, normb, qkvW, qkvB, projW, projB, ab, heads, 32));
    attn_out = from_windows(outs, H, W, C, ws, wd.nH, wd.nW);
  }
  Tensor h = add(x, attn_out);                                   // residual
  // local conv (dw 3x3), replaces h
  h = sp2tok(conv2d(tok2sp(h, C, H, W), lcw, lcb, 1, 1, C));
  // mlp (+ residual); Mlp has its own LayerNorm
  Tensor m = add_rowvec(matmul(layernorm(h, mnw, mnb, 1e-5f), f1w), f1b);
  m = add_rowvec(matmul(gelu(m), f2w), f2b);
  return add(h, m);
}

inline Tensor g_tv_pe=nullptr,g_tv_l0=nullptr,g_tv_l1=nullptr,g_tv_l2=nullptr,g_tv_l3=nullptr;
// ---- full encoder ----
inline Tensor tinyvit_forward(const Tensor& img, TvitW& w) {
  // patch_embed: conv(3->32,s2,p1)->gelu->conv(32->64,s2,p1) -> [1,64,256,256]
  Tensor pe0w = w.take({32, 3, 3, 3}), pe0b = w.take({32});
  Tensor pe2w = w.take({64, 32, 3, 3}), pe2b = w.take({64});
  Tensor x = tv_detach(conv2d(gelu(conv2d(img, pe0w, pe0b, 2, 1, 1)), pe2w, pe2b, 2, 1, 1));   // [1,64,256,256]
  // stage 0: ConvLayer 2x MBConv (dim64, hidden256), spatial, then PatchMerging 64->128 stride2
  g_tv_pe = x;
  for (int i = 0; i < 2; ++i) x = mbconv(x, w, 64, 256);
  Tensor t = patch_merging(x, w, 64, 128, 2);                    // tokens [128*128,128]
  g_tv_l0 = t;
  // stage 1: 2x TinyViTBlock (dim128, heads4, ws7, res128), PatchMerging 128->160 stride2
  for (int i = 0; i < 2; ++i) t = tvit_block(t, w, 128, 128, 128, 4, 7);
  t = patch_merging(tok2sp(t, 128, 128, 128), w, 128, 160, 2);   // tokens [64*64,160]
  g_tv_l1 = t;
  // stage 2: 6x TinyViTBlock (dim160, heads5, ws14, res64), PatchMerging 160->320 stride1
  for (int i = 0; i < 6; ++i) t = tvit_block(t, w, 64, 64, 160, 5, 14);
  t = patch_merging(tok2sp(t, 160, 64, 64), w, 160, 320, 1);     // tokens [64*64,320]
  g_tv_l2 = t;
  // stage 3: 2x TinyViTBlock (dim320, heads10, ws7, res64), no downsample
  for (int i = 0; i < 2; ++i) t = tvit_block(t, w, 64, 64, 320, 10, 7);
  g_tv_l3 = t;
  // neck: [1,320,64,64] -> conv(320->256,1x1,no bias)->LN2d->conv(256->256,3x3,p1,no bias)->LN2d
  Tensor s = tok2sp(t, 320, 64, 64);
  Tensor nc0 = w.take({256, 320, 1, 1}), ln0w = w.take({256}), ln0b = w.take({256});
  Tensor nc1 = w.take({256, 256, 3, 3}), ln1w = w.take({256}), ln1b = w.take({256});
  Tensor nb = nullptr;
  s = layernorm2d(conv2d(s, nc0, nb, 1, 0, 1), ln0w, ln0b, 1e-6f);
  s = layernorm2d(conv2d(s, nc1, nb, 1, 1, 1), ln1w, ln1b, 1e-6f);
  return s;                                                      // [1,256,64,64]
}

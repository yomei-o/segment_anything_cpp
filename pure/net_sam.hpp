// Pure-C++ Segment Anything SHARED decoder: PromptEncoder + MaskDecoder (MobileSAM & SAM ViT-* share
// these). B=1. Weights read in export_sam.py order from ref/{weights.bin,config.txt}. The image
// encoder (TinyViT / ViT-B) is separate; this file takes the 256x64x64 image embedding as input.
#pragma once
#include "autograd.hpp"
#include "sam_ops.hpp"       // layernorm2d
#include "depth_ops.hpp"     // layernorm, gelu, slice_cols, hcat, slice_rows, vcat, convtranspose2d
#include "ops2d.hpp"         // reshape, mul_scalar, softmax_rows
#include "linalg.hpp"        // matmul, transpose2d
#include "face_ops.hpp"      // relu, add_rowvec
#include <fstream>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>

struct SamCfg { int embed = 256, img = 64, input = 1024, mask_tokens = 4, heads = 8, depth = 2; };

struct SamW {
  std::vector<float> buf; size_t off = 0; SamCfg cfg;
  std::vector<Tensor> cache; size_t ti = 0; bool built = false;
  Tensor take(std::vector<int64_t> shape) {
    if (built) return cache[ti++];
    int64_t n = 1; for (auto d : shape) n *= d;
    Tensor t = from_data(shape, std::vector<float>(buf.begin() + off, buf.begin() + off + n), true);
    off += n; cache.push_back(t); ++ti; return t;
  }
  void rewind() { ti = 0; }
  void finalize() { built = true; }
  std::vector<Tensor>& parameters() { return cache; }
};

inline float sam_half2float(uint16_t h) {
  uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF, f;
  if (e == 0) { if (m == 0) f = s; else { int k = -1; do { ++k; m <<= 1; } while (!(m & 0x400)); m &= 0x3FF; f = s | ((uint32_t)(127 - 15 - k) << 23) | (m << 13); } }
  else if (e == 0x1F) f = s | 0x7F800000u | (m << 13); else f = s | ((e - 15 + 127) << 23) | (m << 13);
  float o; std::memcpy(&o, &f, 4); return o;
}
inline SamW load_sam_decoder(const std::string& dir, bool fp16 = false) {
  std::string D = dir; if (!D.empty() && D.back() != '/') D += '/';
  SamW w; { std::ifstream f(D + "config.txt"); std::string k; int v;
    while (f >> k >> v) { if (k == "embed_dim") w.cfg.embed = v; else if (k == "image_embed") w.cfg.img = v;
      else if (k == "input_image") w.cfg.input = v; else if (k == "num_mask_tokens") w.cfg.mask_tokens = v;
      else if (k == "heads") w.cfg.heads = v; else if (k == "tf_depth") w.cfg.depth = v; } }
  if (fp16) { std::ifstream f(D + "weights_fp16.bin", std::ios::binary); if (!f) { printf("missing %sweights_fp16.bin\n", D.c_str()); std::exit(1); }
    f.seekg(0, std::ios::end); size_t n = f.tellg() / 2; f.seekg(0); std::vector<uint16_t> h(n); f.read((char*)h.data(), n * 2);
    w.buf.resize(n); for (size_t i = 0; i < n; ++i) w.buf[i] = sam_half2float(h[i]); return w; }
  std::ifstream f(D + "weights.bin", std::ios::binary); if (!f) { printf("missing %sweights.bin\n", D.c_str()); std::exit(1); }
  f.seekg(0, std::ios::end); size_t n = f.tellg() / 4; f.seekg(0); w.buf.resize(n); f.read((char*)w.buf.data(), n * 4);
  return w;
}

// ---- flexible attention (B=1): q[Nq,E], k[Nk,E], v[Nk,E]; internal dim = qW->shape[1] ----
struct AttnW { Tensor qW, qB, kW, kB, vW, vB, oW, oB; };
inline AttnW read_attn(SamW& w, int64_t E, int64_t internal) {
  AttnW a; a.qW = w.take({E, internal}); a.qB = w.take({internal});
  a.kW = w.take({E, internal}); a.kB = w.take({internal});
  a.vW = w.take({E, internal}); a.vB = w.take({internal});
  a.oW = w.take({internal, E}); a.oB = w.take({E}); return a;
}
inline Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v, const AttnW& a, int heads) {
  Tensor Q = add_rowvec(matmul(q, a.qW), a.qB), K = add_rowvec(matmul(k, a.kW), a.kB), V = add_rowvec(matmul(v, a.vW), a.vB);
  int64_t internal = a.qW->shape[1], hd = internal / heads; float scale = 1.f / std::sqrt((float)hd);
  std::vector<Tensor> outs;
  for (int h = 0; h < heads; ++h) {
    Tensor qh = slice_cols(Q, h * hd, h * hd + hd), kh = slice_cols(K, h * hd, h * hd + hd), vh = slice_cols(V, h * hd, h * hd + hd);
    Tensor att = softmax_rows(mul_scalar(matmul(qh, transpose2d(kh)), scale));
    outs.push_back(matmul(att, vh));
  }
  return add_rowvec(matmul(hcat(outs), a.oW), a.oB);
}

// ---- positional encoding (constant): _pe(coords[N,2]) = cat(sin,cos)(2π·((2c-1)@G)), G[2,128] ----
inline Tensor pe_encode(const std::vector<float>& coords, int64_t N, const Tensor& G) {
  int64_t F = G->shape[1];                 // 128
  std::vector<float> out(N * 2 * F);
  const float TWO_PI = 6.28318530717958648f;
  for (int64_t n = 0; n < N; ++n) {
    float cx = 2.f * coords[n * 2 + 0] - 1.f, cy = 2.f * coords[n * 2 + 1] - 1.f;
    for (int64_t j = 0; j < F; ++j) {
      float p = TWO_PI * (cx * G->data[0 * F + j] + cy * G->data[1 * F + j]);
      out[n * 2 * F + j] = std::sin(p); out[n * 2 * F + F + j] = std::cos(p);
    }
  }
  return from_data({N, 2 * F}, out, false);
}

// ---- SHARED forward: image embedding [1,256,64,64] + point prompts -> masks[4,256,256], iou[4] ----
// points: coords (x,y) in the 1024-input space, labels (1=fg,0=bg). A pad point is appended.
struct SamOut { Tensor masks, iou, sparse, image_pe; };
inline SamOut sam_decode(const Tensor& img_embed, const std::vector<float>& pts, const std::vector<int>& labels, SamW& w) {
  auto& c = w.cfg; int64_t E = c.embed, S = c.img, HW = S * S, Np = (int64_t)labels.size();
  // ---- PROMPT ENCODER weights ----
  Tensor G = w.take({2, E / 2});
  Tensor ptemb[4]; for (int i = 0; i < 4; ++i) ptemb[i] = w.take({1, E});
  Tensor not_a_point = w.take({1, E}), no_mask = w.take({1, E});
  Tensor mc0w = w.take({4, 1, 2, 2}), mc0b = w.take({4}), ml1w = w.take({4}), ml1b = w.take({4});
  Tensor mc3w = w.take({16, 4, 2, 2}), mc3b = w.take({16}), ml4w = w.take({16}), ml4b = w.take({16});
  Tensor mc6w = w.take({E, 16, 1, 1}), mc6b = w.take({E});

  // sparse embeddings: real points (pe + point_embeddings[label]) then a pad point (= not_a_point)
  std::vector<float> pc(Np * 2);
  for (int64_t i = 0; i < Np; ++i) { pc[i * 2] = (pts[i * 2] + 0.5f) / c.input; pc[i * 2 + 1] = (pts[i * 2 + 1] + 0.5f) / c.input; }
  Tensor ppe = pe_encode(pc, Np, G);                                 // [Np,256]
  std::vector<Tensor> srows;
  for (int64_t i = 0; i < Np; ++i) srows.push_back(add(slice_rows(ppe, i, i + 1), ptemb[labels[i]]));
  srows.push_back(not_a_point);                                      // pad point (pe zeroed -> not_a_point)
  Tensor sparse = srows.size() == 1 ? srows[0] : vcat(srows);         // [Np+1,256]

  // image positional encoding as tokens [HW,256], token (h,w)=h*S+w, coords ((w+.5)/S,(h+.5)/S)
  std::vector<float> gc(HW * 2);
  for (int64_t h = 0; h < S; ++h) for (int64_t ww = 0; ww < S; ++ww) { gc[(h * S + ww) * 2] = (ww + 0.5f) / S; gc[(h * S + ww) * 2 + 1] = (h + 0.5f) / S; }
  Tensor image_pe = pe_encode(gc, HW, G);                            // [HW,256]

  // ---- MASK DECODER ----
  Tensor iou_token = w.take({1, E}), mask_tokens = w.take({c.mask_tokens, E});
  // tokens = [iou_token; mask_tokens; sparse]; query_pe = tokens (constant); key_pe = image_pe
  Tensor tokens = vcat({iou_token, mask_tokens, sparse});            // [5+Np,256]
  // keys = image_embed tokens + no_mask (dense, no-mask case): src_tokens[i][c] = img[c,h,w] + no_mask[c]
  Tensor img_tok = add_rowvec(transpose2d(reshape(img_embed, {E, HW})), no_mask);   // [HW,256]
  Tensor queries = tokens, keys = img_tok;
  Tensor query_pe = tokens, key_pe = image_pe;

  for (int L = 0; L < c.depth; ++L) {
    AttnW self_a = read_attn(w, E, E);
    Tensor n1w = w.take({E}), n1b = w.take({E});
    AttnW cross_ti = read_attn(w, E, E / 2);
    Tensor n2w = w.take({E}), n2b = w.take({E});
    int64_t MD = 2048;   // TwoWayTransformer mlp_dim (fixed, not 4*E)
    Tensor l1w = w.take({E, MD}), l1b = w.take({MD}), l2w = w.take({MD, E}), l2b = w.take({E});
    Tensor n3w = w.take({E}), n3b = w.take({E});
    Tensor n4w = w.take({E}), n4b = w.take({E});
    AttnW cross_it = read_attn(w, E, E / 2);
    // 1. self-attn: layer 0 skips pe AND has no residual (queries replaced); later layers add pe + residual
    if (L == 0) queries = attention(queries, queries, queries, self_a, c.heads);
    else { Tensor q = add(queries, query_pe); queries = add(queries, attention(q, q, queries, self_a, c.heads)); }
    queries = layernorm(queries, n1w, n1b, 1e-5f);
    // 2. cross token->image
    { Tensor q = add(queries, query_pe), k = add(keys, key_pe);
      queries = add(queries, attention(q, k, keys, cross_ti, c.heads)); }
    queries = layernorm(queries, n2w, n2b, 1e-5f);
    // 3. mlp (two-way transformer MLPBlock uses ReLU, not GELU)
    { Tensor h = add_rowvec(matmul(queries, l1w), l1b); h = add_rowvec(matmul(relu(h), l2w), l2b);
      queries = add(queries, h); }
    queries = layernorm(queries, n3w, n3b, 1e-5f);
    // 4. cross image->token
    { Tensor q = add(keys, key_pe), k = add(queries, query_pe);
      keys = add(keys, attention(q, k, queries, cross_it, c.heads)); }
    keys = layernorm(keys, n4w, n4b, 1e-5f);
  }
  // final attn token->image
  AttnW fin = read_attn(w, E, E / 2);
  Tensor nfw = w.take({E}), nfb = w.take({E});
  { Tensor q = add(queries, query_pe), k = add(keys, key_pe);
    queries = add(queries, attention(q, k, keys, fin, c.heads)); }
  queries = layernorm(queries, nfw, nfb, 1e-5f);

  Tensor iou_out = slice_rows(queries, 0, 1);                        // [1,256]
  Tensor mask_out = slice_rows(queries, 1, 1 + c.mask_tokens);       // [4,256]

  // upscaling: keys[HW,256] -> [1,256,S,S] -> ConvT->LN2d->GELU->ConvT->GELU  (4x -> [1,32,4S,4S])
  Tensor ct0w = w.take({E, E / 4, 2, 2}), ct0b = w.take({E / 4}), ul1w = w.take({E / 4}), ul1b = w.take({E / 4});
  Tensor ct3w = w.take({E / 4, E / 8, 2, 2}), ct3b = w.take({E / 8});
  Tensor src = reshape(transpose2d(keys), {1, E, S, S});
  Tensor u = gelu(layernorm2d(convtranspose2d(src, ct0w, ct0b, 2), ul1w, ul1b, 1e-6f));
  u = gelu(convtranspose2d(u, ct3w, ct3b, 2));                       // [1,32,4S,4S]
  int64_t US = 4 * S, C32 = E / 8;

  // hypernetworks: per mask token MLP(256->256->256->32); masks = hyper[4,32] @ upscaled[32,US*US]
  std::vector<Tensor> hyper;
  for (int i = 0; i < c.mask_tokens; ++i) {
    Tensor h0w = w.take({E, E}), h0b = w.take({E}), h1w = w.take({E, E}), h1b = w.take({E}), h2w = w.take({E, C32}), h2b = w.take({C32});
    Tensor mt = slice_rows(mask_out, i, i + 1);                      // [1,256]
    Tensor h = relu(add_rowvec(matmul(mt, h0w), h0b));
    h = relu(add_rowvec(matmul(h, h1w), h1b));
    h = add_rowvec(matmul(h, h2w), h2b);                            // [1,32]
    hyper.push_back(h);
  }
  Tensor hyper_in = vcat(hyper);                                     // [4,32]
  Tensor masks = reshape(matmul(hyper_in, reshape(u, {C32, US * US})), {c.mask_tokens, US, US});

  // iou head MLP(256->256->256->4)
  Tensor i0w = w.take({E, E}), i0b = w.take({E}), i1w = w.take({E, E}), i1b = w.take({E}), i2w = w.take({E, c.mask_tokens}), i2b = w.take({c.mask_tokens});
  Tensor ih = relu(add_rowvec(matmul(iou_out, i0w), i0b));
  ih = relu(add_rowvec(matmul(ih, i1w), i1b));
  Tensor iou = add_rowvec(matmul(ih, i2w), i2b);                     // [1,4]

  return {masks, iou, sparse, image_pe};
}

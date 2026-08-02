// MobileSAM in WASM: encode an image ONCE (TinyViT, slow), then segment per click (decoder, fast).
//   fn_ready()                  load fp16 weights from /sam/
//   fn_encode(rgba,w,h)         preprocess + run TinyViT -> store 256x64x64 embedding (once per image)
//   fn_decode(px,py) -> float*  run prompt+mask decoder for a click (original-image px,py) -> 256x256
//                               mask logits (best of the 3 multimask by IoU). Threshold >0 in JS.
#include "net_tinyvit.hpp"
#include "net_sam.hpp"
#include <emscripten/emscripten.h>
#include <vector>
#include <algorithm>
#include <cmath>

static TvitW* g_tw = nullptr; static SamW* g_sw = nullptr;
static Tensor g_emb = nullptr; static float g_scale = 1.f; static int g_W0 = 0, g_H0 = 0;
static std::vector<float> g_mask(256 * 256); static float g_iou[3] = {0, 0, 0};

extern "C" {

EMSCRIPTEN_KEEPALIVE int fn_ready() {
  if (!g_tw) g_tw = new TvitW(load_tinyvit("/sam/", true));
  if (!g_sw) g_sw = new SamW(load_sam_decoder("/sam/", true));
  return (g_tw && g_sw) ? 1 : 0;
}
EMSCRIPTEN_KEEPALIVE float fn_scale() { return g_scale; }
EMSCRIPTEN_KEEPALIVE float fn_iou(int i) { return (i >= 0 && i < 3) ? g_iou[i] : 0.f; }

EMSCRIPTEN_KEEPALIVE int fn_encode(unsigned char* rgba, int w, int h) {
  if (!g_tw) fn_ready();
  g_W0 = w; g_H0 = h;
  const int S = 1024; g_scale = (float)S / std::max(w, h);
  int newW = (int)std::round(w * g_scale), newH = (int)std::round(h * g_scale);
  const float mean[3] = {123.675f, 116.28f, 103.53f}, sd[3] = {58.395f, 57.12f, 57.375f};
  std::vector<float> x(3 * S * S, 0.f);
  auto samp = [&](int yy, int xx, int c) { yy = std::clamp(yy, 0, h - 1); xx = std::clamp(xx, 0, w - 1); return (float)rgba[((size_t)yy * w + xx) * 4 + c]; };
  for (int y = 0; y < newH; ++y) for (int xx = 0; xx < newW; ++xx) {
    float sy = (y + 0.5f) / g_scale - 0.5f, sx = (xx + 0.5f) / g_scale - 0.5f;
    int y0 = (int)std::floor(sy), x0 = (int)std::floor(sx); float fy = sy - y0, fx = sx - x0;
    for (int c = 0; c < 3; ++c) { float v = samp(y0, x0, c) * (1 - fx) * (1 - fy) + samp(y0, x0 + 1, c) * fx * (1 - fy)
                                          + samp(y0 + 1, x0, c) * (1 - fx) * fy + samp(y0 + 1, x0 + 1, c) * fx * fy;
      x[((size_t)c * S + y) * S + xx] = (v - mean[c]) / sd[c]; }
  }
  g_emb = tinyvit_forward(from_data({1, 3, S, S}, x), *g_tw);
  return 1;
}

// click in original-image pixels -> best 256x256 mask logits (aligned to the padded 1024 canvas)
EMSCRIPTEN_KEEPALIVE float* fn_decode(float px, float py) {
  if (!g_emb) return g_mask.data();
  std::vector<float> pts = {px * g_scale, py * g_scale}; std::vector<int> labels = {1};
  SamOut o = sam_decode(g_emb, pts, labels, *g_sw);
  int best = 1; for (int i = 2; i <= 3; ++i) if (o.iou->data[i] > o.iou->data[best]) best = i;
  for (int i = 0; i < 3; ++i) g_iou[i] = o.iou->data[i + 1];
  std::copy(&o.masks->data[best * 256 * 256], &o.masks->data[best * 256 * 256] + 256 * 256, g_mask.begin());
  return g_mask.data();
}

}

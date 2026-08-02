// Segment Anything (MobileSAM) end-to-end inference (pure C++, no Python):
//   image + click point -> TinyViT encoder -> mask decoder -> best mask -> overlay PNG.
// Preprocess = ResizeLongestSide(1024) + ImageNet-ish SAM normalize + pad to 1024. Postprocess =
// upscale 256->1024, crop to the resized region, resize to original, threshold at 0.
//   build: cl /std:c++20 /O2 /EHsc /Zc:preprocessor /DNOMINMAX /Ipure\third_party pure\infer_sam.cpp
//   run:   infer_sam <img> <x> <y> [out.png] [ref_dir]      (x,y = click in original image pixels)
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "net_tinyvit.hpp"
#include "net_sam.hpp"
#include <cstdio>
#include <cmath>
#include <algorithm>

int main(int argc, char** argv) {
  if (argc < 4) { printf("usage: infer_sam <img> <x> <y> [out.png] [ref_dir]\n"); return 1; }
  std::string src = argv[1]; float px = (float)atof(argv[2]), py = (float)atof(argv[3]);
  std::string out = argc > 4 ? argv[4] : "sam_mask.png", RF = argc > 5 ? argv[5] : "pure/ref/";
  if (RF.back() != '/') RF += '/';
  int W0, H0, ch; unsigned char* im = stbi_load(src.c_str(), &W0, &H0, &ch, 3);
  if (!im) { printf("cannot load %s\n", src.c_str()); return 1; }

  const int S = 1024;
  float scale = (float)S / std::max(W0, H0);
  int newW = (int)std::round(W0 * scale), newH = (int)std::round(H0 * scale);   // <= 1024
  const float mean[3] = {123.675f, 116.28f, 103.53f}, sd[3] = {58.395f, 57.12f, 57.375f};
  std::vector<float> x(3 * S * S, 0.f);                       // zero-padded to 1024x1024
  auto samp = [&](int yy, int xx, int c) { yy = std::clamp(yy, 0, H0 - 1); xx = std::clamp(xx, 0, W0 - 1); return (float)im[(yy * W0 + xx) * 3 + c]; };
  for (int y = 0; y < newH; ++y) for (int xx = 0; xx < newW; ++xx) {
    float sy = (y + 0.5f) / scale - 0.5f, sx = (xx + 0.5f) / scale - 0.5f;
    int y0 = (int)std::floor(sy), x0 = (int)std::floor(sx); float fy = sy - y0, fx = sx - x0;
    for (int c = 0; c < 3; ++c) { float v = samp(y0, x0, c) * (1 - fx) * (1 - fy) + samp(y0, x0 + 1, c) * fx * (1 - fy)
                                          + samp(y0 + 1, x0, c) * (1 - fx) * fy + samp(y0 + 1, x0 + 1, c) * fx * fy;
      x[(c * S + y) * S + xx] = (v - mean[c]) / sd[c]; }
  }

  printf("encoding (TinyViT, 1024x1024)...\n"); fflush(stdout);
  TvitW tw = load_tinyvit(RF); SamW sw = load_sam_decoder(RF);
  Tensor emb = tinyvit_forward(from_data({1, 3, S, S}, x), tw);          // [1,256,64,64]
  std::vector<float> pts = {px * scale, py * scale}; std::vector<int> labels = {1};   // click -> 1024 space
  SamOut o = sam_decode(emb, pts, labels, sw);                           // masks[4,256,256], iou[4]

  // pick best multimask (highest predicted IoU among indices 1..3)
  int best = 1; for (int i = 2; i <= 3; ++i) if (o.iou->data[i] > o.iou->data[best]) best = i;
  printf("iou = [%.3f %.3f %.3f] -> mask %d\n", o.iou->data[1], o.iou->data[2], o.iou->data[3], best);
  const float* m = &o.masks->data[best * 256 * 256];                     // 256x256 logits

  // upscale mask 256->1024 (aligned to padded canvas), sample at original pixels, threshold 0
  std::vector<unsigned char> outimg(W0 * H0 * 3);
  long area = 0;
  for (int y = 0; y < H0; ++y) for (int xx = 0; xx < W0; ++xx) {
    float rx = (xx + 0.5f) * scale, ry = (y + 0.5f) * scale;             // original -> resized(1024) space
    float mx = rx / 1024.f * 256.f - 0.5f, my = ry / 1024.f * 256.f - 0.5f;
    int mxi = std::clamp((int)std::round(mx), 0, 255), myi = std::clamp((int)std::round(my), 0, 255);
    bool fg = m[myi * 256 + mxi] > 0.f; if (fg) ++area;
    unsigned char* p = &outimg[(y * W0 + xx) * 3];
    for (int c = 0; c < 3; ++c) { float v = im[(y * W0 + xx) * 3 + c];
      p[c] = (unsigned char)std::clamp(fg ? v * 0.5f + (c == 0 ? 255.f : c == 1 ? 80.f : 40.f) * 0.5f : v, 0.f, 255.f); }
  }
  // draw the click point (green cross)
  for (int d = -6; d <= 6; ++d) { int cx = (int)px, cy = (int)py;
    auto pt = [&](int yy, int xx) { if (xx >= 0 && xx < W0 && yy >= 0 && yy < H0) { unsigned char* p = &outimg[(yy * W0 + xx) * 3]; p[0] = 0; p[1] = 255; p[2] = 0; } };
    pt(cy, cx + d); pt(cy + d, cx); }
  stbi_write_png(out.c_str(), W0, H0, 3, outimg.data(), W0 * 3);
  printf("mask area = %ld px (%.1f%%) -> wrote %s\n", area, 100.0 * area / (W0 * H0), out.c_str());
  stbi_image_free(im); return 0;
}

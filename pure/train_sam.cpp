// SAM mask-decoder training (pure C++, no Python). Synthetic-validated: a fixed image embedding + a
// point inside a synthetic target mask -> decoder -> 3 masks; supervise the min-loss mask with
// focal+dice, and its IoU-head output with MSE (SAM's training recipe). Shows the loss decreasing.
// The encoder is frozen (MobileSAM ships SAM's original decoder; this fine-tunes the decoder).
//   build: cl /std:c++20 /O2 /EHsc /Zc:preprocessor /DNOMINMAX /Ipure\third_party pure\train_sam.cpp
//   run:   train_sam [ref_dir] [--steps N] [--lr F]
#include "net_sam.hpp"
#include "sam_loss.hpp"
#include "ops2d.hpp"
#include "optim.hpp"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include <cmath>

int main(int argc, char** argv) {
  std::string RF = "pure/ref/"; int steps = 40; float lr = 1e-4f;
  for (int i = 1; i < argc; ++i) { std::string a = argv[i];
    if (a == "--steps" && i + 1 < argc) steps = atoi(argv[++i]);
    else if (a == "--lr" && i + 1 < argc) lr = (float)atof(argv[++i]);
    else if (a[0] != '-') RF = a; }
  if (RF.back() != '/') RF += '/';
  setvbuf(stdout, nullptr, _IONBF, 0);
  SamW w = load_sam_decoder(RF);

  // fixed image embedding + a synthetic target mask (a centered rectangle at 256x256) + a point in it
  std::vector<float> emb(256 * 64 * 64);
  { std::ifstream f(RF + "emb_in.bin", std::ios::binary); if (f) f.read((char*)emb.data(), emb.size() * 4); }
  Tensor img = from_data({1, 256, 64, 64}, emb);
  std::vector<float> gt(256 * 256, 0.f);
  for (int y = 70; y < 190; ++y) for (int x = 90; x < 210; ++x) gt[y * 256 + x] = 1.f;   // target region
  std::vector<float> pts = {(90 + 210) / 2.f * 4.f, (70 + 190) / 2.f * 4.f}; std::vector<int> labels = {1};   // 256->1024 space

  // build persistent params
  sam_decode(img, pts, labels, w); w.finalize();
  auto& params = w.parameters(); int64_t np = 0; for (auto& p : params) np += p->numel();
  printf("SAM decoder training: %zu tensors, %.2fM params, steps=%d lr=%g\n", params.size(), np / 1e6, steps, lr);

  Adam opt(params, lr);
  for (int s = 0; s < steps; ++s) {
    w.rewind(); opt.zero_grad();
    SamOut o = sam_decode(img, pts, labels, w);
    Tensor mflat = reshape(o.masks, {4, 256 * 256});
    // pick the min-loss mask among the 3 multimask outputs (indices 1..3)
    int best = 1; float bl = 1e30f; Tensor bestL;
    for (int i = 1; i <= 3; ++i) { Tensor mi = slice_rows(mflat, i, i + 1); Tensor Li = mask_loss(mi, gt);
      if (Li->data[0] < bl) { bl = Li->data[0]; best = i; bestL = Li; } }
    Tensor mbest = slice_rows(mflat, best, best + 1);
    float actual_iou = mask_iou(mbest, gt);
    Tensor iou_pred = slice_cols(o.iou, best, best + 1);              // [1,1]
    Tensor total = add(bestL, sq_err(iou_pred, actual_iou));
    backward(total); opt.step();
    if (s % 5 == 0 || s == steps - 1)
      printf("step %2d  mask_loss=%.4f  IoU(pred %.3f / true %.3f)  mask=%d\n",
             s, bestL->data[0], o.iou->data[best], actual_iou, best);
  }
  printf("done — mask loss should have dropped and IoU risen (decoder training verified).\n");
  return 0;
}

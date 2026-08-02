// Parity: pure-C++ SHARED decoder (prompt encoder + mask decoder) vs PyTorch mobile_sam, on a fixed
// image embedding + one foreground point. Compares sparse embeddings, IoU predictions, and masks.
//   build: cl /std:c++20 /O2 /EHsc /Zc:preprocessor /DNOMINMAX /Ipure\third_party pure\m1_decoder.cpp
#include "net_sam.hpp"
#include <cstdio>
#include <fstream>
#include <vector>
#include <cmath>

static std::vector<float> readbin(const std::string& p, size_t n) {
  std::vector<float> v(n); std::ifstream f(p, std::ios::binary);
  if (!f) { printf("missing %s\n", p.c_str()); std::exit(1); }
  f.read((char*)v.data(), n * 4); return v;
}
static float worst(const std::vector<float>& a, const float* b, size_t n) {
  float w = 0; for (size_t i = 0; i < n; ++i) w = std::max(w, std::fabs(a[i] - b[i])); return w;
}

int main(int argc, char** argv) {
  std::string RF = argc > 1 ? argv[1] : "pure/ref/"; if (RF.back() != '/') RF += '/';
  SamW w = load_sam_decoder(RF);
  Tensor img = from_data({1, 256, 64, 64}, readbin(RF + "emb_in.bin", 256 * 64 * 64));
  std::vector<float> pts = {420.f, 380.f}; std::vector<int> labels = {1};   // one fg point
  SamOut o = sam_decode(img, pts, labels, w);

  auto sref = readbin(RF + "sparse_ref.bin", 2 * 256);
  printf("sparse   worst = %.3e\n", worst(o.sparse->data, sref.data(), 2 * 256));

  auto iref = readbin(RF + "iou_ref.bin", 3);                 // multimask = C++ iou[1..3]
  std::vector<float> iou3 = {o.iou->data[1], o.iou->data[2], o.iou->data[3]};
  printf("iou pure = [%.5f %.5f %.5f]  ref = [%.5f %.5f %.5f]  worst = %.3e\n",
         iou3[0], iou3[1], iou3[2], iref[0], iref[1], iref[2], worst(iou3, iref.data(), 3));

  auto mref = readbin(RF + "masks_ref.bin", 3 * 256 * 256);   // multimask = C++ masks channels[1..3]
  std::vector<float> m3(o.masks->data.begin() + 256 * 256, o.masks->data.begin() + 4 * 256 * 256);
  float mw = worst(m3, mref.data(), 3 * 256 * 256);
  double mean_ref = 0; for (float v : mref) mean_ref += std::fabs(v); mean_ref /= mref.size();
  printf("masks    worst = %.3e  (mean|ref| %.3f)  %s\n", mw, mean_ref, mw < 1e-3 ? "MATCH" : "MISMATCH");
  return 0;
}

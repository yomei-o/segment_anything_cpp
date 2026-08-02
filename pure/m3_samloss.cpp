// Gradcheck SAM mask_loss (focal + dice) and sanity-check it decreases under SGD.
//   build: cl /std:c++20 /O2 /EHsc /Zc:preprocessor /DNOMINMAX pure\m3_samloss.cpp
#include "autograd.hpp"
#include "sam_loss.hpp"
#include "optim.hpp"
#include <cstdio>
#include <vector>
#include <cmath>
#include <random>

int main() {
  std::mt19937 rng(0); std::normal_distribution<float> nd(0, 1); std::uniform_real_distribution<float> ud(0, 1);
  const int64_t H = 6, W = 7, N = H * W;
  std::vector<float> zd(N), gt(N);
  for (int64_t i = 0; i < N; ++i) { zd[i] = nd(rng); gt[i] = ud(rng) < 0.4f ? 1.f : 0.f; }

  Tensor z = from_data({1, 1, H, W}, zd, true);
  Tensor L = mask_loss(z, gt);
  std::fill(z->grad.begin(), z->grad.end(), 0.f); backward(L);
  std::vector<float> ana(z->grad.begin(), z->grad.end());
  float eps = 1e-3f, worst = 0;
  for (int64_t i = 0; i < N; ++i) {
    float o = z->data[i];
    z->data[i] = o + eps; float lp = mask_loss(z, gt)->data[0];
    z->data[i] = o - eps; float lm = mask_loss(z, gt)->data[0];
    z->data[i] = o;
    worst = std::max(worst, std::fabs((lp - lm) / (2 * eps) - ana[i]));
  }
  printf("mask_loss gradcheck: worst |num-ana| = %.2e  %s\n", worst, worst < 1e-2 ? "PASS" : "FAIL");

  Tensor q = from_data({1, 1, H, W}, zd, true);
  SGD opt({q}, 2.f);
  float l0 = mask_loss(q, gt)->data[0], i0 = mask_iou(q, gt);
  for (int s = 0; s < 150; ++s) { opt.zero_grad(); Tensor l = mask_loss(q, gt); backward(l); opt.step(); }
  float l1 = mask_loss(q, gt)->data[0], i1 = mask_iou(q, gt);
  printf("mask_loss %.3f -> %.3f   IoU %.3f -> %.3f   %s\n", l0, l1, i0, i1, (l1 < l0 * 0.5f && i1 > 0.9f) ? "LEARNS" : "check");
  return 0;
}

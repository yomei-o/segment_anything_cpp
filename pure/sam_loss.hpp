// SAM training losses. mask_loss = focal + dice on mask logits vs a binary ground-truth mask (SAM's
// mask supervision, focal:dice = 20:1). iou_loss = MSE(pred IoU, actual mask-vs-gt IoU). Both fused
// with analytic backward (deterministic in the prediction -> gradcheckable).
#pragma once
#include "autograd.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

// binary focal (alpha=0.25, gamma=2, mean over pixels) + dice, on logits z[N] vs gt g[N] in {0,1}.
inline Tensor mask_loss(const Tensor& z, const std::vector<float>& g, float focal_w = 20.f, float dice_w = 1.f) {
  int64_t N = z->numel(); const float a = 0.25f, gm = 2.f, eps = 1e-6f;
  std::vector<float> p(N);
  double Sp = 0, Spt = 0, St = 0, focal = 0;
  for (int64_t i = 0; i < N; ++i) {
    float pi = 1.f / (1.f + std::exp(-z->data[i])); p[i] = pi;
    float t = g[i]; Sp += pi; St += t; Spt += pi * t;
    float pt = t * pi + (1 - t) * (1 - pi), at = t * a + (1 - t) * (1 - a);
    focal += -at * std::pow(1 - pt, gm) * std::log(std::max(pt, eps));
  }
  focal /= (double)N;
  double denom = Sp + St + eps, dice = 1.0 - (2.0 * Spt + eps) / denom;
  Tensor o = make_tensor({1}, z->requires_grad); o->data[0] = (float)(focal_w * focal + dice_w * dice);
  o->parents = {z};
  o->backward_fn = [z, o, g, N, a, gm, eps, focal_w, dice_w, Sp, Spt, St, denom, p] {
    float go = o->grad[0], num = (float)(2.0 * Spt + eps), den = (float)denom;
    for (int64_t i = 0; i < N; ++i) {
      float pi = p[i], t = g[i], dpz = pi * (1 - pi);
      // focal dFL/dp
      float dfl;
      if (t > 0.5f) { float om = 1 - pi;
        dfl = a * gm * std::pow(om, gm - 1) * std::log(std::max(pi, eps)) - a * std::pow(om, gm) / std::max(pi, eps); }
      else { dfl = -(1 - a) * gm * std::pow(pi, gm - 1) * std::log(std::max(1 - pi, eps)) + (1 - a) * std::pow(pi, gm) / std::max(1 - pi, eps); }
      float dfocal = focal_w * (dfl * dpz) / (float)N;
      // dice dD/dp = -(2*t*den - num)/den^2 ; then * dp/dz
      float ddice = dice_w * (-(2.f * t * den - num) / (den * den)) * dpz;
      z->grad[i] += (dfocal + ddice) * go;
    }
  };
  return o;
}

// (x - target)^2 for a 1-element tensor (IoU-head supervision)
inline Tensor sq_err(const Tensor& x, float target) {
  Tensor o = make_tensor({1}, x->requires_grad); float d = x->data[0] - target; o->data[0] = d * d;
  o->parents = {x}; o->backward_fn = [x, o, target] { x->grad[0] += 2.f * (x->data[0] - target) * o->grad[0]; };
  return o;
}

// actual IoU between sigmoid(logits)>0.5 and gt (for supervising the IoU head; returned as a constant)
inline float mask_iou(const Tensor& z, const std::vector<float>& g) {
  int64_t N = z->numel(); double inter = 0, uni = 0;
  for (int64_t i = 0; i < N; ++i) { bool pm = z->data[i] > 0.f, gm = g[i] > 0.5f;
    if (pm && gm) inter += 1; if (pm || gm) uni += 1; }
  return uni > 0 ? (float)(inter / uni) : 0.f;
}

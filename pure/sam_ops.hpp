// SAM-specific ops on the pure autograd engine.
//   layernorm2d — channels-first LayerNorm (normalize across C for each spatial location) used by the
//     mask-decoder upscaling + prompt-encoder mask path. torch LayerNorm2d, default eps 1e-6.
// Attention / positional-encoding assembly lives in net_sam.hpp (built from matmul/softmax/slices).
#pragma once
#include "autograd.hpp"
#include <vector>
#include <cmath>

// x:[N,C,H,W] -> normalize over C at each (n,h,w); affine g,b:[C]. Forward + backward (train-ready).
inline Tensor layernorm2d(const Tensor& x, const Tensor& g, const Tensor& b, float eps = 1e-6f) {
  int64_t N = x->shape[0], C = x->shape[1], HW = x->shape[2] * x->shape[3];
  auto o = make_tensor(x->shape, true);
  auto mean = std::make_shared<std::vector<float>>(N * HW);
  auto istd = std::make_shared<std::vector<float>>(N * HW);
  for (int64_t n = 0; n < N; ++n) for (int64_t p = 0; p < HW; ++p) {
    double m = 0; for (int64_t c = 0; c < C; ++c) m += x->data[(n * C + c) * HW + p]; m /= C;
    double v = 0; for (int64_t c = 0; c < C; ++c) { double d = x->data[(n * C + c) * HW + p] - m; v += d * d; } v /= C;
    float is = (float)(1.0 / std::sqrt(v + eps));
    (*mean)[n * HW + p] = (float)m; (*istd)[n * HW + p] = is;
    for (int64_t c = 0; c < C; ++c) {
      float xh = (float)((x->data[(n * C + c) * HW + p] - m) * is);
      o->data[(n * C + c) * HW + p] = xh * g->data[c] + b->data[c];
    }
  }
  o->parents = {x, g, b}; Node* op = o.get();
  o->backward_fn = [x, g, b, op, N, C, HW, mean, istd] {
    for (int64_t n = 0; n < N; ++n) for (int64_t p = 0; p < HW; ++p) {
      float is = (*istd)[n * HW + p], m = (*mean)[n * HW + p];
      // grads wrt g,b and accumulate for x (layernorm backward over the C axis)
      double sum_dy = 0, sum_dy_xh = 0;
      for (int64_t c = 0; c < C; ++c) {
        float go = op->grad[(n * C + c) * HW + p];
        float xh = (x->data[(n * C + c) * HW + p] - m) * is;
        g->grad[c] += go * xh; b->grad[c] += go;
        float dy = go * g->data[c];
        sum_dy += dy; sum_dy_xh += dy * xh;
      }
      for (int64_t c = 0; c < C; ++c) {
        float go = op->grad[(n * C + c) * HW + p];
        float xh = (x->data[(n * C + c) * HW + p] - m) * is;
        float dy = go * g->data[c];
        x->grad[(n * C + c) * HW + p] += is * (dy - (float)(sum_dy / C) - xh * (float)(sum_dy_xh / C));
      }
    }
  };
  return o;
}

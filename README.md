# segment_anything_cpp — pure C++ Segment Anything (no PyTorch/CMake) — WIP

[Segment Anything](https://github.com/facebookresearch/segment-anything) (promptable segmentation)
ported to a dependency-free C++ autograd engine — **inference + training**, no Python at run time —
same approach as the sibling repos ([facenet_cpp](https://github.com/yomei-o/facenet_cpp),
[lpr_cpp](https://github.com/yomei-o/lpr_cpp), [depth_anything_cpp](https://github.com/yomei-o/depth_anything_cpp)).
Extracted from the real model (no guessing).

SAM = **image encoder** (heavy, run once) + **prompt encoder** + **mask decoder** (point/box → mask).
The prompt encoder + mask decoder are **shared across all variants**; only the image encoder differs:
- **MobileSAM** — TinyViT encoder (~6M), browser-friendly (interactive click-to-segment demo). ✅
- **SAM ViT-B** — plain ViT encoder (91M) + decomposed rel-pos. ✅

Reference = `mobile_sam.pt` (TinyViT encoder + SAM's original prompt encoder + mask decoder, ~10M).

## 🖱️ Live demo — [**yomei-o.github.io/segment_anything_cpp/wasm/**](https://yomei-o.github.io/segment_anything_cpp/wasm/)
Pick an image → **Encode** (runs the ViT encoder once, in-browser) → **click any object** to segment it.
Pure C++ → WebAssembly, no server, no upload. Ships fp16 weights (~22 MB).

## Status — MobileSAM complete (matches PyTorch)
1. ✅ shared **prompt encoder + mask decoder** — parity vs PyTorch (masks 7.8e-5, IoU 4.5e-7)
2. ✅ **TinyViT** (2.86e-6) **& SAM ViT-B** (2.98e-6) image encoders — both MATCH
3. ✅ **end-to-end inference** (`pure/infer_sam.cpp`) — validated vs PyTorch SamPredictor (IoU + mask area identical)
4. ✅ **WASM click-to-segment demo** (`wasm/`) — encode once (~15–60 s), decode per click (~2–3 s)
5. ✅ **training** (focal + dice + IoU MSE, gradchecked) ; 6. ✅ **GPU/Colab notebook** (cuBLAS seam; real T4 run pending)

Build (weights: `python pure/ref/export_sam.py && python pure/ref/export_tinyvit.py` first, needs
`mobile_sam.pt`): `cl /std:c++20 /O2 /EHsc /Zc:preprocessor /DNOMINMAX /Ipure\third_party pure\infer_sam.cpp`
then `infer_sam <img> <x> <y> [out.png]` (x,y = click in original pixels).

Shared engine reused from depth_anything_cpp: `autograd/backend/ops2d/linalg/bn/optim/depth_ops/
face_ops/...` + stb + flat Eigen. SAM-specific: `sam_ops.hpp`, `net_sam.hpp`, `pure/ref/*`.

License: own code BSD-3-Clause; bundled deps keep their licenses.

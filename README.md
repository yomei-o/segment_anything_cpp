# segment_anything_cpp — pure C++ Segment Anything (no PyTorch/CMake) — WIP

[Segment Anything](https://github.com/facebookresearch/segment-anything) (promptable segmentation)
ported to a dependency-free C++ autograd engine — **inference + training**, no Python at run time —
same approach as the sibling repos ([facenet_cpp](https://github.com/yomei-o/facenet_cpp),
[lpr_cpp](https://github.com/yomei-o/lpr_cpp), [depth_anything_cpp](https://github.com/yomei-o/depth_anything_cpp)).
Extracted from the real model (no guessing).

SAM = **image encoder** (heavy, run once) + **prompt encoder** + **mask decoder** (point/box → mask).
The prompt encoder + mask decoder are **shared across all variants**; only the image encoder differs:
- **MobileSAM** — TinyViT encoder (~6M), browser-friendly (interactive click-to-segment demo).
- **SAM ViT-B** — plain ViT encoder (91M), reuses the depth_anything DINOv2 ViT ops.

Reference = `mobile_sam.pt` (TinyViT encoder + SAM's original prompt encoder + mask decoder, ~10M).

## Status — building the shared decoder first
1. shared **prompt encoder + mask decoder** → parity vs PyTorch  ← in progress
2. image encoders: MobileSAM TinyViT + SAM ViT-B → embedding parity
3. end-to-end inference (image + click → mask overlay)
4. WASM interactive demo (encoder once → decoder per click)
5. training (focal + dice + IoU) ; 6. GPU (cuBLAS seam)

Shared engine reused from depth_anything_cpp: `autograd/backend/ops2d/linalg/bn/optim/depth_ops/
face_ops/...` + stb + flat Eigen. SAM-specific: `sam_ops.hpp`, `net_sam.hpp`, `pure/ref/*`.

License: own code BSD-3-Clause; bundled deps keep their licenses.

# RESUME — segment_anything_cpp remaining work

Pure-C++ Segment Anything (MobileSAM). SAM = image encoder (once) + prompt encoder + mask decoder.
The prompt encoder + mask decoder are shared across variants; only the image encoder differs.

## ✅ Stage 1 — shared decoder (DONE 2026-08-02)
`sam_ops.hpp` (layernorm2d, channels-first, fwd+bwd) + `net_sam.hpp` (prompt encoder + mask decoder:
PositionEmbeddingRandom, flexible attention, TwoWayTransformer depth-2, output upscaling +
hypernetwork masks + IoU head). `export_sam.py` extracts weights from `mobile_sam.pt` + parity refs.
**`m1_decoder.cpp` = sparse 2.4e-7, IoU 4.5e-7, masks 7.8e-5 MATCH vs PyTorch mobile_sam.**
Bugs found: MLP hidden dim is 2048 (not 4*E); layer-0 self-attn has no residual (queries replaced);
two-way MLP uses ReLU (not GELU).

## ✅ Stage 2 — image encoders (TinyViT DONE 2026-08-02; ViT-B pending)
- **MobileSAM TinyViT** (~6M): conv stem (patch_embed) → 4 stages (MBConv blocks + windowed attention
  with attention_biases + patch merging) → neck → [1,256,64,64]. New ops: MBConv (depthwise-sep, have
  conv2d groups), attention with a learned relative-bias table, windowing. Parity vs PyTorch.
- **SAM ViT-B** (91M): plain ViT (16x16 patch, 1024 input), windowed attn + decomposed rel-pos, neck.
  Reuses the depth_anything DINOv2 ViT ops closely.

## ✅ Stage 3 — end-to-end inference (DONE 2026-08-02, validated vs SamPredictor)
CLI: image → resize/pad 1024 → encoder → embedding → point/box → `sam_decode` → mask → overlay PNG.
(Preprocessing: ResizeLongestSide to 1024 + pad; the SAM pixel mean/std normalization.)

## ✅ Stage 4 — WASM click-to-segment demo (DONE 2026-08-02)
Encoder runs once per image (heavy), decoder per click (cheap, ~ms). Click → segment. Ship fp16.

## ⏭ Stage 5 — training (focal + dice mask loss + IoU MSE) ; Stage 6 — GPU (cuBLAS seam)

## Notes
- B=1 throughout. embed 256, image embedding 64x64, input 1024, 4 mask tokens, 8 heads, tf depth 2.
- Weights from `mobile_sam.pt` (pip install git+https://github.com/ChaoningZhang/MobileSAM.git).
  MobileSAM's decoder == SAM's original decoder (only the encoder is distilled/replaced).
- Linear weights exported transposed [in,out]; ConvTranspose2d [in,out,k,k]; token LN eps 1e-5,
  LayerNorm2d eps 1e-6. weights.bin (16 MB) gitignored — regen via export_sam.py (needs mobile_sam.pt).

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

## ✅ Stage 5 — training (DONE 2026-08-02)
`sam_loss.hpp` (focal+dice mask_loss, gradcheck 1.83e-4; sq_err IoU MSE) + `train_sam.cpp` (SAM recipe:
supervise min-loss mask; Adam on 4.06M decoder params, encoder frozen). Synthetic: mask_loss 2.13->0.89.

## ✅ Stage 6 — GPU notebook (DONE 2026-08-02; real Colab run pending)
`colab_sam_gpu.ipynb`: build CPU + GPU (nvcc -DUSE_CUDA -lcublas, cuBLAS seam) + time + train. infer_sam
/ train_sam verified c++17-clean (nvcc uses c++17). Real T4 run is the user's final step.

## SAM ViT-B encoder (the other 'both' half)
`net_vitb.hpp` + `export_vitb.py` + `m4_vitb.cpp`: plain ViT (patch16, embed768, 12 blocks, windowed 14
with global at 2/5/8/11), abs pos_embed, **decomposed relative position embeddings** (MViTv2, new),
neck. sam_vit_b.pth (~375MB, gitignored). Parity validating (slow: global blocks attend over 4096 tokens).

## Notes
- B=1 throughout. embed 256, image embedding 64x64, input 1024, 4 mask tokens, 8 heads, tf depth 2.
- Weights from `mobile_sam.pt` (pip install git+https://github.com/ChaoningZhang/MobileSAM.git).
  MobileSAM's decoder == SAM's original decoder (only the encoder is distilled/replaced).
- Linear weights exported transposed [in,out]; ConvTranspose2d [in,out,k,k]; token LN eps 1e-5,
  LayerNorm2d eps 1e-6. weights.bin (16 MB) gitignored — regen via export_sam.py (needs mobile_sam.pt).

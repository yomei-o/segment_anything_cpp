# Segment Anything (MobileSAM) — architecture for the C++ port

Extracted from `mobile_sam.pt` via `inspect_sam.py`/`dump_shared.py`. B=1 throughout (interactive).
embed_dim=256, image_embedding=64×64×256, input_image=1024×1024, num_mask_tokens=4 (1 + 3 multimask).

## Shared: PromptEncoder (0.01M) + MaskDecoder (4.06M)  — same for MobileSAM & SAM ViT-*

### PositionEmbeddingRandom (pe_layer, num_pos_feats=128)
buffer `positional_encoding_gaussian_matrix` G[2,128].
`_pe(coords[.,2]∈[0,1])` = cat(sin, cos)(2π·((2·coords−1) @ G))  → [.,256].
- image_pe (get_dense_pe): grid 64×64, x=(col_cumsum−0.5)/64, y=(row_cumsum−0.5)/64 → _pe → [256,64,64]. CONSTANT.
- points: (pt+0.5)/1024 → _pe ; then += point_embeddings[label] (0=bg,1=fg,2/3=box corners); label −1 (pad) = not_a_point_embed.

### PromptEncoder.forward (points, no box/mask = common case)
sparse = _embed_points(coords,labels,pad=True): append pad point (0,0,label −1) → [N+1,256].
dense  = no_mask_embed broadcast → [256,64,64].   (mask path: mask_downscaling convs — implemented but unused in demo)

### MaskDecoder.predict_masks
tokens = cat(iou_token[1,256], mask_tokens[4,256], sparse[N+1,256]) = [5+N,256].
src = image_embed[256,64,64] + dense ; pos_src = image_pe.
TwoWayTransformer(depth=2, heads=8): flatten image → [4096,256]; layers; then final_attn_token_to_image + norm_final_attn.
  TwoWayAttentionBlock(queries q, keys k, query_pe, key_pe):
    1. self_attn: layer0 q=k=v=queries (skip pe); else q=k=queries+query_pe,v=queries; queries+=attn; norm1(eps1e-5)
    2. cross token→image: q=queries+query_pe, k=keys+key_pe, v=keys; queries+=attn; norm2
    3. mlp: lin2(GELU(lin1)) 256→2048→256; queries+=mlp; norm3
    4. cross image→token: q=keys+key_pe, k=queries+query_pe, v=queries; keys+=attn; norm4
  Attention(q,k,v): sep q/kv proj (self internal=256; cross internal=128), 8 heads, softmax(qkᵀ/√cph)v, out_proj.
hs=queries, src=keys. iou_token_out=hs[0]; mask_tokens_out=hs[1:5].
src→[256,64,64]; output_upscaling: ConvT(256→64,2,2)→LayerNorm2d(eps1e-6)→GELU→ConvT(64→32,2,2)→GELU → [32,256,256].
hyper_in[i]=MLP3(256→256→256→32, relu)(mask_tokens_out[i]); masks = hyper_in[4,32] @ upscaled[32,256²] → [4,256,256].
iou_pred = MLP3(256→256→256→4, relu)(iou_token_out).  multimask: masks[1:4], iou[1:4].

## New C++ ops vs depth_anything
- **layernorm2d** (channels-first LN over C of [1,C,H,W], eps1e-6) — NEW, gradcheck.
- flexible **attention(q,k,v, qkvo weights, heads, internal_dim)** — assemble from matmul/softmax/slice_cols/hcat.
- pos-enc (sin/cos of coords@G) — computed as CONSTANTS (no autograd needed at inference).
Reuse: layernorm(eps1e-5 for token norms), gelu, relu, matmul, conv2d, convtranspose2d(w=[in,out,k,k]).

## Weights (mobile_sam.pt), torch layouts
Linear W=[out,in] → export transposed [in,out] (C++ matmul(x[N,in],W[in,out])). Conv2d [out,in,k,k].
ConvTranspose2d [in,out,k,k] (matches C++ convtranspose2d). Embedding [n,256].

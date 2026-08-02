# Extract the SHARED PromptEncoder + MaskDecoder from mobile_sam.pt into a forward-order weight blob
# + config, and save parity refs (a fixed image_embedding + a point prompt -> masks, iou) computed by
# the real PyTorch modules. B=1. Linear weights are stored transposed [in,out] (C++ matmul(x,W)).
#   python export_sam.py
import os, numpy as np, torch
from mobile_sam import sam_model_registry
HERE = os.path.dirname(__file__)
sam = sam_model_registry["vit_t"](checkpoint=os.path.join(HERE, "mobile_sam.pt")); sam.eval()
pe, md = sam.prompt_encoder, sam.mask_decoder

blob = bytearray()
def put(a): blob.extend(np.ascontiguousarray(np.asarray(a, np.float32)).ravel().tobytes())
def P(t): return t.detach().cpu().numpy()
def lin(m):  # torch Linear W[out,in] -> [in,out], bias
    put(P(m.weight).T); put(P(m.bias))
def conv(m): put(P(m.weight)); put(P(m.bias))       # [out,in,k,k]
def ln(m):   put(P(m.weight)); put(P(m.bias))        # LayerNorm / LayerNorm2d affine
def emb(m):  put(P(m.weight))                         # [n,256]

# ---- PROMPT ENCODER ----
put(P(pe.pe_layer.positional_encoding_gaussian_matrix))      # G [2,128]
for i in range(4): emb(pe.point_embeddings[i])              # [1,256]
emb(pe.not_a_point_embed); emb(pe.no_mask_embed)
mds = pe.mask_downscaling
conv(mds[0]); ln(mds[1]); conv(mds[3]); ln(mds[4]); conv(mds[6])

# ---- MASK DECODER ----
emb(md.iou_token); emb(md.mask_tokens)
def attn(a): lin(a.q_proj); lin(a.k_proj); lin(a.v_proj); lin(a.out_proj)
for layer in md.transformer.layers:
    attn(layer.self_attn); ln(layer.norm1)
    attn(layer.cross_attn_token_to_image); ln(layer.norm2)
    lin(layer.mlp.lin1); lin(layer.mlp.lin2); ln(layer.norm3)
    ln(layer.norm4); attn(layer.cross_attn_image_to_token)
attn(md.transformer.final_attn_token_to_image); ln(md.transformer.norm_final_attn)
up = md.output_upscaling
put(P(up[0].weight)); put(P(up[0].bias))                    # ConvT [256,64,2,2]
ln(up[1])                                                   # LayerNorm2d(64)
put(P(up[3].weight)); put(P(up[3].bias))                    # ConvT [64,32,2,2]
for m in md.output_hypernetworks_mlps:
    for l in m.layers: lin(l)
for l in md.iou_prediction_head.layers: lin(l)

open(os.path.join(HERE, "weights.bin"), "wb").write(blob)
np.frombuffer(bytes(blob), np.float32).astype(np.float16).tofile(os.path.join(HERE, "weights_fp16.bin"))
with open(os.path.join(HERE, "config.txt"), "w") as f:
    f.write("embed_dim 256\nimage_embed 64\ninput_image 1024\nnum_mask_tokens 4\nheads 8\ntf_depth 2\n")

# ---- parity refs: fixed image embedding + a point prompt ----
torch.manual_seed(0)
img_embed = torch.sin(torch.arange(1*256*64*64, dtype=torch.float32).reshape(1,256,64,64)*0.001)
coords = torch.tensor([[[420.0, 380.0]]])   # one foreground click (x,y) in 1024 space
labels = torch.tensor([[1]])
with torch.no_grad():
    sparse, dense = pe(points=(coords, labels), boxes=None, masks=None)
    image_pe = pe.get_dense_pe()
    masks, iou = md(image_embeddings=img_embed, image_pe=image_pe,
                    sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=True)
P(img_embed).tofile(os.path.join(HERE, "emb_in.bin"))
np.array([[420.0,380.0]], np.float32).tofile(os.path.join(HERE, "point.bin"))
P(sparse).tofile(os.path.join(HERE, "sparse_ref.bin"))     # [1,2,256]
P(image_pe).tofile(os.path.join(HERE, "imgpe_ref.bin"))    # [1,256,64,64]
P(masks).tofile(os.path.join(HERE, "masks_ref.bin"))       # [1,3,256,256]
P(iou).tofile(os.path.join(HERE, "iou_ref.bin"))           # [1,3]
print(f"weights.bin {len(blob)/1e6:.2f} MB")
print("sparse", tuple(sparse.shape), "masks", tuple(masks.shape), "iou", P(iou).ravel())

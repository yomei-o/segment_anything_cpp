# Extract the SAM ViT-B image encoder into a forward-order weight blob + staged parity refs.
# Plain ViT: patch_embed(conv16) + abs pos_embed + 12 blocks (windowed 14; global at 2,5,8,11) with
# decomposed relative position embeddings, + neck. Linear weights stored transposed [in,out]. 1024->[1,256,64,64].
import os, numpy as np, torch
from segment_anything import sam_model_registry
HERE = os.path.dirname(__file__)
sam = sam_model_registry["vit_b"](checkpoint=os.path.join(HERE, "sam_vit_b.pth")); sam.eval()
enc = sam.image_encoder
blob = bytearray()
def put(a): blob.extend(np.ascontiguousarray(np.asarray(a, np.float32)).ravel().tobytes())
def P(t): return t.detach().cpu().numpy()
def lin(m): put(P(m.weight).T); put(P(m.bias))
def ln(m): put(P(m.weight)); put(P(m.bias))

put(P(enc.patch_embed.proj.weight)); put(P(enc.patch_embed.proj.bias))    # conv [768,3,16,16]
put(P(enc.pos_embed))                                                     # [1,64,64,768]
for b in enc.blocks:
    ln(b.norm1); lin(b.attn.qkv); lin(b.attn.proj)
    put(P(b.attn.rel_pos_h)); put(P(b.attn.rel_pos_w))                    # [2*S-1, 64]
    ln(b.norm2); lin(b.mlp.lin1); lin(b.mlp.lin2)
put(P(enc.neck[0].weight)); ln(enc.neck[1]); put(P(enc.neck[2].weight)); ln(enc.neck[3])
open(os.path.join(HERE, "vitb_weights.bin"), "wb").write(blob)
np.frombuffer(bytes(blob), np.float32).astype(np.float16).tofile(os.path.join(HERE, "vitb_weights_fp16.bin"))

# staged refs
x = torch.sin(torch.arange(1*3*1024*1024, dtype=torch.float32).reshape(1,3,1024,1024)*0.0005)
feats = {}
with torch.no_grad():
    y = enc.patch_embed(x); y = y + enc.pos_embed; feats['pe_pos'] = P(y)   # [1,64,64,768]
    for i, blk in enumerate(enc.blocks):
        y = blk(y)
        if i in (0, 2, 11): feats[f'block{i}'] = P(y)
    y = enc.neck(y.permute(0, 3, 1, 2)); feats['final'] = P(y)
    emb = enc(x); feats['enc'] = P(emb)
P(x).tofile(os.path.join(HERE, "vitb_in.bin"))
for k, v in feats.items(): v.tofile(os.path.join(HERE, f"vitb_{k}.bin"))
print(f"vitb_weights.bin {len(blob)/1e6:.1f} MB")
for k, v in feats.items(): print(f"  {k:10s} {tuple(v.shape)}")
print("final vs enc:", np.abs(feats['final']-feats['enc']).max())

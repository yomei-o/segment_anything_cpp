# Extract the MobileSAM TinyViT image encoder into a forward-order weight blob + staged parity refs.
# Conv2d_BN layers are BN-fused into biased convs. Attention biases are exported pre-expanded
# (ab = attention_biases[:, attention_bias_idxs], shape [heads,N,N]). B=1, input 1024 -> [1,256,64,64].
#   python export_tinyvit.py
import os, numpy as np, torch, itertools
from mobile_sam import sam_model_registry
HERE = os.path.dirname(__file__)
sam = sam_model_registry["vit_t"](checkpoint=os.path.join(HERE, "mobile_sam.pt")); sam.eval()
enc = sam.image_encoder

blob = bytearray()
def put(a): blob.extend(np.ascontiguousarray(np.asarray(a, np.float32)).ravel().tobytes())
def P(t): return t.detach().cpu().numpy()
def cbn(m):                       # fused Conv2d_BN -> biased conv weights
    f = m.fuse(); put(P(f.weight)); put(P(f.bias))
def conv_nobias(m): put(P(m.weight))     # raw Conv2d (neck), no bias
def lin(m): put(P(m.weight).T); put(P(m.bias))
def ln(m):  put(P(m.weight)); put(P(m.bias))

# patch_embed
cbn(enc.patch_embed.seq[0]); cbn(enc.patch_embed.seq[2])
# layers
for li, layer in enumerate(enc.layers):
    for blk in layer.blocks:
        if hasattr(blk, 'conv1'):          # MBConv
            cbn(blk.conv1); cbn(blk.conv2); cbn(blk.conv3)
        else:                              # TinyViTBlock
            a = blk.attn
            ln(a.norm); lin(a.qkv); lin(a.proj)
            ab = a.attention_biases[:, a.attention_bias_idxs]   # [heads,N,N]
            put(P(ab))
            cbn(blk.local_conv)
            ln(blk.mlp.norm); lin(blk.mlp.fc1); lin(blk.mlp.fc2)
    if layer.downsample is not None:
        d = layer.downsample; cbn(d.conv1); cbn(d.conv2); cbn(d.conv3)
# neck
conv_nobias(enc.neck[0]); ln(enc.neck[1]); conv_nobias(enc.neck[2]); ln(enc.neck[3])

open(os.path.join(HERE, "tvit_weights.bin"), "wb").write(blob)
np.frombuffer(bytes(blob), np.float32).astype(np.float16).tofile(os.path.join(HERE, "tvit_weights_fp16.bin"))

# ---- staged parity refs (hook the forward) ----
torch.manual_seed(0)
x = torch.sin(torch.arange(1*3*1024*1024, dtype=torch.float32).reshape(1,3,1024,1024)*0.0005)
feats = {}
with torch.no_grad():
    y = enc.patch_embed(x); feats['patch_embed'] = P(y)
    for i, layer in enumerate(enc.layers):
        y = layer(y); feats[f'layer{i}'] = P(y)
    # replicate the encoder tail: last layer output is tokens [B, L, C=320]; -> [B,320,64,64] -> neck
    B = y.shape[0]; C = y.shape[-1]
    y = y.view(B, 64, 64, C).permute(0, 3, 1, 2)
    y = enc.neck(y); feats['neck'] = P(y)
    emb = enc(x); feats['final'] = P(emb)
P(x).tofile(os.path.join(HERE, "tvit_in.bin"))
for k, v in feats.items(): v.tofile(os.path.join(HERE, f"tvit_{k}.bin"))
with open(os.path.join(HERE, "tvit_config.txt"), "w") as f:
    f.write("img 1024\nembed 256\nout_res 64\n")
print(f"tvit_weights.bin {len(blob)/1e6:.2f} MB")
for k, v in feats.items(): print(f"  {k:12s} {tuple(v.shape)}")
print("final vs neck same?", np.abs(feats['final']-feats['neck']).max())

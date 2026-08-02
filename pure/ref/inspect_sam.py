import torch, sys
from mobile_sam import sam_model_registry
sam = sam_model_registry["vit_t"](checkpoint="mobile_sam.pt"); sam.eval()
def summ(mod, name):
    ps = list(mod.named_parameters())
    n = sum(p.numel() for _,p in ps)
    print(f"\n===== {name}: {n/1e6:.2f}M params, {len(ps)} tensors =====")
    return ps
for nm, mod in [("image_encoder (TinyViT)", sam.image_encoder),
                ("prompt_encoder", sam.prompt_encoder),
                ("mask_decoder", sam.mask_decoder)]:
    ps = summ(mod, nm)
print("\n----- config -----")
print("image_size", sam.image_encoder.img_size if hasattr(sam.image_encoder,'img_size') else '?')
pe = sam.prompt_encoder
print("prompt embed_dim", pe.embed_dim, "image_embedding_size", pe.image_embedding_size, "input_image_size", pe.input_image_size)
md = sam.mask_decoder
print("mask_decoder num_mask_tokens", md.num_mask_tokens, "transformer_dim", md.transformer_dim if hasattr(md,'transformer_dim') else '?')

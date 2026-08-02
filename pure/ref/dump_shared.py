import torch
from mobile_sam import sam_model_registry
sam = sam_model_registry["vit_t"](checkpoint="mobile_sam.pt"); sam.eval()
print("################## PROMPT ENCODER (structure) ##################")
print(sam.prompt_encoder)
print("\n-- prompt_encoder params --")
for n,p in sam.prompt_encoder.named_parameters(): print(f"  {n:52s} {tuple(p.shape)}")
# buffers (pos enc matrix is a buffer, not a param!)
print("-- prompt_encoder buffers --")
for n,b in sam.prompt_encoder.named_buffers(): print(f"  {n:52s} {tuple(b.shape)}")
print("\n################## MASK DECODER (structure) ##################")
print(sam.mask_decoder)

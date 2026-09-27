# Reference for the Rust HGNetv2 port: D-FINE-N's backbone (COCO weights), eval mode, on a fixed
# input. Saves the weights, the input and both returned feature maps in one safetensors file.
import sys, math, torch
sys.path.insert(0, "/workspace/DEIM")
from engine.backbone.hgnetv2 import HGNetv2
from safetensors.torch import save_file
m = HGNetv2("B0", use_lab=True, return_idx=[2, 3], freeze_norm=False, pretrained=False).eval()
ck = torch.load("/workspace/weights/dfine_n_coco.pth", map_location="cpu", weights_only=False)
sd = ck["ema"]["module"] if "ema" in ck else ck["model"]
bk = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
missing, unexpected = m.load_state_dict(bk, strict=True), None
x = torch.tensor([math.sin(i * 0.013) * 0.5 + 0.5 for i in range(1 * 3 * 128 * 128)], dtype=torch.float32).view(1, 3, 128, 128)
with torch.no_grad():
    outs = m(x)
t = {f"backbone.{k}": v.float().contiguous() for k, v in bk.items() if "num_batches" not in k}
t["ref.input"] = x
for i, o in enumerate(outs):
    t[f"ref.out{i}"] = o.contiguous()
    print("out", i, tuple(o.shape), float(o.abs().mean()))
save_file(t, "/workspace/hgnetv2_ref.safetensors")
print("saved", len(t), "tensors")

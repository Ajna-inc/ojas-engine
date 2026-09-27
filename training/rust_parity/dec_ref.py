# Reference for the Rust D-FINE decoder: DFINETransformer with the production config (96 wide,
# 4 layers, points [6, 6], relu), every parameter perturbed off its init so the zero-initialised
# heads are exercised, training mode without denoising (all layers' outputs), fixed inputs.
import sys, math, torch
sys.path.insert(0, "/workspace/DEIM")
from engine.deim.dfine_decoder import DFINETransformer
from safetensors.torch import save_file
torch.manual_seed(0)
dec = DFINETransformer(num_classes=15, hidden_dim=96, num_queries=300, feat_channels=[96, 96],
                       feat_strides=[16, 32], num_levels=2, num_points=[6, 6], nhead=8, num_layers=4,
                       dim_feedforward=384, activation="relu", mlp_act="relu", num_denoising=0,
                       reg_max=32, reg_scale=4.0, eval_idx=-1, layer_scale=1)
with torch.no_grad():
    for n, p in dec.named_parameters():
        if n not in ("up", "reg_scale"):
            p.add_(torch.randn_like(p) * 0.05)
dec.train()
B = 2
f0 = torch.tensor([math.sin(i * 0.013) for i in range(B * 96 * 20 * 20)]).view(B, 96, 20, 20)
f1 = torch.tensor([math.cos(i * 0.009) for i in range(B * 96 * 10 * 10)]).view(B, 96, 10, 10)
with torch.no_grad():
    out = dec([f0, f1])
t = {f"decoder.{k}": v.float().clone().contiguous() for k, v in dec.state_dict().items()}
t["ref.in0"], t["ref.in1"] = f0, f1
layers = out["aux_outputs"] + [out]
for i, o in enumerate(layers):
    for key in ("pred_logits", "pred_boxes", "pred_corners", "ref_points"):
        t[f"ref.{key}.{i}"] = o[key].clone().contiguous()
for key in ("pred_logits", "pred_boxes"):
    t[f"ref.pre.{key}"] = out["pre_outputs"][key].clone().contiguous()
    t[f"ref.enc.{key}"] = out["enc_aux_outputs"][0][key].clone().contiguous()
print("layers", len(layers), "logits", tuple(out["pred_logits"].shape), "decoder params", sum(p.numel() for p in dec.parameters() if p.requires_grad))
save_file(t, "/workspace/dec_ref.safetensors")

# Reference for the Rust DEIM criterion: the production D-FINE decoder (perturbed weights, as in
# dec_ref.py) with contrastive denoising, DEIMCriterion as configs/base/deim.yml sets it, one
# backward. Saves the inputs, targets, the exact denoising group drawn, every weighted loss term
# and every decoder parameter's gradient.
import sys, math, torch
sys.path.insert(0, "/workspace/DEIM")
import engine.deim.dfine_decoder as dd
from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.matcher import HungarianMatcher
from safetensors.torch import save_file
torch.manual_seed(0)
NC = 15
dec = dd.DFINETransformer(num_classes=NC, hidden_dim=96, num_queries=300, feat_channels=[96, 96],
                          feat_strides=[16, 32], num_levels=2, num_points=[6, 6], nhead=8, num_layers=4,
                          dim_feedforward=384, activation="relu", mlp_act="relu", num_denoising=100,
                          label_noise_ratio=0.5, box_noise_scale=1.0, reg_max=32, reg_scale=4.0,
                          eval_idx=-1, layer_scale=1)
with torch.no_grad():
    for n, p in dec.named_parameters():
        if n not in ("up", "reg_scale"):
            p.add_(torch.randn_like(p) * 0.05)
    dec.denoising_class_embed.weight[NC].zero_()   # the padding row, as nn.Embedding keeps it
dec.train()

# capture the denoising group exactly as drawn
cap = {}
orig = dd.get_contrastive_denoising_training_group
class Rec(torch.nn.Module):
    def __init__(self, emb): super().__init__(); self.emb = emb
    def forward(self, idx): cap["classes"] = idx.clone(); return self.emb(idx)
def wrapped(targets, num_classes, num_queries, class_embed, **kw):
    r = orig(targets, num_classes, num_queries, Rec(class_embed), **kw)
    cap["boxes_unact"], cap["mask"], cap["meta"] = r[1].detach().clone(), r[2].clone(), r[3]
    return r
dd.get_contrastive_denoising_training_group = wrapped

B = 2
g = torch.Generator().manual_seed(1)
targets = []
for n in (6, 3):
    cxcy = torch.rand(n, 2, generator=g) * 0.7 + 0.15
    wh = torch.rand(n, 2, generator=g) * 0.25 + 0.05
    targets.append({"labels": torch.randint(0, NC, (n,), generator=g), "boxes": torch.cat([cxcy, wh], 1)})
f0 = torch.tensor([math.sin(i * 0.013) for i in range(B * 96 * 20 * 20)]).view(B, 96, 20, 20)
f1 = torch.tensor([math.cos(i * 0.009) for i in range(B * 96 * 10 * 10)]).view(B, 96, 10, 10)
out = dec([f0, f1], targets)

matcher = HungarianMatcher(weight_dict={"cost_class": 2, "cost_bbox": 5, "cost_giou": 2}, use_focal_loss=True, alpha=0.25, gamma=2.0)
crit = DEIMCriterion(matcher, weight_dict={"loss_mal": 1, "loss_bbox": 5, "loss_giou": 2, "loss_fgl": 0.15, "loss_ddf": 1.5},
                     losses=["mal", "boxes", "local"], alpha=0.75, gamma=1.5, num_classes=NC, reg_max=32)
losses = crit(out, targets)
total = sum(losses.values())
total.backward()

t = {f"decoder.{k}": v.float().clone().contiguous() for k, v in dec.state_dict().items()}
t["ref.in0"], t["ref.in1"] = f0, f1
for b, tg in enumerate(targets):
    t[f"ref.tgt{b}.labels"] = tg["labels"].float()
    t[f"ref.tgt{b}.boxes"] = tg["boxes"].clone()
t["ref.dn.classes"] = cap["classes"].float().contiguous()
t["ref.dn.boxes_unact"] = cap["boxes_unact"].contiguous()
t["ref.dn.mask"] = cap["mask"].float().contiguous()
t["ref.dn.num_group"] = torch.tensor([float(cap["meta"]["dn_num_group"])])
for b, p in enumerate(cap["meta"]["dn_positive_idx"]):
    t[f"ref.dn.pos{b}"] = p.float().contiguous()
for k, v in losses.items():
    t[f"loss.{k}"] = v.detach().reshape(1).float()
t["loss.total"] = total.detach().reshape(1)
for n, p in dec.named_parameters():
    if p.grad is not None:
        t[f"grad.decoder.{n}"] = p.grad.clone().contiguous()
print("terms", len(losses), "total", float(total), "dn", tuple(cap["classes"].shape), "groups", cap["meta"]["dn_num_group"])
save_file(t, "/workspace/loss_ref.safetensors")

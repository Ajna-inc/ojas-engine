# Reference for the Rust D-FINE encoder: HybridEncoder (version 'dfine') with the production
# config, seeded random weights and randomised BatchNorm statistics, eval mode, fixed inputs.
import sys, math, torch
sys.path.insert(0, "/workspace/DEIM")
from engine.deim.hybrid_encoder import HybridEncoder
from safetensors.torch import save_file
torch.manual_seed(0)
enc = HybridEncoder(in_channels=[512, 1024], feat_strides=[16, 32], hidden_dim=96, nhead=8,
                    dim_feedforward=384, dropout=0.0, enc_act="gelu", use_encoder_idx=[1],
                    num_encoder_layers=1, expansion=0.34, depth_mult=0.5, act="silu", version="dfine")
for m in enc.modules():
    if isinstance(m, torch.nn.BatchNorm2d):
        with torch.no_grad():
            m.running_mean.normal_(0, 0.1); m.running_var.uniform_(0.5, 1.5)
            m.weight.uniform_(0.8, 1.2); m.bias.normal_(0, 0.1)
enc.eval()
f0 = torch.tensor([math.sin(i * 0.011) for i in range(512 * 8 * 8)]).view(1, 512, 8, 8)
f1 = torch.tensor([math.cos(i * 0.007) for i in range(1024 * 4 * 4)]).view(1, 1024, 4, 4)
with torch.no_grad():
    outs = enc([f0, f1])
t = {f"encoder.{k}": v.float().contiguous() for k, v in enc.state_dict().items() if "num_batches" not in k}
t["ref.in0"], t["ref.in1"] = f0, f1
for i, o in enumerate(outs):
    t[f"ref.out{i}"] = o.contiguous()
    print("out", i, tuple(o.shape), float(o.abs().mean()))
print("encoder params", sum(p.numel() for p in enc.parameters()))
save_file(t, "/workspace/enc_ref.safetensors")

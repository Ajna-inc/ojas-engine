#!/usr/bin/env python3
"""PyTorch reference outputs for the ojas-cuda `cnn` kernel family.

Reference tooling only. Runs the upstream PyTorch CUDA kernels that
`crates/ojas-cuda/src/kernels/cnn.rs` ports on ragged shapes and adversarial
values (huge magnitudes, signed zeros, NaN, subnormals) and writes, per case,
raw little-endian tensors plus one line of `manifest.json`. The ignored test
`cnn_matches_torch_bitwise` in `crates/ojas-cuda/tests/conformance.rs` replays
every case through our kernels and demands identical bits.

  ~/.venvs/ojas-vision/bin/python scripts/release/torch_kernel_golden.py --out target/torch_golden
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

DTYPES = {"f32": torch.float32, "f16": torch.float16}


def values(shape, dtype, gen, nan=False):
    """randn*4 salted with the values that break sloppy ports."""
    x = torch.randn(shape, generator=gen, dtype=torch.float64) * 4
    flat = x.view(-1)
    big = 60000.0 if dtype == torch.float16 else 3e38
    specials = [0.0, -0.0, 1e-5, -1e-5, 20.0, -20.0, 40.0, -40.0, 800.0, -800.0, big, -big]
    if dtype == torch.float32:
        specials += [1e-40, -1e-40]
    idx = torch.randperm(flat.numel(), generator=gen)[: min(len(specials), flat.numel() // 4)]
    for i, v in zip(idx.tolist(), specials):
        flat[i] = v
    if nan:
        flat[torch.randperm(flat.numel(), generator=gen)[:3]] = float("nan")
    return x.to(dtype).cuda()


class Writer:
    def __init__(self, out: Path):
        self.out = out
        self.cases = []

    def case(self, name, kernel, dtype, tensors: dict, **params):
        files = {}
        for k, t in tensors.items():
            f = f"{name}.{k}.bin"
            t.detach().float().cpu().contiguous().numpy().tofile(self.out / f)  # exact for half
            files[k] = {"file": f, "shape": list(t.shape)}
        self.cases.append({"name": name, "kernel": kernel, "dtype": dtype, "tensors": files, **params})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="target/torch_golden")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(args.seed)
    w = Writer(out)

    for tag, dt in DTYPES.items():
        # conv epilogue: bias add then activation (act 0 none, 1 SiLU, 2 sigmoid)
        for act, fn in [(0, lambda t: t), (1, F.silu), (2, torch.sigmoid)]:
            x = values((3, 7, 13, 11), dt, gen)
            b = values((7,), dt, gen)
            y = fn(x + b.view(1, 7, 1, 1))
            w.case(f"bias_act{act}_{tag}", "cnn_bias_act", tag, {"x": x, "bias": b, "y": y},
                   plane=13 * 11, ch=7, act=act)
            x = values((5, 3, 17, 9), dt, gen)
            w.case(f"act{act}_{tag}", "cnn_act", tag, {"x": x, "y": fn(x)}, act=act)

        a, b = values((2, 6, 9, 7), dt, gen), values((2, 6, 9, 7), dt, gen)
        w.case(f"add_{tag}", "cnn_add", tag, {"a": a, "b": b, "y": a + b})

        # concat along axis 1 of three ragged inputs, and split back
        parts = [values((3, c, 5, 7), dt, gen) for c in (4, 1, 6)]
        cat = torch.cat(parts, dim=1)
        w.case(f"concat_{tag}", "cnn_concat", tag,
               {**{f"in{i}": t for i, t in enumerate(parts)}, "y": cat}, axis_lens=[4, 1, 6], inner=35)
        pieces = torch.split(cat, [4, 1, 6], dim=1)
        w.case(f"split_{tag}", "cnn_split", tag,
               {"x": cat, **{f"out{i}": t for i, t in enumerate(pieces)}}, axis_lens=[4, 1, 6], inner=35)

        for sf in (2, 3):
            x = values((2, 5, 9, 7), dt, gen)
            y = F.interpolate(x, scale_factor=sf, mode="nearest")
            w.case(f"upsample{sf}_{tag}", "cnn_upsample_nearest", tag, {"x": x, "y": y}, scale=sf)

        for k, s, pd, dl in [(5, 1, 2, 1), (3, 2, 1, 1), (3, 2, 1, 2), (2, 2, 0, 1)]:
            x = values((2, 4, 15, 12), dt, gen, nan=True)
            y = F.max_pool2d(x, k, s, pd, dl)
            w.case(f"maxpool_k{k}s{s}p{pd}d{dl}_{tag}", "cnn_maxpool", tag, {"x": x, "y": y},
                   k=k, s=s, pad=pd, dil=dl)

        # YOLO DFL shape class and ragged ones; dim < 64 keeps upstream serial.
        for shape in [(2, 16, 4, 8400), (3, 16, 4, 37), (5, 16, 3, 1), (4, 9, 70, 1)]:
            x = values(shape, dt, gen)
            y = torch.softmax(x, dim=1)
            w.case(f"softmax_{'x'.join(map(str, shape))}_{tag}", "cnn_softmax", tag, {"x": x, "y": y},
                   dim=shape[1], inner=shape[2] * shape[3])

        for kh, st, pd, dl, bias in [(3, 1, 1, 1, True), (3, 2, 1, 1, False), (5, 1, 2, 1, True), (3, 1, 2, 2, True)]:
            x = values((2, 6, 13, 10), dt, gen)
            wt = values((6, 1, kh, kh), dt, gen)
            bs = values((6,), dt, gen) if bias else None
            y = torch.ops.aten._conv_depthwise2d(x, wt, [kh, kh], bs, [st, st], [pd, pd], [dl, dl])
            tensors = {"x": x, "w": wt, "y": y}
            if bias:
                tensors["b"] = bs
            w.case(f"dwconv_k{kh}s{st}p{pd}d{dl}{'b' if bias else ''}_{tag}", "cnn_dwconv", tag, tensors,
                   k=kh, s=st, pad=pd, dil=dl)

    (out / "manifest.json").write_text(json.dumps({
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "seed": args.seed,
        "cases": w.cases,
    }, indent=1) + "\n")
    print(f"{len(w.cases)} cases -> {out}")


if __name__ == "__main__":
    main()

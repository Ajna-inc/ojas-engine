#!/usr/bin/env python3
"""Record Laya reference outputs for `examples/laya_gate.rs`.

A test-only client of the upstream checkpoint: it imports the checkpoint's own
`rl_common.py` and `rl_agent_api.py` and runs the PyTorch model in f32 on the CPU,
so the record depends on nothing in this repository. For every case in the case
file it stores, per question, the token ids and option-marker positions the
reference builds, the raw scorer logits, the calibrated probabilities (the
reference's arithmetic, before its 4-decimal rounding) and the act probability.

Usage:
    laya_reference.py <checkpoint dir> <cases.json> <out.json>

`<checkpoint dir>` holds `model.safetensors`, `encoder/`, `tokenizer/`,
`rl_agent_config.json`, `rl_common.py` and `rl_agent_api.py` (the repository root
of `convaiinnovations/laya`, or one of its subfolders plus those two files).
Requires torch, transformers, safetensors and numpy.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    ckpt, cases_path, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    code_dir = ckpt if (ckpt / "rl_common.py").exists() else ckpt.parent
    sys.path.insert(0, str(code_dir))
    from rl_agent_api import RLAgent  # noqa: E402  (the checkpoint's own code)
    from rl_common import QTYPES, build_sequence, collate_items, render_options, temp_bucket  # noqa: E402

    torch.manual_seed(0)
    agent = RLAgent(str(ckpt), device="cpu")
    cases = json.loads(cases_path.read_text())
    records = []
    for case in cases:
        state, questions = case["state"], case["questions"]
        items, internal = [], []
        for qid, qdef in questions.items():
            q = agent._to_internal(qdef)
            ids, markers = build_sequence(agent.tok, state, q, agent.cfg["max_len"], agent.cfg["head_max_len"])
            if len(markers) != len(render_options(q)):
                raise ValueError(f"{case['label']}/{qid}: options do not fit")
            items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]], "target": [0.0] * len(markers),
                          "label": -1, "episode": 0, "ep_step": 0, "ep_len": 1, "src": "reference"})
            internal.append((qid, q))
        b = collate_items([items], agent.tok.pad_token_id)
        with torch.no_grad():
            logits, act = agent.model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
        logits, act = logits.float().numpy(), torch.softmax(act.float(), -1).numpy()
        answers = []
        for r, (qid, q) in enumerate(internal):
            k = len(items[r]["markers"])
            qt = QTYPES[q["t"]]
            t = agent.temperature_by_options.get(temp_bucket(qt, k), agent.temperature[qt])
            z = logits[r, :k] / t
            p = np.exp(z - z.max())
            p = p / p.sum()
            answers.append({"id": qid, "ids": items[r]["ids"], "markers": items[r]["markers"],
                            "logits": logits[r, :k].tolist(), "temperature": float(t),
                            "probabilities": p.tolist(), "act_probability": float(act[r, 0])})
        records.append({"label": case["label"], "answers": answers})

    out = {
        "reference": "rl_agent_api.RLAgent (PyTorch, f32, CPU)",
        "torch": torch.__version__,
        "model_sha256": sha256(ckpt / "model.safetensors"),
        "cases": records,
    }
    out_path.write_text(json.dumps(out, indent=1))
    print(f"wrote {out_path}: {sum(len(c['answers']) for c in records)} questions over {len(records)} cases")


if __name__ == "__main__":
    main()

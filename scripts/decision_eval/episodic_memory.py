#!/usr/bin/env python3
"""Test an episodic memory beside an encoder decision model, with no training.

Reads the features `marker_features` writes (one hidden vector per option, the
option's raw score, the question's temperature and its right answer), splits every
family's questions into memory / dev / test, and compares on the test split:

* the model alone: softmax(score / T);
* the model plus the memory: every option's score gains `lam * m`, where `m` is the
  similarity-weighted answer of its nearest stored options (+1 - 1/n for a right
  option, -1/n for a wrong one), optionally scaled by the model's uncertainty on the
  question (1 - its top probability).

The memory holds every family's memory split together, so a query searches all of
it. `lam`, the neighbour count and the similarity temperature are chosen on dev.

    episodic_memory.py <features_dir> [--memory 0.6] [--dev 0.2] [--seed 3]
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(path):
    meta = json.loads((path / "meta.json").read_text())
    rows = [json.loads(line) for line in (path / "index.jsonl").open()]
    vectors = np.fromfile(path / "vectors.f32", dtype=np.float32).reshape(meta["options"], meta["width"])
    return rows, vectors


def split(rows, memory, dev, seed):
    by_family = defaultdict(list)
    for i, r in enumerate(rows):
        by_family[r["family"]].append(i)
    rng = random.Random(seed)
    parts = {"memory": [], "dev": [], "test": []}
    for family, idx in by_family.items():
        rng.shuffle(idx)
        a, b = int(len(idx) * memory), int(len(idx) * (memory + dev))
        parts["memory"] += idx[:a]
        parts["dev"] += idx[a:b]
        parts["test"] += idx[b:]
    return parts


def option_rows(rows, questions):
    """Flat option indices of `questions`, the question each belongs to, and each
    option's centred answer value."""
    opt, owner, value = [], [], []
    for q in questions:
        r = rows[q]
        n = len(r["scores"])
        for j in range(n):
            opt.append(r["first"] + j)
            owner.append(q)
            value.append((1.0 if j == r["gold"] else 0.0) - 1.0 / n)
    return np.array(opt), np.array(owner), np.array(value, dtype=np.float32)


def neighbours(queries, keys, k, chunk=4096):
    """Top-k cosine similarities and indices of every query among the keys."""
    sims = np.empty((len(queries), k), dtype=np.float32)
    idx = np.empty((len(queries), k), dtype=np.int64)
    for s in range(0, len(queries), chunk):
        block = queries[s:s + chunk] @ keys.T
        top = np.argpartition(-block, k, axis=1)[:, :k]
        val = np.take_along_axis(block, top, axis=1)
        order = np.argsort(-val, axis=1)
        idx[s:s + chunk] = np.take_along_axis(top, order, axis=1)
        sims[s:s + chunk] = np.take_along_axis(val, order, axis=1)
    return sims, idx


def evaluate(rows, questions, opt, memory_signal, lam, gated):
    """Accuracy per family, with `lam` times the memory signal added to the scores."""
    at = {o: i for i, o in enumerate(opt)}
    right = defaultdict(lambda: [0, 0])
    for q in questions:
        r = rows[q]
        z = np.array(r["scores"], dtype=np.float64) / r["temperature"]
        p = np.exp(z - z.max()); p /= p.sum()
        m = np.array([memory_signal[at[r["first"] + j]] for j in range(len(z))])
        weight = lam * (1.0 - p.max()) if gated else lam
        best = int(np.argmax(z + weight * m))
        right[r["family"]][0] += best == r["gold"]
        right[r["family"]][1] += 1
    return {f: a / n for f, (a, n) in right.items()}


def mean(d):
    return sum(d.values()) / len(d)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("features", type=Path)
    parser.add_argument("--memory", type=float, default=0.6)
    parser.add_argument("--dev", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    rows, vectors = load(args.features)
    parts = split(rows, args.memory, args.dev, args.seed)
    m_opt, _, m_val = option_rows(rows, parts["memory"])
    centre = vectors[m_opt].mean(axis=0)

    def keys_of(opt):
        x = vectors[opt] - centre
        return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-6)

    keys = keys_of(m_opt)
    print(f"memory: {len(parts['memory'])} questions, {len(m_opt)} options; dev {len(parts['dev'])}; test {len(parts['test'])}")

    signals = {}
    for name in ("dev", "test"):
        opt, _, _ = option_rows(rows, parts[name])
        sims, idx = neighbours(keys_of(opt), keys, 64)
        signals[name] = (opt, sims, idx)

    def signal(name, k, tau):
        opt, sims, idx = signals[name]
        w = np.exp((sims[:, :k] - sims[:, :1]) / tau)
        w /= w.sum(axis=1, keepdims=True)
        return opt, (w * m_val[idx[:, :k]]).sum(axis=1)

    base_dev = evaluate(rows, parts["dev"], *signal("dev", 1, 1.0), 0.0, False)
    best = (mean(base_dev), 0.0, 1, 1.0, False)
    for k in (4, 16, 64):
        for tau in (0.02, 0.05, 0.2):
            opt, m = signal("dev", k, tau)
            for gated in (False, True):
                for lam in (0.5, 1, 2, 4, 8, 16):
                    acc = mean(evaluate(rows, parts["dev"], opt, m, lam, gated))
                    if acc > best[0]:
                        best = (acc, lam, k, tau, gated)
    _, lam, k, tau, gated = best
    print(f"chosen on dev: lam {lam}, k {k}, tau {tau}, uncertainty-gated {gated}")

    opt, m = signal("test", k, tau)
    alone = evaluate(rows, parts["test"], opt, m, 0.0, False)
    with_memory = evaluate(rows, parts["test"], opt, m, lam, gated)
    memory_only = evaluate(rows, parts["test"], opt, m, 1e6, False)
    print(f"\n{'family':22s} {'model':>7s} {'+memory':>8s} {'change':>7s} {'memory only':>12s}")
    for f in sorted(alone):
        print(f"{f:22s} {alone[f]*100:6.1f}% {with_memory[f]*100:7.1f}% {(with_memory[f]-alone[f])*100:+6.1f} {memory_only[f]*100:11.1f}%")
    print(f"{'mean':22s} {mean(alone)*100:6.1f}% {mean(with_memory)*100:7.1f}% {(mean(with_memory)-mean(alone))*100:+6.1f} {mean(memory_only)*100:11.1f}%")


if __name__ == "__main__":
    main()

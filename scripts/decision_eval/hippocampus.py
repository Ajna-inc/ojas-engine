#!/usr/bin/env python3
"""A fact memory for decision models: stores knowledge as short passages, finds the
ones a request needs, and hands them to the model as background in the request's
state, where an encoder model reads them.

Sources, each optional: Simple English Wikipedia articles split into passages, the
OpenBookQA "open book" of science facts, SciQ support passages (training split
only), and Natural Earth country facts. Retrieval is BM25 over word unigrams and
bigrams, so it needs no model.

    hippocampus.py build <data_dir> <index_dir>
    hippocampus.py augment <index_dir> <in.jsonl|dir> <out_dir> [--k 3] [--keep]
    hippocampus.py show <index_dir> "query text"

`augment` rewrites `{"body": ..., "gold": ...}` lines (or bare request bodies) with
the state replaced by `{"background": [passages], "input": <original state>}`; with
`--keep` it leaves the request as it is and adds the passages as a `background` field
beside it, the form `decision_rl` trains with and without.
"""

import argparse
import glob
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

PASSAGE_WORDS = 70
STOP = "a an the of to in on for and or is are was were be been by with as at from that this it its which who what when where how why do does did not no yes".split()


def chunks(text, words=PASSAGE_WORDS):
    """`text` split at sentence ends into passages of about `words` words."""
    out, cur, n = [], [], 0
    for sentence in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text).strip()):
        w = len(sentence.split())
        if cur and n + w > words:
            out.append(" ".join(cur))
            cur, n = [], 0
        cur.append(sentence)
        n += w
    if cur:
        out.append(" ".join(cur))
    return out


def wikipedia(data):
    files = glob.glob(str(data / "wikimedia_wikipedia" / "**" / "*.parquet"), recursive=True)
    for f in files:
        for row in pd.read_parquet(f, columns=["title", "text"]).itertuples():
            for p in chunks(row.text)[:6]:
                if len(p.split()) >= 12:
                    yield f"{row.title}: {p}"


def open_book(data):
    for f in glob.glob(str(data / "allenai_openbookqa" / "additional" / "*.parquet")):
        for fact in pd.read_parquet(f)["fact1"].dropna().unique():
            yield fact.strip().capitalize() + "."


def sciq_support(data):
    for f in glob.glob(str(data / "allenai_sciq" / "**" / "train-*.parquet"), recursive=True):
        for s in pd.read_parquet(f)["support"].dropna().unique():
            for p in chunks(s):
                if len(p.split()) >= 8:
                    yield p


def natural_earth():
    path = Path.home() / ".cache" / "ojas" / "decision-eval" / "ne_110m_admin_0_countries.geojson"
    if not path.is_file():
        return
    for f in json.loads(path.read_text())["features"]:
        p = f["properties"]
        yield (f"{p['ADMIN']} is a country in {p['CONTINENT']}, in the region of {p['SUBREGION']}. "
               f"Its population is about {int(p['POP_EST']):,}.")
    places = path.parent / "ne_110m_populated_places.geojson"
    if places.is_file():
        for f in json.loads(places.read_text())["features"]:
            p = f["properties"]
            if p.get("FEATURECLA") == "Admin-0 capital":
                yield f"{p['NAME']} is the capital of {p['ADM0NAME']}."


def build(data, index):
    index.mkdir(parents=True, exist_ok=True)
    passages = []
    for name, source in (("wikipedia", wikipedia(data)), ("open book", open_book(data)),
                         ("sciq support", sciq_support(data)), ("natural earth", natural_earth())):
        before = len(passages)
        passages.extend(source)
        print(f"{name:14s} {len(passages) - before:8d} passages")
    vec = CountVectorizer(ngram_range=(1, 2), min_df=2, max_features=2_000_000, stop_words=STOP, dtype=np.float32)
    tf = vec.fit_transform(passages).tocsc().astype(np.float32)
    n, k1, b = tf.shape[0], 1.2, 0.75
    df = np.diff(tf.indptr)
    idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
    lengths = np.asarray(tf.sum(axis=1)).ravel()
    norm = k1 * (1 - b + b * lengths / lengths.mean())
    tf = tf.tocsr()
    rows = np.repeat(np.arange(n), np.diff(tf.indptr))
    tf.data = tf.data * (k1 + 1) / (tf.data + norm[rows])
    weights = (tf @ sparse.diags(idf)).astype(np.float32).tocsc()
    sparse.save_npz(index / "weights.npz", weights)
    with (index / "vocab.pkl").open("wb") as f:
        pickle.dump(vec, f)
    with (index / "passages.jsonl").open("w") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"{n} passages, {weights.shape[1]} terms in {index}")


class Memory:
    def __init__(self, index):
        self.weights = sparse.load_npz(index / "weights.npz").tocsc()
        with (index / "vocab.pkl").open("rb") as f:
            self.vec = pickle.load(f)
        self.passages = [json.loads(line) for line in (index / "passages.jsonl").open()]

    def search(self, queries, k):
        q = self.vec.transform(queries)
        q.data[:] = 1.0
        scores = (q @ self.weights.T).toarray() if len(queries) < 64 else None
        out = []
        for i in range(len(queries)):
            row = (q[i] @ self.weights.T).toarray().ravel() if scores is None else scores[i]
            top = np.argpartition(-row, k)[:k]
            top = top[np.argsort(-row[top])]
            out.append([self.passages[j] for j in top if row[j] > 0])
        return out


def query_text(body):
    """What a request asks about: its instructions, option texts and state."""
    parts = []
    for q in (body.get("questions") or {}).values():
        parts.append(str(q.get("instructions", "")))
        crit = q.get("criteria")
        if isinstance(crit, dict):
            parts += [f"{k} {v or ''}" for k, v in crit.items()]
        elif isinstance(crit, list):
            parts += [str(c) for c in crit]
    state = body.get("state")
    parts.append(state if isinstance(state, str) else json.dumps(state, ensure_ascii=False))
    return " ".join(parts)[:4000]


def augment_body(body, facts):
    out = dict(body)
    out["state"] = {"background": facts, "input": body.get("state")}
    return out


def augment(index, src, out, k, keep=False):
    memory = Memory(index)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.jsonl")) if src.is_dir() else [src]
    for f in files:
        rows = [json.loads(line) for line in f.open()]
        bodies = [r["body"] if "body" in r else r for r in rows]
        found = []
        for s in range(0, len(bodies), 256):
            found += memory.search([query_text(b) for b in bodies[s:s + 256]], k)
        with (out / f.name).open("w") as o:
            for r, b, facts in zip(rows, bodies, found):
                if keep:
                    line = {**r, "background": facts} if "body" in r else {"body": b, "background": facts}
                else:
                    nb = augment_body(b, facts)
                    line = {**r, "body": nb} if "body" in r else nb
                o.write(json.dumps(line, ensure_ascii=False) + "\n")
        print(f"{f.name:30s} {len(rows):6d} augmented")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("data", type=Path); b.add_argument("index", type=Path)
    a = sub.add_parser("augment"); a.add_argument("index", type=Path); a.add_argument("src", type=Path)
    a.add_argument("out", type=Path); a.add_argument("--k", type=int, default=3)
    a.add_argument("--keep", action="store_true")
    s = sub.add_parser("show"); s.add_argument("index", type=Path); s.add_argument("query")
    args = parser.parse_args()
    if args.cmd == "build":
        build(args.data, args.index)
    elif args.cmd == "augment":
        augment(args.index, args.src, args.out, args.k, args.keep)
    else:
        for p in Memory(args.index).search([args.query], 5)[0]:
            print("-", p[:300])


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Reference sentence embeddings for the engine's embedding parity test.

Picks texts of varied length from a JSONL corpus (one JSON string, or an object with
a `text` field, per line), adds longer texts made by joining consecutive ones so the
sliding-window layers see sequences past their reach, and embeds every text with the
sentence-transformers model on the CPU in float32, once per named prompt:

    embedding_reference.py <model dir> <texts.jsonl> <out.jsonl>
        [--prompt-name Document --prompt-name QuestionAnswering] [--count 64]
        [--long 600 --long 1200 --long 2500]

Each output line is `{"prompt_name", "prompt", "text", "ids", "embedding"}`: the token
ids of prompt and text with the special tokens, and the normalized embedding.
`crates/ojas-models/tests/embedding_parity.rs` reads the file. Needs
`sentence-transformers` and `torch`.
"""

import argparse
import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer


def read_texts(path: Path) -> list[str]:
    texts = []
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        value = json.loads(line)
        texts.append(value if isinstance(value, str) else value.get("text", ""))
    return [t for t in texts if t]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path)
    parser.add_argument("texts", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--prompt-name", action="append", dest="prompts")
    parser.add_argument("--count", type=int, default=64, help="texts taken across the length range")
    parser.add_argument("--long", type=int, action="append", help="token lengths of the joined long texts")
    parser.add_argument("--pool", type=int, default=20000, help="corpus lines considered")
    args = parser.parse_args()
    prompts = args.prompts or ["Document", "QuestionAnswering"]
    longs = args.long or [600, 1200, 2500]

    model = SentenceTransformer(str(args.model), device="cpu", model_kwargs={"torch_dtype": torch.float32})
    tok = model.tokenizer
    corpus = read_texts(args.texts)[: args.pool]
    lengths = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in corpus]
    # Evenly spaced through the corpus sorted by length: short, typical and long texts.
    order = sorted(range(len(corpus)), key=lambda i: lengths[i])
    step = max(1, len(order) // args.count)
    chosen = [corpus[order[min(len(order) - 1, k * step + step // 2)]] for k in range(args.count)]
    # Joined texts reaching each target length.
    at = 0
    for target in longs:
        parts, n = [], 0
        while n < target and at < len(corpus):
            parts.append(corpus[at])
            n += lengths[at] + 1
            at += 1
        chosen.append(" ".join(parts))

    with args.out.open("w") as f:
        for name in prompts:
            prompt = model.prompts[name]
            emb = model.encode(chosen, prompt_name=name, normalize_embeddings=True, batch_size=8, convert_to_numpy=True)
            for text, vec in zip(chosen, emb):
                ids = tok(prompt + text)["input_ids"]
                f.write(json.dumps({"prompt_name": name, "prompt": prompt, "text": text, "ids": ids,
                                    "embedding": [float(x) for x in vec]}) + "\n")
    print(f"wrote {len(prompts) * len(chosen)} embeddings of {len(chosen)} texts to {args.out}")


if __name__ == "__main__":
    main()

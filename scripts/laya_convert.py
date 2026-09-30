#!/usr/bin/env python3
"""Convert a Laya checkpoint to one GGUF file that `ojas decide` loads.

A Laya checkpoint (https://huggingface.co/convaiinnovations/laya) is a ModernBERT
encoder plus a decision head, saved together in one `model.safetensors`:

    encoder.*   ModernBERT (ModernBertModel) weights
    head.*      two nn.TransformerEncoderLayer blocks (norm_first, ReLU)
    type_emb    one row per question type (choice, score, noul)
    scorer.*    LayerNorm -> Linear(d, d) -> GELU -> Linear(d, 1), per option marker
    act_head.*  Linear(d + 4, 256) -> GELU -> Linear(256, n_act)
    temperature per-type calibration temperatures

The encoder is written under llama.cpp's `modern-bert` architecture: its tensor names
and `modern-bert.*` keys, as llama.cpp's own converter writes them. The head follows
under `laya.*` tensor names, and the calibration configuration under `laya.*` keys.

Two tokenizer families occur, both Hugging Face BPE:

* byte-level BPE (the English and typed-decisions checkpoints, ModernBERT's own):
  `tokenizer.ggml.model = "gpt2"`, `tokenizer.ggml.pre = "modern-bert"`;
* SentencePiece-style BPE over U+2581 with byte fallback and a Metaspace
  pre-tokenizer (the multilingual checkpoint, mmBERT's Gemma-derived vocabulary):
  `tokenizer.ggml.model = "gemma4"`, the llama.cpp name for that family, with
  `tokenizer.ggml.add_space_prefix` and `tokenizer.ggml.pre = "metaspace"` carrying
  the Metaspace prefix and word split.

As for every BERT-family model in llama.cpp, `[CLS]` is `tokenizer.ggml.bos_token_id`
and `[SEP]` is `tokenizer.ggml.seperator_token_id`. llama.cpp's loader refuses a file
whose tensors its architecture does not declare, so this file is for Ojas.

Usage:
    laya_convert.py <checkpoint dir> <out.gguf> [--outtype f16|f32]

`<checkpoint dir>` is the repository root for the English checkpoint, or its
`multilingual/` or `typed-decisions/` subfolder. Requires numpy, safetensors and the
`gguf` package (pip install gguf).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gguf
import numpy as np
from safetensors import safe_open

ARCH = "modern-bert"
QUESTION_TYPES = ["choice", "score", "noul"]

# checkpoint name (after `encoder.model.` / `encoder.`) -> GGUF name, per layer
ENCODER_LAYER_TENSORS = {
    "attn_norm.weight": "attn_norm.weight",
    "attn.Wqkv.weight": "attn_qkv.weight",
    "attn.Wo.weight": "attn_output.weight",
    "mlp_norm.weight": "ffn_norm.weight",
    "mlp.Wi.weight": "ffn_up.weight",
    "mlp.Wo.weight": "ffn_down.weight",
}
ENCODER_TENSORS = {
    "embeddings.tok_embeddings.weight": "token_embd.weight",
    "embeddings.norm.weight": "token_embd_norm.weight",
    "final_norm.weight": "output_norm.weight",
}

# head checkpoint name -> GGUF name; the GPU encoder block reads the head's blocks by
# the same names as the encoder's own.
HEAD_BLOCK_TENSORS = {
    "norm1.weight": "attn_norm.weight",
    "norm1.bias": "attn_norm.bias",
    "self_attn.in_proj_weight": "attn_qkv.weight",
    "self_attn.in_proj_bias": "attn_qkv.bias",
    "self_attn.out_proj.weight": "attn_output.weight",
    "self_attn.out_proj.bias": "attn_output.bias",
    "norm2.weight": "ffn_norm.weight",
    "norm2.bias": "ffn_norm.bias",
    "linear1.weight": "ffn_up.weight",
    "linear1.bias": "ffn_up.bias",
    "linear2.weight": "ffn_down.weight",
    "linear2.bias": "ffn_down.bias",
}
OTHER_TENSORS = {
    "type_emb.weight": "laya.type_emb.weight",
    "scorer.0.weight": "laya.scorer_norm.weight",
    "scorer.0.bias": "laya.scorer_norm.bias",
    "scorer.1.weight": "laya.scorer_fc.weight",
    "scorer.1.bias": "laya.scorer_fc.bias",
    "scorer.3.weight": "laya.scorer_out.weight",
    "scorer.3.bias": "laya.scorer_out.bias",
    "act_head.0.weight": "laya.act_fc.weight",
    "act_head.0.bias": "laya.act_fc.bias",
    "act_head.2.weight": "laya.act_out.weight",
    "act_head.2.bias": "laya.act_out.bias",
}

# Matrices the GPU multiplies keep the chosen precision. Vectors, norms, the type
# embedding and the host-side scorer and act-head weights are stored as f32, as
# llama.cpp stores norms and biases.
GPU_MATRICES = ("token_embd.weight", "attn_qkv.weight", "attn_output.weight", "ffn_up.weight",
                "ffn_down.weight", "laya.scorer_fc.weight")


def fail(msg: str) -> None:
    sys.exit(f"laya_convert: {msg}")


def read_checkpoint(ckpt: Path) -> tuple[dict[str, np.ndarray], dict, dict, dict, dict]:
    for rel in ("model.safetensors", "encoder/config.json", "tokenizer/tokenizer.json",
                "tokenizer/tokenizer_config.json", "rl_agent_config.json"):
        if not (ckpt / rel).exists():
            fail(f"{ckpt / rel} not found; is {ckpt} a Laya checkpoint directory?")
    tensors = {}
    with safe_open(str(ckpt / "model.safetensors"), framework="np") as f:
        for name in f.keys():
            tensors[name] = f.get_tensor(name)

    def load(rel: str) -> dict:
        return json.loads((ckpt / rel).read_text())

    return (tensors, load("encoder/config.json"), load("tokenizer/tokenizer.json"),
            load("tokenizer/tokenizer_config.json"), load("rl_agent_config.json"))


def encoder_tensors(tensors: dict[str, np.ndarray], layers: int) -> list[tuple[str, np.ndarray]]:
    prefix = "encoder.model." if any(k.startswith("encoder.model.") for k in tensors) else "encoder."
    out = []
    for src, dst in ENCODER_TENSORS.items():
        if prefix + src not in tensors:
            fail(f"missing encoder tensor {prefix + src}")
        out.append((dst, tensors[prefix + src]))
    for i in range(layers):
        for src, dst in ENCODER_LAYER_TENSORS.items():
            key = f"{prefix}layers.{i}.{src}"
            if key not in tensors:
                if src == "attn_norm.weight" and i == 0:
                    continue  # ModernBERT's first layer has no attention norm
                fail(f"missing encoder tensor {key}")
            out.append((f"blk.{i}.{dst}", tensors[key]))
    return out


def head_tensors(tensors: dict[str, np.ndarray]) -> tuple[list[tuple[str, np.ndarray]], int]:
    out = []
    blocks = sorted({int(k.split(".")[2]) for k in tensors if k.startswith("head.layers.")})
    for i in blocks:
        for src, dst in HEAD_BLOCK_TENSORS.items():
            key = f"head.layers.{i}.{src}"
            if key not in tensors:
                fail(f"missing head tensor {key}")
            out.append((f"laya.blk.{i}.{dst}", tensors[key]))
    for src, dst in OTHER_TENSORS.items():
        if src not in tensors:
            fail(f"missing head tensor {src}")
        out.append((dst, tensors[src]))
    return out, len(blocks)


def write_tokenizer(w: gguf.GGUFWriter, tok: dict, tok_cfg: dict, vocab_size: int) -> str:
    """Write the vocabulary and its special ids; returns the family written."""
    model = tok["model"]
    if model.get("type") != "BPE":
        fail(f"tokenizer model {model.get('type')!r} is not supported (BPE only)")
    vocab: dict[str, int] = dict(model["vocab"])
    added = {a["content"]: a for a in tok.get("added_tokens", [])}
    for a in added.values():
        vocab[a["content"]] = a["id"]
    by_id = {i: t for t, i in vocab.items()}
    pre = tok.get("pre_tokenizer") or {}
    metaspace = pre.get("type") == "Metaspace"
    byte_fallback = bool(model.get("byte_fallback"))

    tokens, types = [], []
    for i in range(vocab_size):
        t = by_id.get(i)
        if t is None:
            # The embedding is padded past the tokenizer; llama.cpp names these rows
            # the same way.
            tokens.append(f"[PAD{i}]")
            types.append(gguf.TokenType.UNUSED)
        elif t in added:
            tokens.append(t)
            types.append(gguf.TokenType.CONTROL if added[t]["special"] else gguf.TokenType.USER_DEFINED)
        elif byte_fallback and len(t) == 6 and t.startswith("<0x") and t.endswith(">"):
            tokens.append(t)
            types.append(gguf.TokenType.BYTE)
        else:
            tokens.append(t)
            types.append(gguf.TokenType.NORMAL)

    merges = []
    for m in model["merges"]:
        a, b = m.split(" ", 1) if isinstance(m, str) else m
        if " " in a or " " in b:
            fail(f"merge ({a!r}, {b!r}) contains a space, which the 'a b' GGUF form cannot carry")
        merges.append(f"{a} {b}")

    if metaspace:
        if pre.get("replacement") != "▁" or not pre.get("split", True):
            fail(f"Metaspace pre-tokenizer {pre} is not the supported form (U+2581, split)")
        if pre.get("prepend_scheme") != "always":
            fail(f"Metaspace prepend_scheme {pre.get('prepend_scheme')!r} is not supported (always)")
        w.add_tokenizer_model("gemma4")
        w.add_tokenizer_pre("metaspace")
        w.add_add_space_prefix(True)
    else:
        w.add_tokenizer_model("gpt2")
        w.add_tokenizer_pre("modern-bert")
    # Every sequence is framed [CLS] ... [SEP].
    w.add_add_bos_token(True)
    w.add_add_eos_token(True)
    w.add_token_list(tokens)
    w.add_token_types(types)
    w.add_token_merges(merges)

    def token_id(key: str) -> int | None:
        t = tok_cfg.get(key)
        if isinstance(t, dict):
            t = t.get("content")
        return vocab.get(t) if t is not None else None

    for key, add in (("cls_token", w.add_bos_token_id), ("sep_token", w.add_sep_token_id),
                     ("pad_token", w.add_pad_token_id), ("mask_token", w.add_mask_token_id),
                     ("unk_token", w.add_unk_token_id)):
        i = token_id(key)
        if i is None:
            fail(f"tokenizer_config.json has no {key}")
        add(i)
    eos = token_id("eos_token")
    w.add_eos_token_id(eos if eos is not None else token_id("sep_token"))
    return "metaspace" if metaspace else "byte-level"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--outtype", choices=["f16", "f32"], default="f16")
    args = ap.parse_args()

    tensors, cfg, tok, tok_cfg, agent = read_checkpoint(args.checkpoint)
    if cfg.get("model_type") != "modernbert":
        fail(f"encoder model_type {cfg.get('model_type')!r} is not ModernBERT")
    if cfg.get("hidden_activation", "gelu") != "gelu":
        fail(f"hidden_activation {cfg['hidden_activation']!r} is not supported (gelu only)")
    layers = cfg["num_hidden_layers"]
    rope = cfg.get("rope_parameters") or {}
    rope_global = rope.get("full_attention", {}).get("rope_theta", cfg.get("global_rope_theta", 160000.0))
    rope_local = rope.get("sliding_attention", {}).get("rope_theta", cfg.get("local_rope_theta", rope_global))

    enc = encoder_tensors(tensors, layers)
    head, n_blocks = head_tensors(tensors)
    d = cfg["hidden_size"]
    wide = np.float16 if args.outtype == "f16" else np.float32

    w = gguf.GGUFWriter(str(args.out), arch=ARCH)
    variant = args.checkpoint.name if args.checkpoint.name in ("multilingual", "typed-decisions") else "english"
    w.add_name(f"Laya {variant}")
    w.add_quantization_version(gguf.GGML_QUANT_VERSION)
    w.add_file_type(gguf.LlamaFileType.MOSTLY_F16 if args.outtype == "f16" else gguf.LlamaFileType.ALL_F32)
    w.add_block_count(layers)
    w.add_context_length(cfg["max_position_embeddings"])
    w.add_embedding_length(d)
    w.add_feed_forward_length(cfg["intermediate_size"])
    w.add_head_count(cfg["num_attention_heads"])
    w.add_layer_norm_eps(cfg.get("norm_eps", cfg.get("layer_norm_eps", 1e-5)))
    w.add_causal_attention(False)
    w.add_rope_freq_base(float(rope_global))
    w.add_key_value(f"{ARCH}.rope.freq_base_swa", float(rope_local), gguf.GGUFValueType.FLOAT32)
    w.add_sliding_window(cfg["local_attention"])
    w.add_key_value(f"{ARCH}.attention.sliding_window_pattern", cfg["global_attn_every_n_layers"],
                    gguf.GGUFValueType.UINT32)
    w.add_vocab_size(cfg["vocab_size"])
    w.add_key_value(f"{ARCH}.hidden_activation", "gelu", gguf.GGUFValueType.STRING)
    family = write_tokenizer(w, tok, tok_cfg, cfg["vocab_size"])

    w.add_uint32("laya.head.block_count", n_blocks)
    w.add_uint32("laya.head.head_count", d // 64)
    w.add_uint32("laya.head.feed_forward_length", tensors["head.layers.0.linear1.weight"].shape[0])
    # nn.TransformerEncoderLayer and nn.LayerNorm defaults.
    w.add_float32("laya.head.layer_norm_epsilon", 1e-5)
    w.add_uint32("laya.max_len", int(agent["max_len"]))
    w.add_uint32("laya.head_max_len", int(agent["head_max_len"]))
    w.add_array("laya.question_types", QUESTION_TYPES)
    w.add_array("laya.temperature", [float(t) for t in agent.get("temperature", [1.0, 1.0, 1.0])])
    by_opt = agent.get("temperature_by_options", {})
    w.add_array("laya.temperature_by_options.keys", list(by_opt.keys()))
    w.add_array("laya.temperature_by_options.values", [float(v) for v in by_opt.values()])

    for name, arr in enc + head:
        dtype = wide if name.endswith(GPU_MATRICES) else np.float32
        w.add_tensor(name, np.ascontiguousarray(arr.astype(dtype)))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.out}: {layers}-layer encoder + {len(head)} head tensors, {family} BPE tokenizer")


if __name__ == "__main__":
    main()

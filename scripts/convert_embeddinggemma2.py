#!/usr/bin/env python3
"""Convert an EmbeddingGemma 2 checkpoint (Hugging Face `EmbeddingGemma2Model`) into
a GGUF text encoder the engine loads.

The text backbone is a Gemma 4 layer stack run bidirectionally, so the file uses the
`gemma4` key and tensor layout and adds what makes it an encoder:

* `gemma4.attention.causal = false` selects the packed text-encoder path;
* `gemma4.pooling_type = 1` (mean over every token, the task prompt included);
* `dense_2.weight` is the `embedding_projection` applied to the pooled vector
  (a linear map commutes with the mean, so pooling first is exact);
* `gemma4.attention.sliding_window` is the full symmetric width of a local layer,
  twice the reach: a query at `i` sees keys with `|i - j| <= reach`, which is how
  the model's bidirectional sliding mask is defined.

The vision and audio towers are dropped. Matrices are written as f16 (the checkpoint
is bf16; every value is checked to fit f16's range), norms and scalars as f32. Norm
weights are stored as they are used (`x * w`): Gemma 4 has no `1 + w` offset.

    convert_embeddinggemma2.py <checkpoint dir> <out.gguf>

needs `numpy`, `safetensors`, `torch` and a `gguf` package recent enough to know the
Gemma 4 keys (`gguf-py` from the reference checkout, on `PYTHONPATH`).
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

import gguf

ARCH = "gemma4"
POOLING_MEAN = 1


def hf_vocab(model):
    """Tokens, scores and types from a Hugging Face `tokenizer.json`, one per id: special
    added tokens are control tokens, `<0xNN>` tokens are bytes, every score -1000 (the
    tokenizer merges by rank, not score)."""
    tj = json.loads((Path(model) / "tokenizer.json").read_text())
    vocab = tj["model"]["vocab"]
    added = {t["id"]: t for t in tj.get("added_tokens", [])}
    n = max(max(vocab.values()), max(added, default=0)) + 1
    tokens = [""] * n
    for tok, i in vocab.items():
        tokens[i] = tok
    for i, t in added.items():
        tokens[i] = t["content"]
    byte = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")
    types = []
    for i, tok in enumerate(tokens):
        if i in added and added[i].get("special"):
            types.append(gguf.TokenType.CONTROL)
        elif byte.match(tok):
            types.append(gguf.TokenType.BYTE)
        else:
            types.append(gguf.TokenType.NORMAL)
    return [t.encode("utf-8") for t in tokens], [-1000.0] * n, types

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, help="checkpoint directory (config.json, model.safetensors, tokenizer.json)")
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    config = json.loads((args.model / "config.json").read_text())
    text = config["text_config"]
    n_layer = text["num_hidden_layers"]
    d = text["hidden_size"]
    layer_types = text["layer_types"]
    per_layer = {int(k): v for k, v in (text.get("per_layer_config") or {}).items()}

    def layer_geom(i: int) -> tuple[int, int]:
        cfg = per_layer.get(i, {})
        return cfg.get("head_dim", text["head_dim"]), cfg.get("num_key_value_heads", text["num_key_value_heads"])

    sliding = [t == "sliding_attention" for t in layer_types]
    geoms = [layer_geom(i) for i in range(n_layer)]
    full = {geoms[i] for i in range(n_layer) if not sliding[i]}
    local = {geoms[i] for i in range(n_layer) if sliding[i]}
    # The layout has one head dim per attention kind; the head count of each layer is
    # an array.
    assert len(full) <= 1 and len(local) <= 1, f"head dims vary within an attention kind: {full} {local}"
    hd_full = next(iter(full))[0] if full else text["head_dim"]
    hd_local = next(iter(local))[0] if local else hd_full
    rope = text["rope_parameters"]
    for kind in ("full_attention", "sliding_attention"):
        assert rope[kind]["rope_type"] == "default", f"{kind}: rope type {rope[kind]['rope_type']} is not supported"

    w = gguf.GGUFWriter(args.out, ARCH)
    w.add_name("EmbeddingGemma 2")
    w.add_block_count(n_layer)
    w.add_context_length(text["max_position_embeddings"])
    w.add_embedding_length(d)
    w.add_feed_forward_length(text["intermediate_size"])
    w.add_head_count(text["num_attention_heads"])
    w.add_head_count_kv([g[1] for g in geoms])
    w.add_key_length(hd_full)
    w.add_value_length(hd_full)
    w.add_key_length_swa(hd_local)
    w.add_value_length_swa(hd_local)
    w.add_rope_dimension_count(hd_full)
    w.add_rope_dimension_count_swa(hd_local)
    w.add_rope_freq_base(rope["full_attention"]["rope_theta"])
    w.add_rope_freq_base_swa(rope["sliding_attention"]["rope_theta"])
    w.add_layer_norm_rms_eps(text["rms_norm_eps"])
    w.add_sliding_window(2 * text["sliding_window"])
    w.add_sliding_window_pattern(sliding)
    w.add_shared_kv_layers(0)
    w.add_embedding_length_per_layer_input(text.get("hidden_size_per_layer_input") or 0)
    w.add_vocab_size(text["vocab_size"])
    w.add_causal_attention(False)
    w.add_pooling_type(gguf.PoolingType(POOLING_MEAN))
    w.add_dense_features_dims("dense_2", d, text["embedding_dim"])
    # GGUF file type 1: every matrix stored as F16.
    w.add_file_type(1)

    # Tokenizer: the same encoding a gemma4 decoder file carries.
    tokens, scores, types = hf_vocab(args.model)
    w.add_tokenizer_model("gemma4")
    w.add_token_list(list(tokens))
    w.add_token_scores(list(scores))
    w.add_token_types(list(types))
    gguf.SpecialVocab(args.model, load_merges=True).add_to_gguf(w)
    w.add_add_space_prefix(False)

    # Sentence-transformers prompts, so a caller can pick a task prefix by name.
    st = args.model / "config_sentence_transformers.json"
    if st.is_file():
        prompts = json.loads(st.read_text()).get("prompts") or {}
        if prompts:
            w.add_array(f"{ARCH}.embedding.prompt_names", list(prompts.keys()))
            w.add_array(f"{ARCH}.embedding.prompts", list(prompts.values()))

    names = {
        "embed_tokens.weight": "token_embd.weight",
        "norm.weight": "output_norm.weight",
        "embedding_projection.weight": "dense_2.weight",
        "ple.per_layer_model_projection.weight": "per_layer_model_proj.weight",
        "ple.per_layer_projection_norm.weight": "per_layer_proj_norm.weight",
    }
    per_block = {
        "input_layernorm.weight": "attn_norm.weight",
        "self_attn.q_proj.weight": "attn_q.weight",
        "self_attn.k_proj.weight": "attn_k.weight",
        "self_attn.v_proj.weight": "attn_v.weight",
        "self_attn.q_norm.weight": "attn_q_norm.weight",
        "self_attn.k_norm.weight": "attn_k_norm.weight",
        "self_attn.o_proj.weight": "attn_output.weight",
        "post_attention_layernorm.weight": "post_attention_norm.weight",
        "pre_feedforward_layernorm.weight": "ffn_norm.weight",
        "mlp.gate_proj.weight": "ffn_gate.weight",
        "mlp.up_proj.weight": "ffn_up.weight",
        "mlp.down_proj.weight": "ffn_down.weight",
        "post_feedforward_layernorm.weight": "post_ffw_norm.weight",
        "ple_block.per_layer_input_gate.weight": "inp_gate.weight",
        "ple_block.per_layer_projection.weight": "proj.weight",
        "ple_block.post_per_layer_input_norm.weight": "post_norm.weight",
        "layer_scalar": "layer_output_scale.weight",
    }
    for i in range(n_layer):
        for src, dst in per_block.items():
            names[f"layers.{i}.{src}"] = f"blk.{i}.{dst}"

    prefix = "language_model."
    written = set()
    with safe_open(args.model / "model.safetensors", "pt") as f:
        for key in f.keys():
            if not key.startswith(prefix):
                continue
            src = key[len(prefix):]
            dst = names.get(src)
            if dst is None:
                raise SystemExit(f"unmapped text tensor {key}")
            t = f.get_tensor(key).to(torch.float32).numpy()
            if t.ndim >= 2:
                peak = float(np.abs(t).max())
                assert peak < 65504.0, f"{key}: |max| {peak} overflows f16"
                t = t.astype(np.float16)
            w.add_tensor(dst, t)
            written.add(src)
    missing = set(names) - written
    if missing:
        raise SystemExit(f"checkpoint lacks {sorted(missing)[:5]}")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.0f} MB, {len(written)} tensors)")


if __name__ == "__main__":
    main()

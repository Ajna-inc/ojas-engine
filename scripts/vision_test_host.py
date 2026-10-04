#!/usr/bin/env python3
"""Build a small decoder that a vision projector loads beside, to test the
projector without the model it belongs to.

The vision tower is loaded as a sidecar of a decoder whose hidden width equals the
projector's `clip.vision.projection_dim`. This writes a Qwen3.5 decoder with the
layer layout of a template decoder (`tinyopenjev`: four layers, random weights)
at that width, with a vocabulary cut to a few thousand tokens. It decodes nothing
useful; it gives `examples/vision_gate.rs` a decoder to load the real tower into,
so the tower can be checked against the CPU tower at its real size.

    pip install gguf numpy
    vision_test_host.py <tinyopenjev-Q8_0.gguf> <mmproj.gguf> <out.gguf> [--vocab N]

Put the output beside the projector: the loader attaches the one projector in the
decoder's directory. The output is deterministic.
"""

import argparse
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGUFValueType, GGUFWriter

SEED = 11


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("template", type=Path, help="a small Qwen3.5 GGUF giving the layer layout")
    parser.add_argument("projector", type=Path, help="the vision projector (mmproj) GGUF")
    parser.add_argument("out", type=Path)
    parser.add_argument("--vocab", type=int, default=4096, help="tokens kept")
    args = parser.parse_args()

    arch = "qwen35"
    template = GGUFReader(args.template)
    width = int(GGUFReader(args.projector).fields["clip.vision.projection_dim"].contents())
    old_width = int(template.fields[f"{arch}.embedding_length"].contents())
    old_vocab = len(template.fields["tokenizer.ggml.tokens"].contents())

    writer = GGUFWriter(args.out, arch)
    skip = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture", "tokenizer.ggml.merges"}
    for name, field in template.fields.items():
        if name in skip or name.startswith(f"{arch}.decision.") or name.startswith("tokenizer.chat_template"):
            continue
        value_type = field.types[0]
        sub_type = field.types[-1] if value_type == GGUFValueType.ARRAY else None
        value = field.contents()
        if name == f"{arch}.embedding_length":
            value = width
        elif name in ("tokenizer.ggml.tokens", "tokenizer.ggml.token_type"):
            value = value[: args.vocab]
        elif name.startswith("tokenizer.ggml.") and name.endswith("_token_id"):
            value = min(int(value), args.vocab - 1)
        writer.add_key_value(name, value, value_type, sub_type=sub_type)

    # The template's layout at the new width: every dimension that was the hidden
    # width becomes the projector's, the vocabulary is cut, and the weights are
    # random, small enough to keep the residual stream finite.
    rng = np.random.default_rng(SEED)
    for tensor in template.tensors:
        dims = [int(d) for d in tensor.shape]
        dims = [width if d == old_width else args.vocab if d == old_vocab else d for d in dims]
        shape = tuple(reversed(dims))
        if tensor.name.endswith("norm.weight"):
            data = np.ones(shape, dtype=np.float32)
        elif len(dims) == 1 or tensor.tensor_type.name == "F32":
            data = (rng.standard_normal(shape) * 0.02).astype(np.float32)
        else:
            data = (rng.standard_normal(shape) * 0.02).astype(np.float16)
        writer.add_tensor(tensor.name, data)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.out}: width {width}, vocabulary {args.vocab}")


if __name__ == "__main__":
    main()

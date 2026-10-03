#!/usr/bin/env python3
"""Build the tiny decision models the parity test uses for the label-code and
pointer readouts.

`tinylev` and `tinykev` are `tinyopenjev` (a four-layer Qwen3.5 with random
weights) under another decision type: the same tensors, the type's own prompt
template (from `crates/ojas-models/tests/decision/templates/`) and temperatures,
and for the pointer readout a seeded random `cls.output` projection. They exercise
every code path of the full-size models in a file of 46 MB.

    pip install gguf numpy
    decision_test_models.py <tinyopenjev-Q8_0.gguf> <out dir> [--prefix NAME]

writes `tinylev-Q8_0.gguf` and `tinykev-Q8_0.gguf` into the output directory. The
output is deterministic. Any Qwen3.5 GGUF serves as the source: from a full-size
one, the files have the full-size models' architecture and cost, which is what a
benchmark needs; `--prefix` names them (`<prefix>lev-<quant>.gguf`).
"""

import argparse
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGUFValueType, GGUFWriter

TEMPLATES = Path(__file__).resolve().parent.parent / "crates/ojas-models/tests/decision/templates"

# Temperatures by type: `<question type>[.<option-count bucket>]`.
TEMPERATURES = {
    "lev": {"noul": 2.3333, "score": 2.8033, "choice": 1.7667, "choice.small": 1.7898, "choice.mid": 1.6086},
    "kev": {"choice": 2.40605, "score": 2.40605, "noul": 2.40605},
}

# The pointer projection: query and key halves of 16 values each, small enough that
# its scores stay in the range a trained projection gives.
POINTER_WIDTH = 32
POINTER_SCALE = 0.1
POINTER_SEED = 7


def derive(source: Path, out: Path, kind: str) -> None:
    reader = GGUFReader(source)
    arch = "qwen35"
    writer = GGUFWriter(out, arch)
    skip = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture",
            "tokenizer.chat_template.systemone"}
    for name, field in reader.fields.items():
        if name in skip or name.startswith(f"{arch}.decision."):
            continue
        value_type = field.types[0]
        sub_type = field.types[-1] if value_type == GGUFValueType.ARRAY else None
        writer.add_key_value(name, field.contents(), value_type, sub_type=sub_type)
    writer.add_string(f"{arch}.decision.type", kind)
    writer.add_string("tokenizer.chat_template.systemone", (TEMPLATES / f"{kind}.jinja").read_text())
    for key, value in TEMPERATURES[kind].items():
        writer.add_float32(f"{arch}.decision.temperature.{key}", value)
    if kind == "kev":
        writer.add_uint32(f"{arch}.embedding_length_out", POINTER_WIDTH)

    for tensor in reader.tensors:
        writer.add_tensor(tensor.name, np.asarray(tensor.data), raw_dtype=tensor.tensor_type)
    if kind == "kev":
        width = int(reader.fields[f"{arch}.embedding_length"].contents())
        rng = np.random.default_rng(POINTER_SEED)
        writer.add_tensor("cls.output.weight", (rng.standard_normal((POINTER_WIDTH, width)) * POINTER_SCALE).astype(np.float32))
        writer.add_tensor("cls.output.bias", (rng.standard_normal(POINTER_WIDTH) * POINTER_SCALE).astype(np.float32))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="a Qwen3.5 GGUF")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--prefix", default="tiny", help="name prefix of the files written")
    parser.add_argument("--quant", default="Q8_0", help="quantization named in the files written")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for kind in ("lev", "kev"):
        out = args.out_dir / f"{args.prefix}{kind}-{args.quant}.gguf"
        derive(args.source, out, kind)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

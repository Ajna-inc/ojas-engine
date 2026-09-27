#!/usr/bin/env python3
"""Strip an in-graph NMS tail (an /end2end/ subgraph) off a detector ONNX.

Finds the tensors crossing from the model body into nodes whose names start
with the tail prefix, extracts the sub-model ending at that boundary, and
onnxslim's the result. Used for the open-image-models plate detectors; the
dense output must land back at `[1, 4+nc, A]`.

usage: strip_onnx_nms.py in_end2end.onnx out_dense.onnx [--prefix /end2end/]
"""
import argparse
import sys

import onnx
import onnx.utils


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--prefix", default="/end2end/")
    a = p.parse_args()

    m = onnx.load(a.src)
    g = m.graph
    produced_by = {o: n.name for n in g.node for o in n.output}
    crossing = sorted({
        i
        for n in g.node
        if n.name.startswith(a.prefix)
        for i in n.input
        if produced_by.get(i, a.prefix).startswith(a.prefix) is False and i in produced_by
    })
    if not crossing:
        sys.exit(f"no tensors cross into nodes prefixed {a.prefix!r} — nothing to strip?")
    if len(crossing) > 1:
        sys.exit(f"expected one dense boundary tensor, found {crossing} — pick manually")
    inputs = [i.name for i in g.input]
    print(f"boundary: {crossing[0]}  (inputs {inputs})")
    onnx.utils.extract_model(a.src, a.dst, inputs, crossing)

    import onnxslim

    slim = onnxslim.slim(onnx.load(a.dst))
    onnx.save(slim, a.dst)
    out = onnx.shape_inference.infer_shapes(slim)
    for o in out.graph.output:
        dims = [d.dim_value or d.dim_param or "?" for d in o.type.tensor_type.shape.dim]
        print(f"dense output: {o.name} {dims}")


if __name__ == "__main__":
    main()

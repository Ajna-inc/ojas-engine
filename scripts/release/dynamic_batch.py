#!/usr/bin/env python3
"""Make a static batch-1 ONNX detector batchable.

For exports we cannot regenerate (e.g. the open-image-models plate detectors,
trained outside Ultralytics): the input's leading dim becomes the dim_param
`batch`, every constant Reshape target whose first entry is 1 becomes 0
("copy the input's batch"), stale value_info is dropped and graph outputs get
the same symbolic batch. The result is checked under onnxruntime: a batch of N
different images must reproduce each image's batch-1 output.

usage: dynamic_batch.py in.onnx out.onnx [--check 8]
"""
import argparse

import numpy as np
import onnx
from onnx import numpy_helper


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--check", type=int, default=8)
    a = p.parse_args()

    m = onnx.load(a.src)
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    consts = {n.output[0]: n for n in g.node if n.op_type == "Constant"}
    fixed = 0
    for n in g.node:
        if n.op_type != "Reshape":
            continue
        s = n.input[1]
        if s in inits:
            v = numpy_helper.to_array(inits[s]).copy()
            if v.size and v[0] == 1:
                v[0] = 0
                inits[s].CopyFrom(numpy_helper.from_array(v, s))
                fixed += 1
        elif s in consts:
            t = consts[s].attribute[0].t
            v = numpy_helper.to_array(t).copy()
            if v.size and v[0] == 1:
                v[0] = 0
                t.CopyFrom(numpy_helper.from_array(v, t.name))
                fixed += 1
    for vi in list(g.input) + list(g.output):
        d = vi.type.tensor_type.shape.dim[0]
        d.ClearField("dim_value")
        d.dim_param = "batch"
    del g.value_info[:]
    onnx.checker.check_model(m)
    onnx.save(m, a.dst)
    print(f"reshapes rebatched: {fixed}")

    import onnxruntime as ort
    src = ort.InferenceSession(a.src, providers=["CPUExecutionProvider"])
    dst = ort.InferenceSession(a.dst, providers=["CPUExecutionProvider"])
    shape = [d if isinstance(d, int) else 1 for d in src.get_inputs()[0].shape]
    rng = np.random.default_rng(0)
    xs = rng.random([a.check] + shape[1:], dtype=np.float32)
    name = src.get_inputs()[0].name
    batched = dst.run(None, {name: xs})[0]
    worst = 0.0
    for i in range(a.check):
        one = src.run(None, {name: xs[i:i + 1]})[0]
        worst = max(worst, float(np.abs(batched[i:i + 1] - one).max()))
    print(f"batch {a.check} vs per-image batch 1: max |diff| {worst:.3e}")
    assert worst < 1e-3, "batched output differs"


if __name__ == "__main__":
    main()

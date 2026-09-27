"""Replace every Reshape target (which paddle2onnx emits with 0/-1 dims that
onnxslim mis-folds) with the explicit runtime shape, probed via onnxruntime.
Model is fully static [1,3,48,320], so runtime shapes are the truth."""
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

SCRIPT_DIR = Path(__file__).resolve().parent
m = onnx.load(str(SCRIPT_DIR / "awiros_raw.onnx"))
g = m.graph

reshapes = [n for n in g.node if n.op_type == "Reshape"]
print(f"{len(reshapes)} Reshape nodes")

probe = onnx.load(str(SCRIPT_DIR / "awiros_raw.onnx"))
del probe.graph.output[:]
for n in reshapes:
    probe.graph.output.append(onnx.helper.make_empty_tensor_value_info(n.output[0]))
onnx.save(probe, str(SCRIPT_DIR / "probe_reshapes.onnx"))
so = ort.SessionOptions()
so.log_severity_level = 3
sess = ort.InferenceSession(str(SCRIPT_DIR / "probe_reshapes.onnx"), so, providers=["CPUExecutionProvider"])
outs = sess.run(None, {"x": np.random.randn(1, 3, 48, 320).astype("float32")})
shapes = {o.name: out.shape for o, out in zip(sess.get_outputs(), outs)}

for idx, n in enumerate(reshapes):
    shp = np.array(shapes[n.output[0]], dtype=np.int64)
    name = f"baked.reshape.shape.{idx}"
    g.initializer.append(onnx.numpy_helper.from_array(shp, name))
    n.input[1] = name
    # drop allowzero if present; explicit positive dims need none
    del n.attribute[:]
    print(f"  {n.name}: target -> {shp.tolist()}")

del g.value_info[:]
onnx.save(m, str(SCRIPT_DIR / "awiros_baked.onnx"))
print("saved awiros_baked.onnx")

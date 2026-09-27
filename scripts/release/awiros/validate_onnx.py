"""Run awiros_rec.onnx under onnxruntime on the test crops, both normalizations."""
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

SCRIPT_DIR = Path(__file__).resolve().parent
H, W = 48, 320

chars = [c for c in (SCRIPT_DIR / "en_dict.txt").read_text().split("\n") if c != ""]
CHARSET = ["<blank>"] + chars + [" "]
print(f"dict chars: {len(chars)}, charset incl blank+space: {len(CHARSET)}")


def preprocess(img_bgr, signed):
    img_h, img_w = img_bgr.shape[:2]
    new_w = min(int(img_w * (H / img_h)), W)
    resized = cv2.resize(img_bgr, (new_w, H))
    img = resized.astype(np.float32) / 255.0
    if signed:
        img = (img - 0.5) / 0.5
        pad_val = -1.0
    else:
        pad_val = 0.0
    out = np.full((H, W, 3), pad_val, dtype=np.float32)
    out[:, :new_w, :] = img
    return out.transpose(2, 0, 1)[None]


def decode(probs):
    ids = probs.argmax(axis=1)
    out, conf = [], []
    prev = -1
    for t, i in enumerate(ids):
        if i != 0 and i != prev:
            out.append(CHARSET[i])
            conf.append(probs[t, i])
        prev = i
    return "".join(out), (float(np.mean(conf)) if conf else 0.0)


sess = ort.InferenceSession(str(SCRIPT_DIR / "awiros_rec.onnx"), providers=["CPUExecutionProvider"])
iname = sess.get_inputs()[0].name
print("input:", iname, sess.get_inputs()[0].shape, "| output:",
      sess.get_outputs()[0].name, sess.get_outputs()[0].shape)

for img_name in ["plate_crop.png", "fa_crop.png", "tamil_crop.png"]:
    p = SCRIPT_DIR.parent / img_name
    img = cv2.imread(str(p))
    if img is None:
        print(f"{img_name}: NOT FOUND")
        continue
    for signed in (True, False):
        x = preprocess(img, signed)
        y = sess.run(None, {iname: x})[0][0]  # [40, 64]
        text, conf = decode(y)
        norm = "signed" if signed else "unit"
        print(f"{img_name:16s} [{norm:6s}] -> {text!r} (conf {conf:.4f}, out {y.shape})")

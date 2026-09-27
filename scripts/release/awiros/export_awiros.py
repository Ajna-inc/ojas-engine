"""Build Awiros anpr-ocr rec model per test.py config, load safetensors weights,
sanity-check dynamic inference, then jit.save a static inference model (CTC branch only).
"""
import copy
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "PaddleOCR"))

import paddle  # noqa: E402

paddle.set_device("cpu")

from ppocr.modeling.architectures import build_model as ppocr_build_model  # noqa: E402

CTC_NUM_CLASSES = 64
NRTR_NUM_CLASSES = 67

ARCH = {
    "model_type": "rec",
    "algorithm": "SVTR_HGNet",
    "Transform": None,
    "Backbone": {"name": "PPHGNetV2_B4", "text_rec": True},
    "Head": {
        "name": "MultiHead",
        "out_channels_list": {
            "CTCLabelDecode": CTC_NUM_CLASSES,
            "NRTRLabelDecode": NRTR_NUM_CLASSES,
        },
        "head_list": [
            {
                "CTCHead": {
                    "Neck": {
                        "name": "svtr",
                        "dims": 120,
                        "depth": 2,
                        "hidden_dims": 120,
                        "kernel_size": [1, 3],
                        "use_guide": True,
                    },
                    "Head": {"fc_decay": 1e-05},
                }
            },
            {"NRTRHead": {"nrtr_dim": 384, "max_text_length": 25}},
        ],
    },
}

IMAGE_SHAPE = [3, 48, 320]


def load_weights(model):
    from safetensors.numpy import load_file

    np_state = load_file(str(SCRIPT_DIR / "model.safetensors"))
    state = {k: paddle.to_tensor(v) for k, v in np_state.items()}
    model.set_state_dict(state)
    print(f"loaded {len(state)} tensors from safetensors")


def preprocess(img_bgr, signed=True):
    import cv2

    _, h, w = IMAGE_SHAPE
    img_h, img_w = img_bgr.shape[:2]
    ratio = h / img_h
    new_w = min(int(img_w * ratio), w)
    resized = cv2.resize(img_bgr, (new_w, h))
    if new_w < w:
        padded = np.zeros((h, w, 3), dtype=np.uint8)
        padded[:, :new_w, :] = resized
        resized = padded
    img = resized.astype(np.float32) / 255.0
    if signed:
        img = (img - 0.5) / 0.5
    return img.transpose((2, 0, 1))


def ctc_decode(probs, dict_path):
    # probs: [T, C]; class map: 0 = blank, 1..N = dict chars, N+1 = space
    chars = Path(dict_path).read_text().split("\n")
    chars = [c for c in chars if c != ""]
    charset = ["blank"] + chars + [" "]
    ids = probs.argmax(axis=1)
    out, conf = [], []
    prev = -1
    for t, i in enumerate(ids):
        if i != 0 and i != prev:
            out.append(charset[i] if i < len(charset) else "?")
            conf.append(probs[t, i])
        prev = i
    return "".join(out), (float(np.mean(conf)) if conf else 0.0)


def main():
    config = copy.deepcopy(ARCH)
    model = ppocr_build_model(config)
    model.eval()
    load_weights(model)

    # --- sanity: dynamic-graph inference on the test plate ---
    import cv2

    img = cv2.imread(str(SCRIPT_DIR.parent / "plate_crop.png"))
    x = paddle.to_tensor(np.expand_dims(preprocess(img, signed=True), 0))
    with paddle.no_grad():
        preds = model(x)
    if isinstance(preds, dict):
        preds = preds.get("ctc", next(iter(preds.values())))
    arr = preds.numpy()[0]
    print("dynamic output shape:", preds.shape)
    text, conf = ctc_decode(arr, SCRIPT_DIR / "en_dict.txt")
    print(f"dynamic decode: {text!r} conf={conf:.4f}")

    # --- rep() pass like PaddleOCR export ---
    for layer in model.sublayers():
        if hasattr(layer, "rep") and not getattr(layer, "is_repped", False):
            layer.rep()

    # --- to_static + save (CTC-only: eval-mode MultiHead returns ctc_out) ---
    static_model = paddle.jit.to_static(
        model,
        input_spec=[paddle.static.InputSpec(shape=[1, 3, 48, 320], dtype="float32", name="x")],
    )
    save_path = str(SCRIPT_DIR / "inference" / "inference")
    paddle.jit.save(static_model, save_path)
    print("saved static model to", save_path)


if __name__ == "__main__":
    main()

# Model manifest

Weights are never committed (`*.onnx`, `*.pt` are gitignored). Every model
used by ojas-vision is recorded here with its hash, source, and the exact
export command, so any machine can reproduce the file and a deployment
can refuse a model whose sha256 is not listed (provenance).

## yolo11n.onnx

- **sha256** `d2efec9b2d9898c2d067e07e1b2cefeb4b784152f41fb79a99af7e1622e62f65`
- **source** Ultralytics YOLO11n, COCO-pretrained (`yolo11n.pt` via ultralytics 8.x auto-download)
- **license** AGPL-3.0 (Ultralytics; enterprise license or a permissive swap — YOLOX/RTMDet — required for production)
- **export** `yolo export model=yolo11n.pt format=onnx opset=17 imgsz=640 simplify=True`
- **shape** `images [1,3,640,640] → output0 [1,84,8400]` (dense head, no in-graph NMS)
- **ops** Conv 88, Mul 79, Sigmoid 78, Concat 21, Add 16, Reshape 11, Split 10, MaxPool 3, Transpose 3, MatMul 2, Softmax 2, Resize 2, Slice 2, Sub 2, Div 1

## yolov8n.onnx

- **sha256** `9c8aa7a0df26c1afc6450b932bf3e384cde8a04272467e5a24406e1619d2bd17`
- **source** Ultralytics YOLOv8n, COCO-pretrained
- **license** AGPL-3.0 (same note as above)
- **export** `yolo export model=yolov8n.pt format=onnx opset=17 imgsz=640 simplify=True`
- **shape** `images [1,3,640,640] → output0 [1,84,8400]`

## plate-v9t-640.onnx

- **sha256** `8e9dbf43560cd59bf4d307c3e243ba6e792d2514ed8ebf666648f2add9d28b79`
- **source** `yolo-v9-t-640-license-plates-end2end.onnx` from ankandrew/open-image-models (release tag `assets`), with the in-graph NMS tail stripped
- **license** MIT (open-image-models, code and weights)
- **export** `python -c "import onnx.utils; onnx.utils.extract_model('yolo-v9-t-640-license-plates-end2end.onnx','dense.onnx',['images'],['/model/model.22/Concat_5_output_0'])"` then `onnxslim`
- **shape** `images [1,3,640,640] → [1,5,8400]` (dense single-class head; NMS host-side)
- **classes** `plate-v9t-640.classes.json`: `["plate"]`; no corner keypoints (box-only model — rectify by homography from the box until a pose-head fine-tune lands)

## plate-v9t-384.onnx / plate-v9t-256.onnx  (vehicle-crop sizes)

- **sha256** `dc9bc76022ec6503be9e80e8e924bf9cd7d797150059342d950ad6c0817a401f` (384) / `9357d0f211973cf94dbbf1f82578a873e8d1cc3ad2ecb6b1e0c6304ef777d170` (256)
- **source / license** same open-image-models release (MIT), `yolo-v9-t-{384,256}-license-plates-end2end.onnx`
- **export** `python scripts/release/strip_onnx_nms.py <end2end.onnx> <dense.onnx>` (finds the /end2end/ boundary, extracts, onnxslim)
- **shape** `images [1,3,S,S] → [1,5,A]` — 384→3024 anchors, 256→1344
- **why** plate detection runs on ~150-px vehicle crops; 384 reads the same test plate at ~25 ms vs ~250 ms for the 640 export (parity-gated: 384 at cosine 1.0)
- a `yolo-v9-s-608` higher-mAP variant exists at the same URL pattern for full-frame runs

## yolox-plates-s.onnx  (Apache-2.0 plate detector — the production licence fallback)

- **sha256** `f9302f8c9a951861cf5e1266951d3962ce299b6636e73fa32032ffe38fe90ee7`
- **source** huggingface.co/autolane/yolox-s-alpr `yolox_plates_s.onnx` (Apache-2.0; trained on US plates — Indian fine-tune advisable before production)
- **shape** `images [1,3,640,640] → output [1,8400,6]` — a RAW YOLOX head (`reg_x, reg_y, log w, log h, obj, cls`), NOT decode-in-inference; the engine grid-decodes over strides 8/16/32 (`HeadLayout::AnchorsFirstObj`, auto-detected)
- **preprocessing** intrinsic to the head layout and auto-selected: raw 0–255 BGR, top-left letterbox pad 114 (YOLOX removed input normalization — feeding the Ultralytics 0–1 RGB letterbox silently kills every activation)
- **classes** `yolox-plates-s.classes.json`: `["plate"]`
- **validated** parity gate vs onnxruntime at cosine 1.0 on both random and real letterboxed input; full chain on the fast-alpr photo: plate 0.84 → Awiros reads `5AU5341` at 0.95

## ppocrv6_tiny_rec.onnx  (step-one OCR)

- **sha256** `9ef676d6ed3c88256a2d92c640c44f25b0c40947e111b14b8be8f594091563e6`
- **source** huggingface.co/PaddlePaddle/PP-OCRv6_tiny_rec_onnx (`inference.onnx`, official conversion)
- **license** Apache-2.0
- **dict** `ppocrv6_tiny_dict.txt` — 6904 chars extracted from the repo's `inference.yml` + trailing space class (blank at index 0 → 6906 CTC classes)
- **norm** `signed` — standard PaddleOCR `(x/255 − 0.5)/0.5` (the default `OcrCfg`)
- **shape** `x [1,3,48,W→320 bound] → [1,40,6906]`; ops include Erf (GELU)
- **note** general-purpose multilingual recognizer; reads clean Latin plates well (0.99 on the fast-alpr test crop), partial on small/angled crops. The Indian fine-tune (0-9A-Z dict, Awiros/anpr-ocr Apache-2.0 checkpoint as teacher) replaces it as step two.

## awiros_rec.onnx  (Indian-plates reader — the step-two model, delivered)

- **sha256** `05349f8cb6bfa143c9c797f3f1b93e6cd45390ad8acae8e36c2de25119d742f9` (model) / `771347c3eea1db31a98917b2183f328c629c3632bf231064881c4d16b9c37238` (`awiros_dict.txt`)
- **source** huggingface.co/Awiros/anpr-ocr (Apache-2.0) — PP-OCRv5 server rec (PPHGNetV2_B4 + SVTR neck, CTC branch only), fine-tuned on 558k Indian plates incl. two-row; 98.42 % claimed
- **export** safetensors → Paddle 3.1.1 → `paddle.jit.save` with dynamic batch `[-1,3,48,320]` → paddle2onnx 2.0.1 (opset 17), kept RAW. Scripts: `scripts/release/awiros/`. Do **not** onnxslim this file — it mis-folds the copy-dim Reshapes into a model that fails session init; both the ojas importer and onnxruntime consume the raw export as-is.
- **batching** batch dim is a `dim_param`: the PlateOcr plan cache runs it batched — measured 93.9 → 46.1 ms/crop (2.0×) at batch 32 on M2 Max CPU, identical text at every batch size
- **norm** `signed`, **BGR channel order** (`OcrCfg { bgr: true }`, CLI `--ocr-norm signed,bgr`) — trained on cv2 BGR frames; matters for yellow/green plates
- **dict** 63 lines: 0-9, A-Z, a-z, trailing space (blank=0 → 64 CTC classes); fold case downstream
- **shape** `x [N,3,48,320] → [N,40,64]`; needs ceil-mode pooling and SAME_UPPER auto_pad (both in the importer)
- **reads** synthetic GJ plate at 1.000 mean conf; fast-alpr crop `5AU5341` at 0.92; ≥92 % exact-match down to 20 px crop height (synthetic curve, `examples/ocr_bench.rs`); parity-gated vs onnxruntime at cosine 1.0

## ppocrv5_en_rec.onnx

- **sha256** `4e16deb22c4da6468bdca539b2cd3c8687825538b67109177c47d359ab994cd7`
- **source** huggingface.co/monkt/paddleocr-onnx `languages/english/rec.onnx` (en_PP-OCRv5_mobile_rec)
- **license** Apache-2.0
- **norm** `unit` — this conversion folded mean/std into the stem Conv/BN, so it takes RAW `x/255` (`OcrCfg { norm: OcrNorm::Unit }`, CLI `--ocr-norm unit`). Feeding it the standard signed normalization double-normalizes and drives the net into a confident noise attractor ("cYanmaGentaYellow…" — random noise reproduces the same string at 0.94). Diagnosed by input probes under onnxruntime; with unit norm it reads the fast-alpr test crop at 0.94 and small clean crops at 0.98.
- **shape** `x [1,3,48,W→320 bound] → [1,T,438]` (dict 436 chars + space + blank)

## field_ft_s-dyn.onnx  (our vehicle detector — DEIM D-FINE-S, 10.19 M, Field fine-tune)

- **sha256** `d188c5aa0ffdad2872ccf9cdf1cda716c19c93f4f5e8e43683813b2f485256f7` (model) / `8d22d6c59145a5f3f09ada79363a8d6b0e994eb5553f07c5ed6ed30590109617` (`field_ft_s-dyn.classes.json`)
- **source / lineage** own training (`training/deim/`): D-FINE-S Objects365→COCO weights (`dfine_s_obj2coco.pth`, Apache-2.0) → Step 2 DEIM on UVH-26 ST + BMD-45 clean (`step2_deim_dfine_s.yml`, 72 ep, best_stg2) → Field fine-tune on the reviewed Field frames only (`field_ft_s.yml`, 24 ep on a a rented GPU pod; checkpoint `cloud_field_ft_s/…/field_ft_s/best_stg1.pth`, epoch 22, 0.7517 mAP@[.5:.95] on the day+night Field val, the `ours_field_ft_s` preset of `training/benchmark/dump_detections.py`)
- **license** Apache-2.0 (D-FINE and DEIM code and weights, HGNetv2 backbone; the training data are ours / IISc UVH-26 under its terms)
- **export** `python training/deim/export_onnx.py --config …/DEIM/configs/uvh/step2_deim_dfine_s.yml --weights …/best_stg1.pth --out …/detr/field_ft_s` — TorchScript exporter, opset 17, traced at batch 2 with a symbolic batch, `onnxslim`; the sibling static batch-1 `field_ft_s.onnx` (sha256 `14a05d68006f189b93ba9a4f32c8a9fdbc34cf8fdf047bc349babb6805a3d3c8`) is the same graph with the batch bound.
- **shape** `images [batch,3,640,640] → logits [batch,300,14] + boxes [batch,300,4]` — a DETR head: raw class logits (sigmoid on the consumer's side, top class per query, no NMS) and cxcywh boxes normalised by the input square. Bind `batch` on import (`import_check … batch=1`)
- **preprocessing** stretch to 640×640 (no letterbox), RGB, 0–1, no mean/std — what the trainer's `ConvertPILImage` feeds
- **classes** `field_ft_s-dyn.classes.json`: the 14 UVH-26 names in id order, `["Hatchback","Sedan","SUV","MUV","Bus","Truck","Three-wheeler","Two-wheeler","LCV","Mini-bus","Tempo-traveller","Bicycle","Van","Others"]` (column *i* is UVH category id *i*+1; the yml's dead column 0 is dropped at export — its sigmoid never exceeds 0.001)
- **ops** (after import) Conv 91, Gemm 55, MatMul 15, GridSample 9, LayerNorm 12, Softmax 11, TopK 1, TopKGather 2, Transpose 43, Slice 34, Concat 28, Split 10, Pad 1, MaxPool 1, ResizeNearest 2, ScaleShift 55, Binary 120, Unary 24, Reduce 5, View 217
- **validated** PyTorch vs onnxruntime on gold frames: scores and boxes within 5.5e-5 (static, dynamic at batch 1 and 3); engine CPU vs onnxruntime on 20 gold frames: 412/412 detections ≥ 0.4 matched at IoU 1.0000, |Δscore| 0; engine CUDA (f16): 411/412 at IoU ≥ 0.9 (one 17×29 px two-wheeler at 0.885), worst IoU 0.924, |Δscore| median 0.0012 / max 0.08

## field_ft_n32-dyn.onnx  (the 3.39 M edge model — not in `models/`, at `…/dev-cache/detr/`)

- **sha256** `72fd5d2c76fc60ab76744f271a70231284f4c1a5a0effc9189491e9589048a1b` (dynamic) / `b5ed6d3af66c66301b6956c9954c0b41aa01b1a564103c76ac5fcde8f4d3a7aa` (static `field_ft_n32.onnx`)
- **source / lineage** Step 3 `ojas_n32` (DEIM D-FINE-N widened to 96, 4 decoder layers) → Step 3b P3 multi-scale (`step3b_ojas_n32_p3.yml`) → Field fine-tune (`field_ft.yml`; `cloud_field_ft/…/field_ft/best_stg1.pth`, epoch 23, 0.6596 on the Field val; the `ours_field_ft` preset). Apache-2.0, same export command with `step3b_ojas_n32_p3.yml`; same shapes, classes and preprocessing as above (eval at 640 as the yml says; `--size 800` rebuilds the anchor grid for the 800-px lane)
- **validated** PyTorch vs onnxruntime within 5.9e-4 on raw logits (1.4e-5 boxes); engine CPU 402/402 at IoU 1.0000; CUDA (f16) 397/402 at IoU ≥ 0.9 — three 16–20 px two-wheelers at IoU 0.89, two boxes at 0.41–0.48 that fall under 0.4 on the GPU, one Hatchback/Sedan class flip on a near-tie

## dfine-s-coco-dyn.onnx  (D-FINE-S COCO — the Apache-2.0 objects detector shipped in place of yolo11n)

- **sha256** `2c2f90bbda5a76d9ed6f6cc80a576c43cc453a05592198ac4e9054e74c2c9418`
- **source** D-FINE-S COCO (github.com/Peterande/D-FINE, `dfine_s_coco.pth`), Apache-2.0; the same 80 COCO classes and ids as yolo11n (`person` = 0, `car` = 2, `motorcycle` = 3, `bus` = 5, `truck` = 7), so `classes = "coco80"` in the deployment manifest
- **export** `training/deim/export_onnx.py --config …/DEIM/configs/deim_dfine/dfine_hgnetv2_s_coco.yml --weights …/dev-cache/train/weights/dfine_s_coco.pth --out …/dfine-s-coco --keep-class0` (COCO's column 0 is `person`, so nothing is dropped) — the TorchScript exporter traced at batch 2 with a symbolic batch, opset 17, `onnxslim`; the sibling static `dfine-s-coco.onnx` (sha256 `e83d4c0bc836558105f49216019288cac98cadbc9b904715b56473c7471dcb17`) is the same graph with the batch bound. Self-check: PyTorch vs onnxruntime on 3 gold frames, static and dynamic at batch 1 and 3, max |Δ| 3.5e-5 on scores and boxes, every detection ≥ 0.3 matched at IoU 1.0. (Rebatching the older static `dev-cache/detr/dfine_s_slim.onnx` by hand agrees with this file to 3.3e-3 on scores / 2e-4 on boxes at batch 3, but the export above is the recipe every DETR model of ours follows.)
- **shape** `images [batch,3,640,640] → logits [batch,300,80] + boxes [batch,300,4]` — a DETR head (raw logits, sigmoid on the consumer's side, top class per query, no NMS; cxcywh normalised by the input square); `Detector::load` recognises the two outputs and runs it with the stretch preprocessing below (`crates/ojas-vision/src/detr.rs`)
- **preprocessing** stretch to 640×640 (no letterbox), RGB, 0–1, no mean/std
- **validated** `crates/ojas-vision/tests/detr_detector.rs`: engine `Detector` (CPU) == the survey's raw-executor decode on 3 Field gold frames (290 boxes at conf 0.25, exact); CUDA (f16, batch 3 through the 4-plan) vs CPU: every box ≥ 0.35 on either side has a twin within 1.6 px / 0.045 score (gate 3 px / 0.08)

## Planned (not yet exported — see the research notes in the vision plan)

| Role | Candidate | License | Status |
|---|---|---|---|
| Vehicle detector (production) | YOLOX-s / RTMDet-s fine-tuned on IDD + DriveIndia with `auto_rickshaw` | Apache-2.0 | dataset access forms pending |
| Plate detector | open-image-models `yolo-v9-t-640-license-plate` (strip in-graph NMS tail) or YOLOX-nano fine-tune | MIT / Apache-2.0 | to fetch + re-export |
| Plate OCR | PP-OCRv5_mobile_rec fine-tuned, dict `0-9A-Z`, 1×3×48×320 static | Apache-2.0 | fine-tune pending; Awiros/anpr-ocr (Apache-2.0, Indian plates) as teacher |

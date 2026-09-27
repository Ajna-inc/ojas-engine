# DEIM training (official trainer, `…/dev-cache/train/DEIM`)

Two local patches were needed on this box (torch 2.8 / torchvision 0.23):

1. `engine/data/transforms/_transforms.py` — torchvision ≥ 0.18 renamed `Transform._transform`
   to `transform`; the checkout predates that, so the three custom transforms are aliased at the
   bottom of the file (upstream D-FINE already defines both names).
2. Loading D-FINE weights (`dfine_s_obj2coco.pth`, trained with ReLU decoder activations) into
   DEIM's model needs `DFINETransformer: {activation: relu, mlp_act: relu}` — DEIM's base config
   switches to SiLU, and the mismatch yields invalid boxes at iteration 0 (an assertion in the
   matcher, not a NaN).

The pipeline is resumable exactly like `training/rtdetrv2/`: rerun `step2_pipeline.sh` after any
interruption. It exports `PYTHONUNBUFFERED=1` because the trainer's progress lines are otherwise
held in a block buffer when redirected to a file.

## Export to the engine

`export_onnx.py --config CFG.yml --weights ck.pth --out DIR/name` turns a checkpoint into the ONNX
files the engine imports (`name-dyn.onnx` with a symbolic batch for `models/`, `name.onnx` static
batch 1 for the probes, `name.classes.json`), with an onnxruntime self-check against PyTorch on gold
frames. The engine-side gate is `crates/ojas-vision/examples/detr_parity.rs`; recipe, numbers and pitfalls are recorded with the export.

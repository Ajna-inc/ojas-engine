# Reference mAP for the Rust D-FINE evaluator: DEIM's model, val transform (Resize 640, PIL
# bilinear) and post-processor with the step-3 run's EMA weights, scored by pycocotools on the
# first N images of uvh_val_st.json. `python eval_ref.py [N] [checkpoint]`
import sys, json, torch
sys.path.insert(0, "/workspace/DEIM")
from engine.core import YAMLConfig
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image
import torchvision.transforms.v2 as T
N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
CKPT = sys.argv[2] if len(sys.argv) > 2 else "/workspace/out/step3_ojas_n32/last.pth"
cfg = YAMLConfig("/workspace/DEIM/configs/uvh/step3_ojas_n32.yml")
model, post = cfg.model, cfg.postprocessor
sd = torch.load(CKPT, map_location="cpu")
model.load_state_dict(sd["ema"]["module"])
model.eval(); post.eval()
ann = json.load(open("/workspace/data/annotations/uvh_val_st.json"))
imgs = ann["images"][:N]
ids = {im["id"] for im in imgs}
sub = dict(ann, images=imgs, annotations=[a for a in ann["annotations"] if a["image_id"] in ids])
json.dump(sub, open("/tmp/val_sub.json", "w"))
gt = COCO("/tmp/val_sub.json")
tf = T.Compose([T.Resize((640, 640)), T.ToImage(), T.ToDtype(torch.float32, scale=True)])
import glob, os
index = {os.path.basename(p): p for p in glob.glob("/workspace/data/images/**/*", recursive=True)}
dets = []
with torch.no_grad():
    for im in imgs:
        pil = Image.open(index[os.path.basename(im["file_name"])]).convert("RGB")
        x = tf(pil)[None]
        out = post(model(x), torch.tensor([[im["width"], im["height"]]]))[0]
        for l, b, s in zip(out["labels"].tolist(), out["boxes"].tolist(), out["scores"].tolist()):
            dets.append({"image_id": im["id"], "category_id": l, "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "score": s})
e = COCOeval(gt, gt.loadRes(dets), "bbox"); e.evaluate(); e.accumulate(); e.summarize()
print(f"PyTorch mAP {e.stats[0]:.4f} on {N} val images")

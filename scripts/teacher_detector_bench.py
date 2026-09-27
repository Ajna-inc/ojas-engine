#!/usr/bin/env python3
"""Run released DETR-family teachers with strict official model loading and common metrics."""
import contextlib
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path('evidence/teacher_benchmark_2026_09_20').resolve()
BASE = Path('/data/datasets/iisc-aim')
TRAIN = Path('/data/dev-cache/train')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def load_model(name, checkpoint_override=None):
    if checkpoint_override is not None:
        assert name in ['rtdetr_s', 'rtdetr_x'], 'checkpoint override supports RT-DETR only'
    if name in ['rtdetr_s', 'rtdetr_x']:
        repo = TRAIN / 'RT-DETR/rtdetrv2_pytorch'
        suffix = 'S' if name == 'rtdetr_s' else 'X'
        checkpoint = BASE / f'models-UVH-26/weights/RT-DETRv2-{suffix}/UVH-26-MV-RT-DETRv2-{suffix}.pth'
        if checkpoint_override is not None:
            checkpoint = Path(checkpoint_override)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)['ema']['module']
        depth = 18 if suffix == 'S' else (101 if max(int(k.split('.')[4]) for k in state if k.startswith('backbone.res_layers.2.blocks.')) > 5 else 50)
        file = 'rtdetrv2_r18vd_120e_coco.yml' if depth == 18 else ('rtdetrv2_r101vd_6x_coco.yml' if depth == 101 else 'rtdetrv2_r50vd_6x_coco.yml')
        config = repo / 'configs/rtdetrv2' / file
        sys.path.insert(0, str(repo))
        from src.core import YAMLConfig
        cfg = YAMLConfig(str(config))
        cfg.yaml_cfg['num_classes'] = 15
        cfg.yaml_cfg['remap_mscoco_category'] = False
        cfg.yaml_cfg['PResNet']['pretrained'] = False
        if depth != 18:
            width = state['encoder.input_proj.0.conv.weight'].shape[0]
            cfg.yaml_cfg['HybridEncoder'].update(hidden_dim=width, dim_feedforward=2048 if width == 384 else 1024)
            cfg.yaml_cfg['RTDETRTransformerv2']['feat_channels'] = [width] * 3
        mapping = list(range(15))
    elif name == 'dfine_x':
        repo = TRAIN / 'D-FINE'
        checkpoint = BASE / 'models-BMD-45/weights/D-FINE/best_stg1.pth'
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)['ema']['module']
        sys.path.insert(0, str(repo))
        from src.core import YAMLConfig
        config = repo / 'configs/dfine/dfine_hgnetv2_x_coco.yml'
        cfg = YAMLConfig(str(config))
        cfg.yaml_cfg['num_classes'] = 13
        cfg.yaml_cfg['remap_mscoco_category'] = False
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False
        mapping = list(range(1, 14))
    else:
        raise ValueError(name)
    model = cfg.model
    model.load_state_dict(state, strict=True)
    del state
    parameters = sum(p.numel() for p in model.parameters())
    model.eval().cuda()
    return model, mapping, {'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint),
        'config': str(config), 'resolved_config': cfg.yaml_cfg, 'parameters': parameters,
        'checkpoint_bytes_including_training_state': checkpoint.stat().st_size,
        'strict_loading': True, 'raw_to_canonical': mapping}

def overlap(box, boxes):
    inter = np.maximum(0, np.minimum(box[2:], boxes[:, 2:]) - np.maximum(box[:2], boxes[:, :2])).prod(1)
    area = np.maximum(0, box[2:] - box[:2]).prod()
    areas = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(1)
    return inter / np.maximum(area + areas - inter, 1e-8)

def main():
    global ROOT
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['rtdetr_s', 'rtdetr_x', 'dfine_x'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output-name')
    args = parser.parse_args()
    name = args.model
    ROOT = args.root.resolve()
    output_name = args.output_name or name
    assert Path(output_name).name == output_name
    output = ROOT / f'{output_name}.json'
    assert not output.exists(), 'do not overwrite benchmark'
    contract = json.loads((ROOT / 'contract.json').read_text())
    assert sha(contract['annotations']) == contract['annotations_sha256'], 'annotations changed'
    annotations = json.loads(Path(contract['annotations']).read_text())
    image_map = {r['id']: r for r in contract['images']}
    by_image = {i: [] for i in image_map}
    for ann in annotations['annotations']:
        if ann['image_id'] in by_image:
            by_image[ann['image_id']].append(ann)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, mapping, metadata = load_model(name, args.checkpoint)
    valid_raw = [i for i, c in enumerate(mapping) if 1 <= c <= 14]
    canonical = np.array([mapping[i] for i in valid_raw])
    metadata['torch_version'] = torch.__version__
    metadata['gpu'] = torch.cuda.get_device_name(0)
    metadata['precision'] = 'FP32, TF32 disabled'
    metadata['deploy_fusion'] = False
    metadata['script_sha256'] = sha(__file__)
    metadata['contract_sha256'] = sha(ROOT / 'contract.json')
    detections, per_image, cached, crop_manifest = [], [], {}, []
    latencies = []
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        first = Image.open(contract['images'][0]['path']).convert('RGB').resize((640, 640), Image.Resampling.BILINEAR)
        x = torch.from_numpy(np.array(first).transpose(2, 0, 1).copy()).float().div_(255).unsqueeze(0).cuda()
        for _ in range(10): model(x)
        for _ in range(50):
            torch.cuda.synchronize(); start = time.perf_counter(); result = model(x); torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)
        del result, x
        start_eval = time.perf_counter()
        for index, (id, info) in enumerate(image_map.items()):
            assert sha(info['path']) == info['sha256'], 'evaluation image changed'
            with Image.open(info['path']) as im:
                rgb = im.convert('RGB')
                assert rgb.size == (info['width'], info['height'])
                resized = rgb.resize((640, 640), Image.Resampling.BILINEAR)
            x = torch.from_numpy(np.array(resized).transpose(2, 0, 1).copy()).float().div_(255).unsqueeze(0).cuda()
            out = model(x)
            logits = out['pred_logits'][0].float().cpu().numpy()
            boxes = out['pred_boxes'][0].float().cpu().numpy()
            assert np.isfinite(logits).all() and np.isfinite(boxes).all()
            assert logits.shape[1] == len(mapping)
            xyxy = np.concatenate([boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2], axis=1)
            xyxy *= np.array([info['width'], info['height']] * 2)
            scores = torch.from_numpy(logits[:, valid_raw]).sigmoid().numpy()
            # AP keeps top300 query-by-class hypotheses; top1 diagnostic uses unique queries.
            flat = np.argsort(-scores.ravel(), kind='stable')[:300]
            for f in flat:
                q, c = divmod(int(f), len(valid_raw)); b = xyxy[q]
                detections.append({'image_id': id, 'category_id': int(canonical[c]), 'score': float(scores[q, c]),
                    'bbox': [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]})
            labels = canonical[scores.argmax(1)]
            confidence = scores.max(1)
            gt = by_image[id]
            gt_boxes = np.array([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]] for a in gt], dtype=float).reshape(-1, 4)
            used, matches, proposals = set(), [], []
            for q in np.argsort(-confidence, kind='stable'):
                if confidence[q] < .3: continue
                b = xyxy[q]; match = None
                if len(gt):
                    ious = overlap(b, gt_boxes)
                    for j in used: ious[j] = -1
                    j = int(ious.argmax())
                    if ious[j] >= .5: used.add(j); match = gt[j]
                proposal = {'query': int(q), 'box': b.tolist(), 'class': int(labels[q]), 'score': float(confidence[q]),
                    'annotation_id': match['id'] if match else None, 'truth': match['category_id'] if match else None}
                proposals.append(proposal)
                if match and 1 <= match['category_id'] <= 4:
                    matches.append({'annotation_id': match['id'], 'truth': match['category_id'], 'predicted': int(labels[q]), 'query': int(q)})
                if name == 'rtdetr_s' and 1 <= labels[q] <= 4 and b[2] - b[0] >= 48 and b[3] - b[1] >= 48:
                    crop_manifest.append(dict(proposal, image_id=id, image_path=info['path'], image_sha256=info['sha256']))
            per_image.append({'id': id, 'passenger_gt': sum(1 <= a['category_id'] <= 4 for a in gt),
                'localized': len(matches), 'correct': sum(r['truth'] == r['predicted'] for r in matches), 'matches': matches})
            cached[str(id)] = {'logits': logits.tolist(), 'boxes_cxcywh_normalized': boxes.tolist(), 'proposals': proposals}
            if (index + 1) % 64 == 0: print(name, index + 1, '/', len(image_map), flush=True)
    metadata['evaluation_seconds_including_decode_preprocess_transfers_matching'] = time.perf_counter() - start_eval
    metadata['forward_batch1_ms'] = {'p50': float(np.median(latencies)), 'p95': float(np.percentile(latencies, 95)), 'warmups': 10, 'iterations': 50}
    metadata['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    metadata['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    # Standard COCO evaluator; baseline rerun with this same stack for comparability.
    coco = COCO()
    coco.dataset = {'images': [r for r in annotations['images'] if r['id'] in image_map],
        'categories': annotations['categories'], 'annotations': [dict(a, iscrowd=a.get('iscrowd', 0), area=a.get('area', a['bbox'][2] * a['bbox'][3])) for a in annotations['annotations'] if a['image_id'] in image_map], 'info': {}}
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex(); result = coco.loadRes(detections)
        ev = COCOeval(coco, result, 'bbox'); ev.params.imgIds = list(image_map); ev.params.catIds = list(range(1, 15)); ev.evaluate(); ev.accumulate(); ev.summarize()
    total = sum(r['passenger_gt'] for r in per_image)
    localized = sum(r['localized'] for r in per_image)
    correct = sum(r['correct'] for r in per_image)
    per_class_ap = []
    for index, category in enumerate(ev.params.catIds):
        values = ev.eval['precision'][:, :, index, 0, -1]
        valid = values[values >= 0]
        per_class_ap.append({'class': int(category), 'AP': float(valid.mean()) if len(valid) else None,
            'gt': sum(a['category_id'] == category for a in coco.dataset['annotations'])})
    report = {'name': name, 'metadata': metadata, 'AP': float(ev.stats[0]), 'AP50': float(ev.stats[1]),
        'passenger_gt': total, 'localized': localized, 'correct': correct, 'conditional_accuracy': correct / max(1, localized),
        'localized_fraction': localized / max(1, total), 'correct_fraction_all_gt': correct / max(1, total),
        'per_class_ap': per_class_ap, 'per_image': per_image}
    output.write_text(json.dumps(report, indent=2, default=str) + '\n')
    import gzip
    with gzip.open(ROOT / f'{output_name}_raw.json.gz', 'wt') as f: json.dump(cached, f)
    if name == 'rtdetr_s':
        crop_path = ROOT / ('baseline_crops.json' if output_name == 'rtdetr_s' else f'{output_name}_crops.json')
        assert not crop_path.exists(), 'do not overwrite proposal cache'
        crop_path.write_text(json.dumps(crop_manifest, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'per_image'}, default=str), flush=True)

if __name__ == '__main__': main()

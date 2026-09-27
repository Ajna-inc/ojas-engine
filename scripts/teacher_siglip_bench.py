#!/usr/bin/env python3
"""Large zero-shot crop teacher: fixed prompts, inference-only baseline routing."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel, AutoProcessor

ROOT = Path('evidence/teacher_benchmark_2026_09_20').resolve()

def main():
    assert not (ROOT / 'siglip2.json').exists(), 'do not overwrite result'
    download = json.loads((ROOT / 'siglip_download.json').read_text())
    contract = json.loads((ROOT / 'contract.json').read_text())
    rows = json.loads((ROOT / 'baseline_crops.json').read_text())
    torch.set_num_threads(4)
    processor = AutoProcessor.from_pretrained(download['path'], local_files_only=True, trust_remote_code=False)
    processor.image_processor = AutoImageProcessor.from_pretrained(download['path'], local_files_only=True, use_fast=False)
    model = AutoModel.from_pretrained(download['path'], torch_dtype=torch.bfloat16,
        local_files_only=True, trust_remote_code=False, attn_implementation='sdpa').eval().cuda()
    total_params = sum(p.numel() for p in model.parameters())
    vision_params = sum(p.numel() for p in model.vision_model.parameters())
    with torch.inference_mode():
        text_inputs = processor(text=contract['prompts'], padding='max_length', max_length=64, return_tensors='pt').to('cuda')
        text_features = torch.nn.functional.normalize(model.get_text_features(**text_inputs).float(), dim=-1)
        scale, bias = model.logit_scale.float().exp(), model.logit_bias.float()
    model.text_model = None  # cached text embeddings; only vision runs per crop
    torch.cuda.empty_cache()
    image_cache = {}
    def crop(row):
        path = row['image_path']
        if path not in image_cache:
            image_cache.clear()
            with Image.open(path) as im: image_cache[path] = im.convert('RGB')
        im = image_cache[path]
        b = row['box']; w, h = b[2] - b[0], b[3] - b[1]
        geometry = [int(max(0, min(im.width, b[0] - .15*w))), int(max(0, min(im.height, b[1] - .15*h))),
                    int(max(0, min(im.width, b[2] + .15*w))), int(max(0, min(im.height, b[3] + .15*h)))]
        assert geometry[2] > geometry[0] and geometry[3] > geometry[1]
        return im.crop(geometry)
    def inputs(images):
        return processor(images=images, return_tensors='pt')['pixel_values'].to(device='cuda', dtype=torch.bfloat16)
    def forward(pixels):
        features = torch.nn.functional.normalize(model.get_image_features(pixel_values=pixels).float(), dim=-1)
        return features @ text_features.T * scale + bias, features
    timing, saved, feature_rows = [], [], []
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        x = inputs([crop(rows[0])])
        for _ in range(10): forward(x)
        for _ in range(50):
            torch.cuda.synchronize(); start = time.perf_counter(); forward(x); torch.cuda.synchronize()
            timing.append((time.perf_counter() - start)*1000)
        del x
        start = time.perf_counter()
        for i in range(0, len(rows), 8):
            batch = rows[i:i+8]
            logits, features = forward(inputs([crop(row) for row in batch]))
            assert torch.isfinite(logits).all()
            probs = logits.softmax(-1).cpu().numpy()
            values = logits.cpu().numpy()
            feature_rows.extend(features.cpu().numpy().astype(np.float16))
            for row, p, scores in zip(batch, probs, values):
                saved.append(dict(row, predicted=int(p.argmax())+1, probabilities=p.tolist(), logits=scores.tolist()))
            if i % 128 == 0: print('SigLIP2', i+len(batch), '/', len(rows), flush=True)
    elapsed = time.perf_counter()-start
    eligible = [r for r in saved if r['truth'] in [1,2,3,4]]
    cm = np.zeros((4,4), dtype=int)
    for r in eligible: cm[r['truth']-1,r['predicted']-1] += 1
    report = {'model': download, 'total_parameters': total_params, 'vision_parameters': vision_params,
        'text_embeddings_cached': True, 'precision': 'BF16 vision, FP32 similarity',
        'prompts': contract['prompts'], 'preprocessing': processor.image_processor.to_dict(),
        'routed_proposals': len(rows), 'matched_passengers': len(eligible), 'correct': int(cm.trace()),
        'accuracy': float(cm.trace()/len(eligible)), 'confusion': cm.tolist(),
        'detector_correct_same_objects': sum(r['class']==r['truth'] for r in eligible),
        'fixed': sum(r['class']!=r['truth'] and r['predicted']==r['truth'] for r in eligible),
        'broken': sum(r['class']==r['truth'] and r['predicted']!=r['truth'] for r in eligible),
        'forward_batch1_ms': {'p50':float(np.median(timing)), 'p95':float(np.percentile(timing,95)), 'warmups':10, 'iterations':50},
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(), 'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'evaluation_seconds_including_preprocessing_transfers':elapsed,
        'scope':'Zero-shot four-way classification of baseline-routed actual boxes. Softmax is not calibrated confidence. All proposals processed before GT-only scoring. No detection AP or miss-recovery claim.',
        'gpu':torch.cuda.get_device_name(0), 'torch_version':torch.__version__,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'contract_sha256':hashlib.sha256((ROOT/'contract.json').read_bytes()).hexdigest(),
        'baseline_crops_sha256':hashlib.sha256((ROOT/'baseline_crops.json').read_bytes()).hexdigest()}
    (ROOT/'siglip2_predictions.json').write_text(json.dumps(saved,indent=2)+'\n')
    np.savez_compressed(ROOT/'siglip2_features.npz', features=np.array(feature_rows), text_features=text_features.cpu().numpy())
    (ROOT/'siglip2.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    print(json.dumps(report,default=str),flush=True)

if __name__ == '__main__': main()

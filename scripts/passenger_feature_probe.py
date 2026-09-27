#!/usr/bin/env python3
"""Frozen DINOv2-small features + fixed-budget supervised four-way linear probes."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path('evidence/passenger_dinov2_2026_09_20').resolve()
MANIFEST = Path('evidence/passenger_screen_2026_09_20/crops.jsonl').resolve()
PROPOSALS = Path('evidence/teacher_benchmark_2026_09_20/baseline_crops.json').resolve()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2) + '\n')


def prepare():
    from huggingface_hub import HfApi, snapshot_download
    ROOT.mkdir(exist_ok=True)
    assert not (ROOT / 'contract.json').exists(), 'Contract already exists'
    model_id = 'facebook/dinov2-small'
    revision = HfApi().model_info(model_id).sha
    path = snapshot_download(model_id, revision=revision, cache_dir=str(ROOT / 'model_cache'),
                             allow_patterns=['config.json', 'model.safetensors', 'preprocessor_config.json'])
    rows = [json.loads(x) for x in MANIFEST.read_text().splitlines()]
    rows = [r for r in rows if r['split'] in ['train', 'dev']]
    proposals = json.loads(PROPOSALS.read_text())
    assert not ({r['source_sha256'] for r in rows} & {r['image_sha256'] for r in proposals}), 'Exact source overlap'
    contract = dict(model_id=model_id, revision=revision, model_path=path,
        model_files={p.name: sha(p) for p in Path(path).iterdir() if p.is_file()},
        manifest_sha256=sha(MANIFEST), proposals_sha256=sha(PROPOSALS),
        train_n=sum(r['split'] == 'train' for r in rows), dev_n=sum(r['split'] == 'dev' for r in rows),
        proposals=len(proposals), seeds=[1, 2], epochs=100, batch=128,
        optimizer='AdamW lr=0.001 weight_decay=0.0001, cosine to zero',
        features='L2-normalized concatenation of final CLS and mean patch tokens; frozen BF16 encoder',
        preprocessing='224px black letterbox, PIL bilinear resize, RGB/255 and official processor mean/std',
        selection='Fixed final epoch 100; no selection or tuning on development or benchmark',
        scope='Reused 512-image teacher benchmark; raw argmax on all baseline-routed matched passenger crops, with original detector comparison on identical objects. Different cohort from ResNet depth experiment. No deployment or miss-recovery claim.',
        script_sha256=sha(__file__))
    write('contract.json', contract)
    (ROOT / 'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(contract), flush=True)


class Crops(Dataset):
    def __init__(self, rows, mean, std, proposals=False):
        self.rows, self.mean, self.std, self.proposals = rows, mean, std, proposals
        self.cached_path, self.cached_image = None, None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        if self.proposals:
            if r['image_path'] != self.cached_path:
                content = Path(r['image_path']).read_bytes()
                assert hashlib.sha256(content).hexdigest() == r['image_sha256']
                with Image.open(io.BytesIO(content)) as im:
                    self.cached_image = im.convert('RGB')
                self.cached_path = r['image_path']
            im = self.cached_image
            b = r['box']; w, h = b[2] - b[0], b[3] - b[1]
            geometry = [int(max(0, min(im.width, b[0] - .15*w))), int(max(0, min(im.height, b[1] - .15*h))),
                        int(max(0, min(im.width, b[2] + .15*w))), int(max(0, min(im.height, b[3] + .15*h)))]
            assert geometry[2] > geometry[0] and geometry[3] > geometry[1]
            im = im.crop(geometry)
        else:
            content = Path(r['path']).read_bytes()
            assert hashlib.sha256(content).hexdigest() == r['crop_sha256']
            with Image.open(io.BytesIO(content)) as opened:
                im = opened.convert('RGB')
        scale = 224 / max(im.size)
        size = (max(1, round(im.width * scale)), max(1, round(im.height * scale)))
        canvas = Image.new('RGB', (224, 224))
        canvas.paste(im.resize(size, Image.Resampling.BILINEAR), ((224-size[0])//2, (224-size[1])//2))
        x = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255
        return torch.from_numpy((x - self.mean) / self.std)


def metrics(probs, labels):
    pred = probs.argmax(1)
    cm = np.zeros((4, 4), dtype=int)
    np.add.at(cm, (labels, pred), 1)
    den = cm.sum(0) + cm.sum(1)
    return dict(n=len(labels), correct=int(cm.trace()), accuracy=float(cm.trace()/len(labels)),
        macro_f1=float(np.mean(np.divide(2*cm.diagonal(), den, out=np.zeros(4), where=den>0))),
        per_class_recall=(cm.diagonal()/np.maximum(cm.sum(1), 1)).tolist(), confusion=cm.tolist(),
        nll=float(-np.log(np.maximum(probs[np.arange(len(labels)), labels], 1e-30)).mean()))


def run():
    from transformers import AutoImageProcessor, Dinov2Model
    assert not (ROOT / 'results.json').exists(), 'Do not overwrite results'
    c = json.loads((ROOT / 'contract.json').read_text())
    assert sha(MANIFEST) == c['manifest_sha256'] and sha(PROPOSALS) == c['proposals_sha256']
    assert sha(__file__) == c['script_sha256'], 'Script changed after contract freeze'
    for name, digest in c['model_files'].items():
        assert sha(Path(c['model_path']) / name) == digest
    torch.set_num_threads(4)
    rows = [json.loads(x) for x in MANIFEST.read_text().splitlines()]
    rows = [r for r in rows if r['split'] in ['train', 'dev']]
    proposals = json.loads(PROPOSALS.read_text())
    processor = AutoImageProcessor.from_pretrained(c['model_path'], local_files_only=True, use_fast=False)
    mean = np.array(processor.image_mean, dtype=np.float32).reshape(3,1,1)
    std = np.array(processor.image_std, dtype=np.float32).reshape(3,1,1)
    model = Dinov2Model.from_pretrained(c['model_path'], local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa').eval().cuda()
    model.requires_grad_(False)
    parameters = sum(p.numel() for p in model.parameters())
    def forward(x):
        tokens = model(pixel_values=x).last_hidden_state.float()
        return torch.nn.functional.normalize(torch.cat([tokens[:,0], tokens[:,1:].mean(1)], 1), dim=1)
    datasets = [Crops(rows, mean, std), Crops(proposals, mean, std, True)]
    cache = ROOT / 'features.npz'
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        x = datasets[0][0].unsqueeze(0).cuda().bfloat16()
        for _ in range(10): forward(x)
        timings = []
        for _ in range(50):
            torch.cuda.synchronize(); start = time.perf_counter(); forward(x); torch.cuda.synchronize()
            timings.append((time.perf_counter()-start)*1000)
        if cache.exists():
            stored = np.load(cache); source, test = stored['source'], stored['test']
        else:
            features = []
            start = time.perf_counter()
            for ds in datasets:
                extracted = []
                for i, x in enumerate(DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)):
                    f = forward(x.cuda().bfloat16()).cpu().numpy()
                    assert np.isfinite(f).all()
                    extracted.append(f)
                    if i % 16 == 0: print('features', min((i+1)*32, len(ds)), '/', len(ds), flush=True)
                features.append(np.concatenate(extracted))
            source, test = features
            np.savez_compressed(cache, source=source, test=test)
            write('extraction.json', dict(seconds=time.perf_counter()-start, features_sha256=sha(cache), contract_sha256=sha(ROOT/'contract.json')))
    extraction = json.loads((ROOT/'extraction.json').read_text())
    assert extraction['features_sha256'] == sha(cache) and extraction['contract_sha256'] == sha(ROOT/'contract.json')
    memory = torch.cuda.max_memory_allocated()
    del model
    torch.cuda.empty_cache()
    # Heads train on CPU; encoder features and both evaluation partitions stay frozen.
    train_mask = np.array([r['split'] == 'train' for r in rows])
    labels = np.array([r['label'] for r in rows])
    truth = np.array([(r['truth'] or 0)-1 for r in proposals])
    eligible = (truth >= 0) & (truth < 4)
    baseline = np.array([r['class']-1 for r in proposals])
    x = torch.from_numpy(source[train_mask]); y = torch.from_numpy(labels[train_mask])
    reports = {}
    for seed in c['seeds']:
        torch.manual_seed(seed)
        head = torch.nn.Linear(x.shape[1], 4)
        opt = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=c['epochs'])
        history = []
        for epoch in range(c['epochs']):
            order = torch.randperm(len(x)); total = 0.
            for idx in order.split(c['batch']):
                opt.zero_grad(); loss = torch.nn.functional.cross_entropy(head(x[idx]), y[idx])
                assert torch.isfinite(loss)
                loss.backward(); opt.step(); total += loss.detach().item()*len(idx)
            scheduler.step(); history.append(total/len(x))
        with torch.inference_mode():
            train_probs = head(x).softmax(1).numpy()
            dev_probs = head(torch.from_numpy(source[~train_mask])).softmax(1).numpy()
            probs = head(torch.from_numpy(test)).softmax(1).numpy()
        reports[f'seed{seed}'] = dict(train=metrics(train_probs, labels[train_mask]),
            dev=metrics(dev_probs, labels[~train_mask]), actual_boxes=metrics(probs[eligible], truth[eligible]),
            loss_history=history, head_parameters=sum(p.numel() for p in head.parameters()))
        pred = probs.argmax(1)
        counts = {}
        for i, r in enumerate(proposals):
            counts.setdefault(r['image_id'], np.zeros(2))
            if eligible[i]: counts[r['image_id']] += [1, int(pred[i]==truth[i])-int(baseline[i]==truth[i])]
        counts = np.array(list(counts.values())); rng = np.random.default_rng(20260920)
        samples = np.array([counts[rng.integers(len(counts), size=len(counts))].sum(0) for _ in range(10000)])
        reports[f'seed{seed}']['paired_vs_detector'] = dict(delta=float(counts[:,1].sum()/counts[:,0].sum()),
            image_bootstrap95=np.percentile(samples[:,1]/samples[:,0], [2.5,97.5]).tolist(),
            caveat='Image bootstrap does not account for camera/video correlation; includes only routed matched passengers.')
        np.savez_compressed(ROOT/f'seed{seed}_predictions.npz', probabilities=probs)
        torch.save(head.state_dict(), ROOT/f'seed{seed}_head.pth')
        print('head',seed,reports[f'seed{seed}']['actual_boxes'],flush=True)
    result = dict(runs=reports, encoder_parameters=parameters,
        detector_correct=int((baseline[eligible]==truth[eligible]).sum()), eligible=int(eligible.sum()),
        detector_accuracy=float((baseline[eligible]==truth[eligible]).mean()),
        timing=dict(p50_ms=float(np.median(timings)), p95_ms=float(np.percentile(timings,95)),
                    scope='BF16 batch-one encoder GPU forward plus feature pooling, 224px, 10 warmups/50 iterations; excludes crop preprocessing and head'),
        peak_allocated_bytes=memory, gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__,
        contract_sha256=sha(ROOT/'contract.json'), features_sha256=sha(cache), scope=c['scope'])
    write('results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'run'])
    args = parser.parse_args()
    prepare() if args.mode == 'prepare' else run()

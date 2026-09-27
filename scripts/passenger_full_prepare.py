#!/usr/bin/env python3
"""Uncapped original-resolution crops on the frozen full-detector split."""
import hashlib
import io
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path('evidence/passenger_full_2026_09_20').resolve()
SPLITS = Path('evidence/iisc_small_full_2026_09_20').resolve()
BASE = Path('/data/datasets/iisc-aim')
NAMES = ['Hatchback', 'Sedan', 'SUV', 'MUV']

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def geometry(box, width, height):
    x, y, w, h = np.asarray(box, dtype=np.float32)
    return tuple(int(v) for v in (max(0, x-.15*w), max(0, y-.15*h),
                                 min(width, x+w+.15*w), min(height, y+h+.15*h)))

def main():
    ROOT.mkdir(exist_ok=False)
    (ROOT/'crops').mkdir()
    annotations = BASE/'UVH-26/UVH-26-Train/UVH-26-MV-Train.json'
    coco = json.loads(annotations.read_text())
    assert {c['id']: c['name'] for c in coco['categories'] if c['id'] in range(1,5)} == dict(enumerate(NAMES,1))
    by_image = defaultdict(list)
    for a in coco['annotations']:
        if 1 <= a['category_id'] <= 4 and not a.get('iscrowd', 0):
            by_image[a['image_id']].append(a)
    jobs = [(s, r) for s in ['train', 'dev'] for r in json.loads((SPLITS/f'{s}_manifest.json').read_text())]
    assert len({r['id'] for _,r in jobs}) == len(jobs)
    def export(job):
        split, source = job
        raw = Path(source['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == source['sha256']
        rows, excluded = [], Counter()
        with Image.open(io.BytesIO(raw)) as im:
            rgb = im.convert('RGB')
            assert rgb.size == (source['width'], source['height'])
            for a in by_image[source['id']]:
                if min(a['bbox'][2:]) < 48:
                    excluded['under48'] += 1
                    continue
                b = geometry(a['bbox'], *rgb.size)
                if b[2] <= b[0]+8 or b[3] <= b[1]+8:
                    excluded['invalid_crop'] += 1
                    continue
                path = ROOT/'crops'/f"{source['id']}_{a['id']}.jpg"
                rgb.crop(b).save(path, quality=80, subsampling=0)
                c = a['category_id']
                rows.append(dict(version=2, dataset='uvh-mv-train', path=str(path),
                    label=c-1, class_name=NAMES[c-1], raw_category_id=c,
                    image_id=source['id'], annotation_id=a['id'], box=a['bbox'],
                    crop_xyxy=b, context=.15, width=b[2]-b[0], height=b[3]-b[1],
                    source=source['path'], source_sha256=source['sha256'],
                    crop_sha256=sha(path), group=source['group'], split=split))
        return rows, excluded
    rows, excluded = [], Counter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, (part, counts) in enumerate(pool.map(export,jobs)):
            rows.extend(part); excluded.update(counts)
            if (i+1)%1000 == 0: print('exported sources',i+1,'crops',len(rows),flush=True)
    # Exclude whole training groups whose exact crop bytes also occur in development.
    dev_hashes = {r['crop_sha256'] for r in rows if r['split']=='dev'}
    quarantine = {r['group'] for r in rows if r['split']=='train' and r['crop_sha256'] in dev_hashes}
    rows = [r for r in rows if r['group'] not in quarantine]
    full = ROOT/'full.jsonl'
    full.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    # Nested group-sampled control: same development set, no category caps.
    groups = sorted({r['group'] for r in rows if r['split']=='train'},
                    key=lambda g: hashlib.sha256(('small:'+g).encode()).hexdigest())
    counts = Counter(r['group'] for r in rows if r['split']=='train')
    selected, n = set(), 0
    for g in groups:
        selected.add(g); n += counts[g]
        if n >= 3961: break
    small = [r for r in rows if r['split']=='dev' or r['group'] in selected]
    (ROOT/'small.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in small))
    init = str(BASE/'models-UVH-26/weights/RT-DETRv2-S/UVH-26-MV-RT-DETRv2-S.pth')
    for arm in ['small','full']:
        for seed in [1,2]:
            cfg=dict(manifest=str(ROOT/f'{arm}.jsonl'),init=init,batch=16,size=224,
                epochs=8,epoch_samples=3961,lr=.0001,head_lr=.001,balanced=True,
                out=str(ROOT/f'{arm}_seed{seed}'),seed=seed,overfit_steps=0,
                backbone_stage=3,evaluate_train=True)
            (ROOT/f'{arm}_seed{seed}.json').write_text(json.dumps(cfg,indent=2)+'\n')
    audit=dict(annotations=str(annotations),annotations_sha256=sha(annotations),
        split_hashes={s:sha(SPLITS/f'{s}_manifest.json') for s in ['train','dev']},
        counts={a:dict(Counter(f"{r['split']}/{r['class_name']}" for r in data)) for a,data in [('full',rows),('small',small)]},
        excluded=dict(excluded),quarantined_groups=sorted(quarantine),
        manifests={a:sha(ROOT/f'{a}.jsonl') for a in ['small','full']},
        contract=dict(updates=1984,epochs=8,draws_per_epoch=3961,seeds=[1,2],
            selection='best development macro-F1; no benchmark selection',
            caveats=['Full-pool sampling does not visit every crop in this fixed-budget screen.',
                'Both arms regenerate crops with Pillow JPEG80 4:4:4; historical runs used Rust JPEG.',
                'Automatic source groups are not camera identity; IISc pretraining saw these development sources.',
                'Small control is newly sampled on this split, not the historical 3961-crop manifest.']))
    (ROOT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__': main()

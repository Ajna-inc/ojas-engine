#!/usr/bin/env python3
"""Versioned crop manifest: validate UVH labels and group source-image duplicates."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path

from PIL import Image

NAMES = ['Hatchback', 'Sedan', 'SUV', 'MUV']


def digest(data):
    return hashlib.sha256(data).hexdigest()


def signature(path):
    content = path.read_bytes()
    with Image.open(io.BytesIO(content)) as im:
        pixels = list(im.convert('L').resize((9, 8)).getdata())
    dhash = sum((pixels[y * 9 + x] > pixels[y * 9 + x + 1]) << (y * 8 + x) for y in range(8) for x in range(8))
    return digest(content), dhash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--crops', type=Path, required=True)
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20260920)
    p.add_argument('--max-sources', type=int, default=0, help='hash-sampled screening subset; 0 means all')
    a = p.parse_args()
    raw = a.annotations.read_bytes()
    coco = json.loads(raw)
    categories = {c['id']: c['name'] for c in coco['categories']}
    assert all(categories[i + 1] == name for i, name in enumerate(NAMES)), 'UVH category map mismatch'
    images = {im['id']: im for im in coco['images']}
    annotations = defaultdict(list)
    for ann in coco['annotations']:
        if not ann.get('iscrowd', 0):
            annotations[ann['image_id']].append(ann)
    with (a.crops / 'crops.tsv').open() as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    ids = sorted({int(r['image_id']) for r in rows}, key=lambda i: digest(f'{a.seed}:{i}'.encode()))
    if a.max_sources:
        ids = ids[:a.max_sources]
    ids = sorted(ids)
    wanted = set(ids)
    by_name = {}
    for path in a.images.rglob('*'):
        if path.is_file():
            assert path.name not in by_name, f'ambiguous source basename {path.name}'
            by_name[path.name] = path
    paths = [by_name[images[i]['file_name']] for i in ids]
    print(f'Hashing and grouping {len(ids)} source images', flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        signatures = list(pool.map(signature, paths))
    parent = list(range(len(ids)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        ri, rj = root(i), root(j)
        parent[max(ri, rj)] = min(ri, rj)
    # Four exact 16-bit buckets find every pair with <=3 differing dhash bits.
    buckets = defaultdict(list)
    exact = {}
    for i, (sha, dh) in enumerate(signatures):
        if sha in exact:
            union(i, exact[sha])
        exact[sha] = i
        candidates = set()
        for band in range(4):
            candidates.update(buckets[(band, (dh >> (16 * band)) & 65535)])
        for j in candidates:
            if (dh ^ signatures[j][1]).bit_count() <= 3:
                union(i, j)
        for band in range(4):
            buckets[(band, (dh >> (16 * band)) & 65535)].append(i)
    info = {}
    for j, image_id in enumerate(ids):
        group = f'uvh-mv-train:{ids[root(j)]}'
        split = 'dev' if int(digest(f'{a.seed}:{group}'.encode())[:8], 16) % 5 == 0 else 'train'
        info[image_id] = (group, split, str(paths[j]), signatures[j][0])
    result = []
    seen = set()
    for row in rows:
        image_id = int(row['image_id'])
        if image_id not in wanted:
            continue
        assert row['path'] not in seen, 'duplicate manifest crop path'
        seen.add(row['path'])
        class_id = int(row['class_id'])
        assert 1 <= class_id <= 4 and row['class'] == NAMES[class_id - 1]
        path = (a.crops / row['path']).resolve()
        crop_index = int(path.stem.rsplit('_', 1)[1])
        ann = annotations[image_id][crop_index]
        assert ann['category_id'] == class_id, 'crop/annotation label mismatch'
        content = path.read_bytes()
        with Image.open(io.BytesIO(content)) as crop:
            assert crop.size == (int(row['w']), int(row['h']))
        group, split, source, sha = info[image_id]
        result.append(dict(version=1, dataset='uvh-mv-train', path=str(path), label=class_id - 1,
                           class_name=row['class'], raw_category_id=class_id, image_id=image_id,
                           annotation_id=ann['id'], box=ann['bbox'], context=0.15,
                           width=int(row['w']), height=int(row['h']), source=source, source_sha256=sha,
                           crop_sha256=digest(content), group=group, split=split))
    # Crop byte duplicates also cannot cross partitions. Quarantine all involved source groups.
    by_crop = defaultdict(set)
    for r in result:
        by_crop[r['crop_sha256']].add(r['group'])
    split_of = {r['group']: r['split'] for r in result}
    quarantine = {g for groups in by_crop.values() if len({split_of[g] for g in groups}) > 1 for g in groups}
    for r in result:
        if r['group'] in quarantine:
            r['split'] = 'excluded_duplicate'
    counts = Counter((r['split'], r['class_name']) for r in result)
    assert all(counts[(s, c)] > 0 for s in ['train', 'dev'] for c in NAMES), 'split lacks a class'
    a.out.mkdir(parents=True, exist_ok=False)
    manifest = a.out / 'crops.jsonl'
    manifest.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in result))
    metadata = dict(version=1, seed=a.seed, source_limit=a.max_sources, annotations=str(a.annotations),
                    annotations_sha256=digest(raw), manifest_sha256=digest(manifest.read_bytes()),
                    source_groups=len({r['group'] for r in result}),
                    counts={f'{s}/{c}': n for (s,c),n in sorted(counts.items())},
                    limitations=['dHash grouping is an automatic near-duplicate heuristic, not verified camera identity; residual scene leakage remains',
                                 'Existing exporter applied first-in-order class caps and a 48px minimum; this is a biased screening pool, not deployment evaluation',
                                 'source hash sampling happens before clustering when max-sources > 0'])
    (a.out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()

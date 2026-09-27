#!/usr/bin/env python3
"""Run unchanged-checkpoint ST/MV baselines and retain reproducibility metadata."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('/data/datasets/iisc-aim'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0, help='0 means complete validation sets')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--cuda-python-root', type=Path, help='NVIDIA Python package directory containing cuda_nvrtc, cublas and cuda_runtime')
    args = parser.parse_args()
    if args.limit < 0 or args.batch < 1:
        parser.error('limit must be nonnegative and batch positive')
    repo = Path(__file__).resolve().parents[1]
    binary = repo / 'target/release/examples/error_report'
    checkpoint = args.data / 'models-UVH-26/weights/RT-DETRv2-S/UVH-26-MV-RT-DETRv2-S.pth'
    validation = args.data / 'UVH-26/UVH-26-Val'
    image_root = validation / 'data'
    env = dict(os.environ)
    env['OJAS_LEARN_PREC'] = 'bf16'
    if args.cuda_python_root:
        cuda_root = args.cuda_python_root.resolve()
        env['LD_LIBRARY_PATH'] = ':'.join([str(cuda_root / 'cuda_nvrtc/lib'), str(cuda_root / 'cublas/lib'), env.get('LD_LIBRARY_PATH', '')])
        env['OJAS_CUDA_INCLUDE'] = str(cuda_root / 'cuda_runtime/include')
    args.out.mkdir(parents=True, exist_ok=False)
    # The Rust loader falls back to basename lookup. Refuse ambiguous basenames.
    files = {}
    for path in sorted(image_root.rglob('*')):
        if path.is_file():
            if path.name in files:
                raise ValueError(f'ambiguous image basename: {path.name}')
            files[path.name] = path
    manifest = {
        'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint),
        'binary_sha256': sha256(binary), 'prefix': 'ema.module.',
        'batch': args.batch, 'resize': [640, 640], 'score_threshold': 0.3,
        'backend': 'CUDA', 'dtype': 'bf16 GEMMs with f32 accumulation; backend f32 buffers',
        'runtime_environment': {k: v for k, v in env.items() if k.startswith('OJAS_') or k in ['LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES']},
        'postprocessing': 'AP: top 300 query x class pairs; subtype: one best class per query, IoU >= 0.5, score >= 0.3',
        'colour': 'not measured: no labelled colour evaluation set',
        'image_identity': 'path, size and mtime only; image content hashes NOT computed',
        'splits': {},
    }
    patch = subprocess.check_output(['git', 'diff', 'HEAD', '--', 'crates/ojas-learn'], cwd=repo)
    (args.out / 'evaluation.patch').write_bytes(patch)
    manifest['evaluation_patch_sha256'] = hashlib.sha256(patch).hexdigest()
    manifest['runner_sha256'] = sha256(Path(__file__))
    manifest['gpu'] = subprocess.check_output(['nvidia-smi'], text=True)
    for split in ['ST', 'MV']:
        annotation = validation / f'UVH-26-{split}-Val.json'
        data = json.loads(annotation.read_text())
        selected = data['images'][:args.limit or None]
        inventory = []
        for image in selected:
            path = files.get(Path(image['file_name']).name)
            if path is None:
                raise FileNotFoundError(image['file_name'])
            stat = path.stat()
            inventory.append({'id': image['id'], 'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
        inventory_path = args.out / f'{split.lower()}_images.json'
        inventory_path.write_text(json.dumps(inventory, indent=2) + '\n')
        command = [str(binary), str(checkpoint), 'ema.module.', str(annotation), str(image_root), str(len(selected)), str(args.batch), '0.3']
        manifest['splits'][split] = {
            'annotations': str(annotation), 'annotations_sha256': sha256(annotation),
            'images': len(selected), 'inventory_sha256': sha256(inventory_path),
            'command': command, 'status': 'pending',
        }
    manifest_path = args.out / 'manifest.json'
    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    save()
    for split, entry in manifest['splits'].items():
        print(f'Starting {split}: {entry["images"]} images', flush=True)
        entry['status'] = 'running'
        save()
        start = time.monotonic()
        with (args.out / f'{split.lower()}_report.txt').open('w') as report:
            result = subprocess.run(entry['command'], cwd=repo, env={**env, 'CLASS_NAMES': entry['annotations']}, stdout=report, stderr=subprocess.STDOUT)
        entry.update(status='complete' if result.returncode == 0 else 'failed', returncode=result.returncode, elapsed_seconds=time.monotonic() - start)
        save()
        print(f'{split}: {entry["status"]} in {entry["elapsed_seconds"]:.1f}s', flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)
    manifest['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()


if __name__ == '__main__':
    main()

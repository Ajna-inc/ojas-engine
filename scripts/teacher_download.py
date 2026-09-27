#!/usr/bin/env python3
"""Download the official, revision-pinned SigLIP2 teacher to the large data drive."""
import json
import os
from pathlib import Path

repo = 'google/siglip2-so400m-patch14-384'
cache = Path('/data/dev-cache/train/teacher-cache')
os.environ['HF_XET_CACHE'] = str(cache / 'xet')
from huggingface_hub import snapshot_download
revision = 'e8e487298228002f3d8a82e0cd5c8ea9c567f57f'
print(f'Downloading {repo}@{revision}', flush=True)
path = snapshot_download(repo_id=repo, revision=revision, cache_dir=str(cache),
                         allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt'], max_workers=2)
out = Path('evidence/teacher_benchmark_2026_09_20')
out.mkdir(parents=True, exist_ok=True)
(out / 'siglip_download.json').write_text(json.dumps({'repo': repo, 'revision': revision,
    'path': path, 'remote_code': False}, indent=2) + '\n')
print(path, flush=True)

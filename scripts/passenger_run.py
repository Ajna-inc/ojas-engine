#!/usr/bin/env python3
"""Run a recorded crop-training configuration using the local CUDA libraries."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

repo = Path(__file__).resolve().parents[1]
config_path = Path(sys.argv[1]).resolve()
config = json.loads(config_path.read_text())
binary_name = config.get('binary', 'passenger_train')
assert binary_name in ['passenger_train', 'passenger_residual_train']
binary = repo / 'target/release/examples' / binary_name
out = Path(config['out'])
if out.exists():
    raise SystemExit(f'Output already exists: {out}')
cuda = Path('/usr/local/lib/python3/site-packages/nvidia')
env = dict(os.environ, LD_LIBRARY_PATH=f'{cuda}/cuda_nvrtc/lib:{cuda}/cublas/lib', OJAS_CUDA_INCLUDE=str(cuda / 'cuda_runtime/include'), OJAS_LEARN_PREC='bf16')
metadata = {'config_sha256': sha(config_path), 'manifest_sha256': sha(config['manifest']), 'init_sha256': sha(config['init']),
            'binary_sha256': sha(binary), 'binary': str(binary),
            'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
            'runtime_environment': {k:v for k,v in env.items() if k.startswith('OJAS_') or k=='LD_LIBRARY_PATH'},
            'checkpoint_semantics':'model weights only; resume unsupported; safetensors initialization resets optimizer and RNG',
            'model':f'PResNet-18-vd through stage {config.get("backbone_stage", 2)}, global average pooling, 4-way linear head, frozen BN statistics',
            'preprocessing':'RGB/255, black letterbox 224, train horizontal flip and brightness 0.85..1.15'}
if binary_name == 'passenger_residual_train':
    metadata['model'] += '; head zero-initialized; outputs added to cached original passenger logits; zero-correction checkpoint eligible for development selection'
for line in Path(config['manifest']).read_text().splitlines():
    row = json.loads(line)
    if row['split'] in ['train','dev'] and sha(row['path']) != row['crop_sha256']:
        raise SystemExit(f'Crop changed: {row["path"]}')
log_path = config_path.with_suffix('.log')
with log_path.open('x') as log:
    process = subprocess.Popen([str(binary), str(config_path)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=repo)
    for line in process.stdout:
        log.write(line)
        log.flush()
        print(line, end='', flush=True)
    code = process.wait()
metadata['exit_code'] = code
if out.exists():
    for name in ['best.safetensors','final.safetensors','best.json','metrics.jsonl']:
        if (out/name).exists(): metadata[f'{name}_sha256'] = sha(out/name)
    (out/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
sys.exit(code)

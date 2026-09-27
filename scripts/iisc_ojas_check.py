#!/usr/bin/env python3
"""Verify that an unchanged reserialized IISc checkpoint keeps Ojas metrics identical."""
import json,os,subprocess
from pathlib import Path
from teacher_detector_bench import sha
root=Path('evidence/iisc_small_preflight_2026_09_20')
original=Path('/data/datasets/iisc-aim/models-UVH-26/weights/RT-DETRv2-S/UVH-26-MV-RT-DETRv2-S.pth')
zero=root/'subtype_gate/zero_update.pth'
base=Path('/data/datasets/iisc-aim/UVH-26/UVH-26-Val')
ann=json.loads((base/'UVH-26-ST-Val.json').read_text())
ids=set(json.loads(Path('evidence/passenger_sam_2026_09_20/evaluation_contract.json').read_text())['image_ids'][:8])
ann['images']=[r for r in ann['images'] if r['id'] in ids];ann['annotations']=[r for r in ann['annotations'] if r['image_id'] in ids]
p=root/'ojas_eight_images.json';p.write_text(json.dumps(ann))
cuda=Path('/usr/local/lib/python3/site-packages/nvidia')
env=dict(os.environ,LD_LIBRARY_PATH=f'{cuda}/cuda_nvrtc/lib:{cuda}/cublas/lib',OJAS_CUDA_INCLUDE=str(cuda/'cuda_runtime/include'),OJAS_LEARN_PREC='bf16')
texts={}
for name,checkpoint in [('original',original),('zero',zero)]:
 result=subprocess.run(['target/release/examples/error_report',str(checkpoint),'ema.module.',str(p),str(base/'data'),'8','1','.3'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=True)
 (root/f'ojas_{name}.log').write_text(result.stdout+'\nSTDERR\n'+result.stderr)
 # Runtime progress goes to stderr; strip time-dependent stdout lines from metrics comparison.
 texts[name]='\n'.join(line for line in result.stdout.splitlines() if not any(word in line.lower() for word in ['elapsed','seconds','time:','evaluated in']))
# Confusion rows with equal counts are emitted from a Rust HashMap in arbitrary order.
from collections import Counter
assert Counter(texts['original'].splitlines())==Counter(texts['zero'].splitlines()),'Ojas metrics differ; inspect logs'
(root/'ojas_parity.json').write_text(json.dumps({'eight_image_metrics_exact':True,'original_sha256':sha(original),'zero_sha256':sha(zero),'binary_sha256':sha('target/release/examples/error_report'),'scope':'Checkpoint packaging/import parity within Ojas BF16. Not FP32 PyTorch versus BF16 numerical parity.'},indent=2)+'\n')
print('Ojas zero-update metric parity passed on eight images.')

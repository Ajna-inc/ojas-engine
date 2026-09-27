#!/usr/bin/env python3
"""Evaluate the development-selected recipe on a predeclared fresh ST subset."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import argparse

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('root',type=Path)
parser.add_argument('--arm',action='append',help='Evaluate only an already declared arm; repeatable')
args=parser.parse_args()
root=args.root.resolve()
repo=Path(__file__).resolve().parents[1]
selection=json.loads((root/'comparison.json').read_text())['selected_sampler']
contract=json.loads((root/'evaluation_contract.json').read_text())
cuda=Path('/usr/local/lib/python3/site-packages/nvidia')
env=dict(os.environ,LD_LIBRARY_PATH=f'{cuda}/cuda_nvrtc/lib:{cuda}/cublas/lib',OJAS_CUDA_INCLUDE=str(cuda/'cuda_runtime/include'),OJAS_LEARN_PREC='bf16')
base=Path('/data/datasets/iisc-aim')
detector=base/'models-UVH-26/weights/RT-DETRv2-S/UVH-26-MV-RT-DETRv2-S.pth'
annotations=base/'UVH-26/UVH-26-Val/UVH-26-ST-Val.json'
binary=repo/'target/release/examples/passenger_eval'
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
arms=contract.get('arms', [
    {'name':f'{selection}_seed{seed}', 'expert':str(root/f'{selection}_seed{seed}'/'best.safetensors')}
    for seed in [1,2]
])
if args.arm:
    assert set(args.arm) <= {a['name'] for a in arms}, 'arm not in frozen contract'
    arms=[a for a in arms if a['name'] in args.arm]
for arm in arms:
    binary_name=arm.get('binary','passenger_eval')
    assert binary_name in ['passenger_eval','passenger_residual_eval']
    binary=repo/'target/release/examples'/binary_name
    name=arm['name']
    assert Path(name).name == name, 'arm name must be a filename component'
    expert=Path(arm['expert']).resolve()
    output=root/f'{name}_fresh_boxes.json'
    assert not output.exists(), 'do not overwrite evaluation'
    command=[str(binary),str(detector),str(expert),str(annotations),str(annotations.parent/'data'),str(output),str(contract['images']),str(contract['skip'])]
    print(f'Evaluating {name}',flush=True)
    with (root/f'{name}_fresh_boxes.log').open('x') as log:
        result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit(f'{name} failed: {result.returncode}; see log')
    report=json.loads(output.read_text())
    assert [r['id'] for r in report['per_image']]==contract['image_ids']
    metadata={'command':command,'expert_sha256':sha(expert),'detector_sha256':sha(detector),'annotations_sha256':sha(annotations),'binary_sha256':sha(binary),'report_sha256':sha(output),'comparison_sha256':sha(root/'comparison.json'),'evaluation_contract_sha256':sha(root/'evaluation_contract.json')}
    (root/f'{name}_fresh_boxes_provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['per_image','proposals','before_confusion','after_confusion']}),flush=True)

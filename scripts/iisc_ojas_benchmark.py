#!/usr/bin/env python3
"""Evaluate an IISc-format checkpoint through Ojas on the frozen full-scene cohort."""
import argparse,json,os,re,subprocess,time
from pathlib import Path
from teacher_detector_bench import sha
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--name',required=True);p.add_argument('--checkpoint',type=Path);a=p.parse_args()
assert Path(a.name).name==a.name
root=a.root.resolve();out=root/'ojas';out.mkdir(exist_ok=True);report=out/f'{a.name}.json';assert not report.exists()
c=json.loads((root/'contract.json').read_text());original=json.loads((root/'baseline.json').read_text())['metadata']['checkpoint'];checkpoint=a.checkpoint or Path(original)
ann=json.loads(Path(c['annotations']).read_text());ids={r['id'] for r in c['images']};ann['images']=[r for r in ann['images'] if r['id'] in ids];ann['annotations']=[r for r in ann['annotations'] if r['image_id'] in ids]
annotation=out/'annotations.json';data=json.dumps(ann)
if annotation.exists():assert annotation.read_text()==data
else:annotation.write_text(data)
cuda=Path('/usr/local/lib/python3/site-packages/nvidia');env=dict(os.environ,LD_LIBRARY_PATH=f'{cuda}/cuda_nvrtc/lib:{cuda}/cublas/lib',OJAS_CUDA_INCLUDE=str(cuda/'cuda_runtime/include'),OJAS_LEARN_PREC='bf16')
command=['target/release/examples/error_report',str(checkpoint),'ema.module.',str(annotation),str(Path(c['annotations']).parent/'data'),str(len(ids)),'8','.3']
start=time.perf_counter();run=subprocess.run(command,env=env,capture_output=True,text=True)
(out/f'{a.name}.log').write_text(run.stdout+'\nSTDERR\n'+run.stderr);assert run.returncode==0,run.stderr
m=re.search(r'GT (\d+); localized (\d+); correct subtype (\d+); missed (\d+)',run.stdout);assert m
n,localized,correct,missed=map(int,m.groups());ap=re.search(r'mAP@\[.5:.95\] ([0-9.]+)  AP50 ([0-9.]+)',run.stdout);assert ap
result={'checkpoint_sha256':sha(checkpoint),'binary_sha256':sha(command[0]),'contract_sha256':sha(root/'contract.json'),'annotation_sha256':sha(annotation),'backend':'Ojas CUDA BF16','images':len(ids),'passenger_gt':n,'localized':localized,'correct':correct,'missed':missed,'conditional_accuracy':correct/max(1,localized),'correct_fraction_all_gt':correct/max(1,n),'AP_rounded':float(ap[1]),'AP50_rounded':float(ap[2]),'wall_seconds_including_loading_evaluation':time.perf_counter()-start,'scope':'Whole-scene detector metrics. Not the fixed routed-crop cohort and not an inference latency benchmark.'}
report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

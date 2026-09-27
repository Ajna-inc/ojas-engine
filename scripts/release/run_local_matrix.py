#!/usr/bin/env python3
"""Run explicit, sequential reference comparisons and retain failures as evidence."""
import argparse, hashlib, json, os, platform, subprocess, sys, time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--manifest',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
p.add_argument('--ojas',type=Path,required=True);p.add_argument('--work',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.work.mkdir(parents=True,exist_ok=True);a.output.mkdir(parents=True,exist_ok=True)
compare=Path(__file__).with_name('compare_logits.py');rows=[]
def run(cmd,log):
    start=time.monotonic()
    with log.open('w') as f:
        try:r=subprocess.run([str(x) for x in cmd],stdout=f,stderr=subprocess.STDOUT,timeout=600)
        except subprocess.TimeoutExpired:return {'exit_code':124,'elapsed_s':time.monotonic()-start}
    return {'exit_code':r.returncode,'elapsed_s':time.monotonic()-start}
def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
items=json.loads(a.manifest.read_text())
if not items:raise SystemExit("empty release manifest")
for item in items:
    label=item['label'];model=Path(item['model']);prompt=a.work/(label+'.prompt.txt')
    prompt.write_text(item['prompt']);ref=a.work/(label+'-ref')
    record={'label':label,'model_filename':model.name,'model_sha256':sha(model),'precision':item.get('precision',4),'prompt':item['prompt'],'reference_backend':'Metal layers, CPU KV operations','runs':{}}
    status=run([a.reference,model,prompt,ref,'gpu'],a.work/(label+'-reference.log'));record['reference']=status
    if status['exit_code']==0:
        record['reference_ids']=[int(x) for x in Path(str(ref)+'.ids').read_text().split()]
        for mode in ['decode','prefill']:
            output=a.work/(label+'-'+mode+'.f32')
            status=run([a.ojas,model,str(ref)+'.ids',output,record['precision'],mode,prompt],a.work/(label+'-'+mode+'.log'))
            if status['exit_code']==0:
                result=a.output/(label+'-'+mode+'.json')
                checked=run([sys.executable,compare,str(ref)+'.f32',output,'--output',result],a.work/(label+'-'+mode+'-compare.log'))
                status['comparison_exit_code']=checked['exit_code'];status['result']=result.name
            record['runs'][mode]=status
    rows.append(record)
    (a.output/'matrix.json').write_text(json.dumps({'platform':platform.platform(),'reference_binary_sha256':sha(a.reference),'ojas_binary_sha256':sha(a.ojas),'engine_git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'engine_diff_sha256':hashlib.sha256(subprocess.check_output(['git','diff','HEAD'])).hexdigest(),'cases':rows},indent=2)+'\n')
    print(label,record['runs'],flush=True)
failed=any(r['reference']['exit_code'] or len(r['runs'])!=2 or any(s['exit_code'] or s.get('comparison_exit_code',1) for s in r['runs'].values()) for r in rows)
sys.exit(1 if failed else 0)

#!/usr/bin/env python3
"""Run explicit MTP/no-MTP fixtures; retain failed gates and require engagement."""
import argparse, hashlib, json, os, re, subprocess, time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--manifest',type=Path,required=True)
p.add_argument('--binary',type=Path,required=True)
p.add_argument('--work',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.work.mkdir(parents=True,exist_ok=True);a.output.parent.mkdir(parents=True,exist_ok=True)
items=json.loads(a.manifest.read_text())
if not items:raise SystemExit('empty MTP manifest')
hashes={}
def sha(path):
 path=Path(path)
 if str(path) not in hashes:
  with path.open('rb') as f:hashes[str(path)]=hashlib.file_digest(f,'sha256').hexdigest()
 return hashes[str(path)]
report={'binary_sha256':sha(a.binary),'rust':subprocess.check_output(['rustc','--version'],text=True).strip(),'cases':[],'timing_note':'Correctness workload wall times, not benchmarks.'}
for item in items:
 env={k:v for k,v in os.environ.items() if not k.startswith('OJAS_')}
 env.update(item.get('env',{}))
 model=Path(item['model']);log=a.work/(item['label']+'.log')
 row={'label':item['label'],'model':model.name,'model_sha256':sha(model),'precision':item.get('precision',4),'environment':{k:Path(v).name if k=='OJAS_MTP' else v for k,v in item.get('env',{}).items()},'expect_mtp':item['expect_mtp']}
 if 'OJAS_MTP' in env:row['sidecar_sha256']=sha(env['OJAS_MTP'])
 cmd=[str(a.binary),str(model),str(row['precision']),str(item.get('tokens',32))]
 start=time.monotonic()
 with log.open('w') as f:
  try:code=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1200).returncode
  except subprocess.TimeoutExpired:code=124
 text=log.read_text();found=re.search(r'architecture=\S+ MTP=(true|false)',text)
 row.update(exit_code=code,elapsed_s=time.monotonic()-start,mtp_present=(found.group(1)=='true') if found else None,result_lines=[s for s in text.splitlines() if s.startswith(('architecture=','prompt ','forced case ','PASS:','Error:'))],log_sha256=sha(log))
 row['passed']=code==0 and 'PASS:' in text and row['mtp_present']==row['expect_mtp']
 report['cases'].append(row);a.output.write_text(json.dumps(report,indent=2)+'\n')
 print(row['label'],'PASS' if row['passed'] else 'FAIL',flush=True)
raise SystemExit(0 if all(r['passed'] for r in report['cases']) else 1)

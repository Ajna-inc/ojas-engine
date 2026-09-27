#!/usr/bin/env python3
from pathlib import Path
import os,subprocess,json,hashlib,statistics,argparse,platform
parser=argparse.ArgumentParser(description='Fixed-reference Flash oracle and forced-draft diagnostics.')
parser.add_argument('--model',required=True)
parser.add_argument('--reference-dir',type=Path,required=True,help='Directory containing france.ids, code.ids, repeat.ids')
parser.add_argument('--output-dir',type=Path,required=True)
parser.add_argument('--cache-gb',type=int,default=8)
parser.add_argument('--modes',nargs='+',choices=['oracle','maintained','draft'],default=['oracle','maintained','draft'])
parser.add_argument('--binary',type=Path,default=Path('target/release/examples/flash_mtp_cost'))
a=parser.parse_args()
if a.cache_gb <= 0:parser.error("cache-gb must be positive")
p=a.output_dir;p.mkdir(parents=True,exist_ok=True)
b=a.binary.resolve();model=a.model
def sha(f):
 with Path(f).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
r={'binary_sha256':sha(b),'source_sha256':{f:sha(f) for f in sorted(str(f) for f in Path('crates').rglob('*.rs'))},'model_entrypoint_sha256':sha(model),'model_hash_scope':'Entrypoint only; retain a separate complete shard/sidecar manifest.','hardware':{'platform':platform.platform(),'cpu':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),'memory_bytes':int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True))},'precision':4,'repetitions':3,'warmups_per_configuration':1,'timing_scope':'Generation of 32 reference tokens; prefill excluded. Oracle skips drafting and draft-cache maintenance but retains normal verify snapshots/head. Maintained mode uses perfect drafts while retaining draft-cache catch-up. Real draft mode forces every step to draft, bypassing adaptive scheduling. Width order rotates between repetitions. Gather reads count lookup requests, including hits. Not an uncontended benchmark or fundamental hardware upper bound.','cases':[]}
for label in ['france','code','repeat']:
 for mode in a.modes:
  if sha(b)!=r['binary_sha256']:raise SystemExit("diagnostic binary changed during run")
  env={k:v for k,v in os.environ.items() if not k.startswith('OJAS_')};env.update(OJAS_EXPERT_CACHE_GB=str(a.cache_gb),OJAS_MTP_DRAFT='3')
  if mode=='oracle':env['OJAS_NO_SPEC']='1'
  ids=a.reference_dir/f'{label}.ids';log=p/f'{label}-{mode}.log'
  memory_before=subprocess.check_output(['vm_stat'],text=True)
  with log.open('w') as f:
   try:code=subprocess.run([str(b),model,str(ids),mode,'3'],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1200).returncode
   except subprocess.TimeoutExpired:code=124
  records=[json.loads(s) for s in log.read_text().splitlines() if s.startswith('{')]
  row={'label':label,'mode':mode,'exit_code':code,'environment':{k:v for k,v in env.items() if k.startswith('OJAS_')},'reference_ids_sha256':sha(ids),'records':records,'log_sha256':sha(log),'memory_before':memory_before,'memory_after':subprocess.check_output(['vm_stat'],text=True),'passed':code==0 and len(records)==(12 if mode=='oracle' else 9)}
  row['summary']={}
  for width in sorted({x['width'] for x in records}):
   samples=[x for x in records if x['width']==width]
   row['summary'][str(width)]={k:{'median':statistics.median(x[k] for x in samples),'min':min(x[k] for x in samples),'max':max(x[k] for x in samples)} for k in ['seconds','tps','draft_s','verify_s','rollback_s','target_s','catchup_s','gather_reads','gather_hits']}
  r['cases'].append(row);(p/'results.json').write_text(json.dumps(r,indent=2)+'\n');print(label,mode,row['passed'],flush=True)
raise SystemExit(0 if all(x['passed'] for x in r['cases']) else 1)

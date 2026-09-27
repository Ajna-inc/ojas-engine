#!/usr/bin/env python3
"""Compare fixed-input logits. Thresholds are explicit, never inferred from results."""
import argparse, array, hashlib, json, math, sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('reference',type=Path);p.add_argument('actual',type=Path);p.add_argument('--rows',type=int,default=9)
p.add_argument('--max-rmse',type=float,default=0.05);p.add_argument('--min-cosine',type=float,default=0.999)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
def load(path):
    v=array.array('f');v.frombytes(path.read_bytes())
    if sys.byteorder!='little':v.byteswap()
    if not all(math.isfinite(x) for x in v):raise SystemExit(f'nonfinite logits: {path}')
    return v
x,y=load(a.reference),load(a.actual)
if a.rows<1 or len(x)!=len(y) or not len(x) or len(x)%a.rows:raise SystemExit('incomplete or mismatched logits')
v=len(x)//a.rows;rows=[]
for i in range(a.rows):
    r,s=x[i*v:(i+1)*v],y[i*v:(i+1)*v]
    rmse=math.sqrt(sum((u-w)**2 for u,w in zip(r,s))/v)
    cosine=sum(u*w for u,w in zip(r,s))/math.sqrt(sum(u*u for u in r)*sum(w*w for w in s))
    ri=max(range(v),key=r.__getitem__);si=max(range(v),key=s.__getitem__)
    rows.append(dict(row=i,rmse=rmse,cosine=cosine,max_abs=max(abs(u-w) for u,w in zip(r,s)),reference_argmax=ri,actual_argmax=si))
passed=all(r['rmse']<=a.max_rmse and r['cosine']>=a.min_cosine and r['reference_argmax']==r['actual_argmax'] for r in rows)
result=dict(passed=passed,vocab=v,thresholds=dict(max_rmse=a.max_rmse,min_cosine=a.min_cosine,require_all_argmax_equal=True),reference_sha256=hashlib.sha256(a.reference.read_bytes()).hexdigest(),actual_sha256=hashlib.sha256(a.actual.read_bytes()).hexdigest(),rows=rows)
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(passed=passed,max_rmse=max(r['rmse'] for r in rows),min_cosine=min(r['cosine'] for r in rows),argmax_matches=sum(r['reference_argmax']==r['actual_argmax'] for r in rows),rows=a.rows)))
sys.exit(0 if passed else 1)

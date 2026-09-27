#!/usr/bin/env python3
"""Time actual classifier weights; run only after other GPU work has stopped."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

def main():
    checkpoint,manifest,output=map(lambda x:Path(x).resolve(),sys.argv[1:])
    assert not output.exists()
    binary=Path('target/release/examples/passenger_crop_bench').resolve()
    cuda=Path('/usr/local/lib/python3/site-packages/nvidia')
    env=dict(os.environ,LD_LIBRARY_PATH=f'{cuda}/cuda_nvrtc/lib:{cuda}/cublas/lib',
             OJAS_CUDA_INCLUDE=str(cuda/'cuda_runtime/include'),OJAS_LEARN_PREC='bf16')
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
    assert not processes, f'Other GPU compute processes present: {processes}'
    command=[str(binary),str(checkpoint),str(manifest)]
    run=subprocess.run(command,env=env,capture_output=True,text=True)
    output.with_suffix('.log').write_text(run.stdout+run.stderr)
    assert run.returncode==0, run.stderr[-2000:]
    rows=[json.loads(line) for line in run.stdout.splitlines() if line.startswith('{')]
    assert [r['batch'] for r in rows]==[1,4,16]
    result=dict(command=command,results=rows,
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [checkpoint,manifest,binary]},
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip())
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()

#!/usr/bin/env python3
"""Paired Flash GPU projection benchmark, independent token checks, one model load."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--reference', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--reps', type=int, default=6)
    p.add_argument('--widths', default='2,3,4')
    p.add_argument('--cache-gb', type=int, default=16)
    a = p.parse_args()
    widths = [int(w) for w in a.widths.split(',')]
    if not (3 <= a.reps <= 20 and widths and all(1 <= w <= 16 for w in widths)
            and len(set(widths)) == len(widths) and a.cache_gb > 0):
        p.error('reps 3..20; distinct widths 1..16; positive cache')
    a.output_dir.mkdir(parents=True, exist_ok=False)
    binary = Path('target/release/examples/flash_gpu_bench').resolve()
    env = {k: v for k, v in os.environ.items() if not k.startswith('OJAS_')}
    settings = dict(OJAS_NO_SPEC='1', OJAS_EXPERT_CACHE_GB=str(a.cache_gb), OJAS_EXPERT_COPY_THREADS='4')
    env.update(settings)
    report = dict(settings=settings, binary_sha256=sha(binary), reference_sha256=sha(a.reference),
                  source_sha256={str(p): sha(p) for p in sorted(Path('crates').rglob('*.rs'))},
                  model=a.model, arguments=vars(a).copy(),
                  cpu=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
                  memory_bytes=int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True)),
                  memory_before=subprocess.check_output(['vm_stat'],text=True),
                  scope='32 oracle target outputs, prefill excluded; no drafting/head work; rotated width/kernel requests, warmup per case; GPU time is inside wait time.')
    report['arguments'] = {k:str(v) if isinstance(v,Path) else v for k,v in report['arguments'].items()}
    dest = a.output_dir/'results.json'
    dest.write_text(json.dumps(report, indent=2)+'\n')
    log = a.output_dir/'raw.log'
    with log.open('w') as f:
        try:
            code = subprocess.run([str(binary), a.model, str(a.reference),str(a.reps),a.widths],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1200).returncode
        except subprocess.TimeoutExpired:
            code=124
    records = [json.loads(l) for l in log.read_text().splitlines() if l.startswith('{')]
    passed = code == 0 and sha(binary)==report['binary_sha256'] and len(records)==a.reps*len(widths)*2 and all(
        len([r for r in records if r['width']==w and r['cooperative']==v])==a.reps for w in widths for v in [False,True])
    report.update(exit_code=code,passed=passed,records=records,log_sha256=sha(log),
                  memory_after=subprocess.check_output(['vm_stat'],text=True),summary={})
    for w in widths:
        for v in [False,True]:
            rows=[r for r in records if r['width']==w and r['cooperative']==v]
            if rows:
                report['summary'][f'width={w},cooperative={v}']={k:dict(median=statistics.median(r[k] for r in rows),min=min(r[k] for r in rows),max=max(r[k] for r in rows)) for k in ['tps','target_s','gpu_s','gather_copy_s','encode_s']}
    dest.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=passed,summary=report['summary'])))
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()

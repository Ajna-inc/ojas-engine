#!/usr/bin/env python3
"""CPU convergence diagnostic: L2-regularized softmax on frozen DINO features."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from passenger_feature_probe import metrics

root = Path('evidence/passenger_dinov2_2026_09_20').resolve()
out = root / 'convergence'
out.mkdir()  # Require a new experiment; no silent replacement.
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
contract = dict(features_sha256=sha(root/'features.npz'), parent_contract_sha256=sha(root/'contract.json'),
    method='L-BFGS convex multiclass softmax; average CE + lambda/2 * squared weights, unpenalized bias',
    lambdas=[1e-3, 1e-4, 1e-5], maxiter=2000, selection='Highest development macro-F1; tie chooses stronger regularization',
    scope='Follow-up optimization diagnostic after weak fixed-budget AdamW head. Benchmark already consumed, not independent confirmation. Encoder unchanged.',
    script_sha256=sha(__file__))
(out/'contract.json').write_text(json.dumps(contract, indent=2)+'\n')
(out/'source.py').write_bytes(Path(__file__).read_bytes())
parent = json.loads((root/'contract.json').read_text())
manifest = Path('evidence/passenger_screen_2026_09_20/crops.jsonl')
assert sha(manifest) == parent['manifest_sha256']
rows = [json.loads(x) for x in manifest.read_text().splitlines()]
rows = [r for r in rows if r['split'] in ['train','dev']]
f = np.load(root/'features.npz')
train = np.array([r['split']=='train' for r in rows])
labels = np.array([r['label'] for r in rows])
x = np.ascontiguousarray(f['source'][train], dtype=np.float64)
y = labels[train]
# Training-only centering changes conditioning, not the class of linear functions.
mean = x.mean(0); x -= mean
dev = np.ascontiguousarray(f['source'][~train], dtype=np.float64)-mean
d = x.shape[1]
targets = np.eye(4)[y]
runs = []
start = time.perf_counter()
for penalty in contract['lambdas']:
    def objective(v):
        w, b = v[:d*4].reshape(d,4), v[d*4:]
        z = x @ w + b
        loss = (logsumexp(z, axis=1)-z[np.arange(len(y)), y]).mean() + penalty/2*np.square(w).sum()
        residual = (softmax(z, axis=1)-targets)/len(y)
        grad = np.concatenate([(x.T @ residual + penalty*w).ravel(), residual.sum(0)])
        return loss, grad
    fit = minimize(objective, np.zeros(d*4+4), jac=True, method='L-BFGS-B',
        options=dict(maxiter=contract['maxiter'], gtol=1e-7, ftol=1e-12, maxcor=20))
    assert fit.success, f'Not converged: {fit.message}'
    w, b = fit.x[:d*4].reshape(d,4), fit.x[d*4:]
    report = dict(penalty=penalty, iterations=fit.nit, objective=float(fit.fun),
        max_abs_gradient=float(np.abs(fit.jac).max()), status=str(fit.message),
        train=metrics(softmax(x@w+b, axis=1), y),
        dev=metrics(softmax(dev@w+b, axis=1), labels[~train]))
    np.savez(out/f'head_{penalty}.npz', weight=w, bias=b, feature_mean=mean)
    runs.append(report)
    print('fit',penalty,'iterations',fit.nit,'train',report['train']['accuracy'],'dev',report['dev']['accuracy'],flush=True)
selected = max(runs, key=lambda r:r['dev']['macro_f1'])
(out/'selection.json').write_text(json.dumps(dict(runs=runs, selected_penalty=selected['penalty'], seconds=time.perf_counter()-start), indent=2)+'\n')
# Only the development-selected head is evaluated on the consumed benchmark.
head = np.load(out/f"head_{selected['penalty']}.npz")
proposals_path = Path('evidence/teacher_benchmark_2026_09_20/baseline_crops.json')
assert sha(proposals_path) == parent['proposals_sha256']
rows = json.loads(proposals_path.read_text())
truth = np.array([(r['truth'] or 0)-1 for r in rows]); eligible = (truth>=0)&(truth<4)
probs = softmax((f['test'].astype(np.float64)-head['feature_mean']) @ head['weight'] + head['bias'], axis=1)
np.savez_compressed(out/'predictions.npz', probabilities=probs)
pred = probs.argmax(1); detector = np.array([r['class']-1 for r in rows])
counts = {}
for i,r in enumerate(rows):
    counts.setdefault(r['image_id'], np.zeros(2))
    if eligible[i]: counts[r['image_id']] += [1, int(pred[i]==truth[i])-int(detector[i]==truth[i])]
counts = np.array(list(counts.values())); rng = np.random.default_rng(20260920)
samples = np.array([counts[rng.integers(len(counts),size=len(counts))].sum(0) for _ in range(10000)])
result = dict(selected_penalty=selected['penalty'], actual_boxes=metrics(probs[eligible],truth[eligible]),
    detector_accuracy=float((detector[eligible]==truth[eligible]).mean()),
    paired_delta=float(counts[:,1].sum()/counts[:,0].sum()),
    image_bootstrap95=np.percentile(samples[:,1]/samples[:,0], [2.5,97.5]).tolist(),
    scope=contract['scope'], uncertainty='Image bootstrap, not camera-disjoint; does not account for prior benchmark use.',
    head_sha256=sha(out/f"head_{selected['penalty']}.npz"))
(out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)

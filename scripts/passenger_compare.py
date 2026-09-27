#!/usr/bin/env python3
"""Compare the two-seed, equal-update crop experiment on its frozen development set."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(sys.argv[1])
old = Path('evidence/passenger_screen_2026_09_20')
records = []
predictions = {}
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for sampler in ['ordinary', 'balanced']:
    for seed in [1, 2]:
        name = f'{sampler}_seed{seed}'
        run = root / name
        provenance = json.loads((run/'provenance.json').read_text())
        assert provenance['exit_code'] == 0
        history = [json.loads(l) for l in (run/'metrics.jsonl').read_text().splitlines()]
        assert history[-1]['step'] == 1984
        best = json.loads((run/'best.json').read_text())
        preds = [json.loads(l) for l in (run/f'epoch{best["epoch"]}_predictions.jsonl').read_text().splitlines()]
        predictions[name] = preds
        record = dict(name=name, sampler=sampler, seed=seed, epoch=best['epoch'], **best['metrics'],
                      seconds=history[-1]['elapsed_seconds'], manifest_sha256=provenance['manifest_sha256'])
        records.append(record)
        for ax, key, label in zip(axes, ['accuracy','macro_f1'], ['Development accuracy','Development macro-F1']):
            ax.plot([h['epoch'] for h in history], [h['metrics'][key] for h in history],
                    label=f'{sampler}, seed {seed}', linestyle='-' if seed == 1 else '--',
                    color='tab:blue' if sampler == 'ordinary' else 'tab:orange')
            ax.set(xlabel='Epoch', ylabel=label, ylim=(0,1))
            ax.axvline(3,color='gray',alpha=.5,linewidth=1)
            ax.grid(alpha=.2)
assert len({r['manifest_sha256'] for r in records}) == 1
axes[1].legend(fontsize=8)
fig.suptitle('Same 2.81M-parameter model, same crop split; only sampling changes')
fig.savefig(root/'learning_curves.png',dpi=160)
plt.close(fig)

def confusion(rows):
    counts = np.zeros((4,4),dtype=np.int64)
    for row in rows:
        counts[row['label'],int(np.argmax(row['probabilities']))] += 1
    return counts

def metrics(counts):
    accuracy = np.trace(counts)/max(1,counts.sum())
    denominators = counts.sum(axis=0)+counts.sum(axis=1)
    f1 = np.mean(np.divide(2*np.diag(counts),denominators,out=np.zeros(4),where=denominators>0))
    return np.array([accuracy,f1])

def paired_interval(left,right):
    # Bootstrap source groups, not individual crops. This is conditional on this
    # development set and checkpoint selection, not an unbiased final-test CI.
    assert [(r['path'],r['label'],r['group']) for r in left] == [(r['path'],r['label'],r['group']) for r in right]
    groups = sorted({r['group'] for r in left})
    indices = {g:i for i,g in enumerate(groups)}
    matrices = np.zeros((2,len(groups),4,4),dtype=np.int64)
    for side, rows in enumerate([left,right]):
        for r in rows:
            matrices[side,indices[r['group']],r['label'],int(np.argmax(r['probabilities']))]+=1
    rng=np.random.default_rng(20260920)
    deltas=[]
    for _ in range(2000):
        sample=rng.integers(len(groups),size=len(groups))
        deltas.append(metrics(matrices[1,sample].sum(axis=0))-metrics(matrices[0,sample].sum(axis=0)))
    return {'source_groups':len(groups),'delta_accuracy_macro_f1':(metrics(confusion(right))-metrics(confusion(left))).tolist(),
            'percentile95_accuracy_macro_f1':np.quantile(deltas,[.025,.975],axis=0).T.tolist()}

comparisons={}
for seed in [1,2]:
    comparisons[f'balanced_minus_ordinary_seed{seed}']=paired_interval(predictions[f'ordinary_seed{seed}'],predictions[f'balanced_seed{seed}'])
    prior_best=json.loads((old/f'seed{seed}/best.json').read_text())
    prior=[json.loads(l) for l in (old/f'seed{seed}'/f'epoch{prior_best["epoch"]}_predictions.jsonl').read_text().splitlines()]
    comparisons[f'longer_minus_pilot_seed{seed}']=paired_interval(prior,predictions[f'ordinary_seed{seed}'])
averages={sampler:float(np.mean([r['macro_f1'] for r in records if r['sampler']==sampler])) for sampler in ['ordinary','balanced']}
selected=max(averages,key=averages.get)
summary={'runs':records,'mean_macro_f1':averages,'selected_sampler':selected,'selection_rule':'highest mean development macro-F1 across the two seeds',
         'comparisons':comparisons,'uncertainty_caveat':'Group bootstrap on development data used for checkpoint/recipe selection; not a pristine final-test interval; residual camera leakage remains'}
(root/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))

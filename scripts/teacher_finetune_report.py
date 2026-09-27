#!/usr/bin/env python3
"""Compare final EMA teacher fine-tunes with their unchanged baseline."""
import json
from pathlib import Path
import numpy as np

ROOT=Path('evidence/teacher_finetune_2026_09_20')
contract=json.loads((ROOT/'contract.json').read_text())
names=['baseline','finetuned_seed1','finetuned_seed2']
reports={n:json.loads((ROOT/f'{n}.json').read_text()) for n in names}
maps={}
for name,r in reports.items():
    assert [im['id'] for im in r['per_image']]==[im['id'] for im in contract['images']]
    assert r['metadata']['parameters']==reports['baseline']['metadata']['parameters']
    m={}
    for im in r['per_image']:
        for x in im['matches']:
            key=(im['id'],x['annotation_id']);assert key not in m;m[key]=x
    maps[name]=m
keys=set.intersection(*(set(m) for m in maps.values()))
common={}
for name,m in maps.items():
    correct=sum(m[k]['truth']==m[k]['predicted'] for k in keys)
    confusion=np.zeros((4,15),dtype=int)
    for k in keys:confusion[m[k]['truth']-1,m[k]['predicted']]+=1
    common[name]={'correct':correct,'n':len(keys),'accuracy':correct/len(keys),'confusion':confusion.tolist()}

def bootstrap(a):
    a=np.array(a);rng=np.random.default_rng(20260920)
    s=np.array([a[rng.integers(len(a),size=len(a))].sum(axis=0) for _ in range(10000)])
    return {'delta':float(a[:,1].sum()/a[:,0].sum()),'image_bootstrap_percentile95':np.percentile(s[:,1]/s[:,0],[2.5,97.5]).tolist(),'images':len(a)}

comparisons={}
base=reports['baseline']
for name in names[1:]:
    r=reports[name];counts={}
    for k in keys:
        a,b=maps['baseline'][k],maps[name][k];assert a['truth']==b['truth']
        counts.setdefault(k[0],np.zeros(2))[:]+=np.array([1,int(b['predicted']==b['truth'])-int(a['predicted']==a['truth'])])
    all_gt=[]
    for a,b in zip(base['per_image'],r['per_image']):
        assert a['passenger_gt']==b['passenger_gt']
        all_gt.append([a['passenger_gt'],b['correct']-a['correct']])
    class_deltas=[{'class':a['class'],'baseline_AP':a['AP'],'candidate_AP':b['AP'],
        'delta':b['AP']-a['AP'] if a['AP'] is not None and b['AP'] is not None else None}
        for a,b in zip(base['per_class_ap'],r['per_class_ap'])]
    comparisons[name]={'AP_delta':r['AP']-base['AP'],'AP50_delta':r['AP50']-base['AP50'],
        'common_object_subtype':bootstrap(list(counts.values())),
        'correct_localized_passenger_fraction_all_GT':bootstrap(all_gt),
        'correct_count_delta':r['correct']-base['correct'],'localized_count_delta':r['localized']-base['localized'],
        'per_class_AP':class_deltas,
        'passenger_mean_AP_delta':float(np.mean([x['delta'] for x in class_deltas if x['class']<=4 and x['delta'] is not None])),
        'nonpassenger_mean_AP_delta':float(np.mean([x['delta'] for x in class_deltas if x['class']>4 and x['delta'] is not None]))}
training={}
for seed in [1,2]:
    meta=json.loads((ROOT/f'seed{seed}/provenance.json').read_text());assert meta['completed'] and meta['updates']==contract['updates']
    assert meta['final_sha256']==reports[f'finetuned_seed{seed}']['metadata']['checkpoint_sha256']
    training[f'seed{seed}']={k:meta[k] for k in ['updates','history','elapsed_seconds','peak_allocated_bytes','trainable_parameters','parameter_probe_max_changes','final_sha256','script_sha256','contract_sha256','training_manifest_sha256']}
for k in ['script_sha256','contract_sha256','training_manifest_sha256']:
    assert training['seed1'][k]==training['seed2'][k]
summary={'training':training,'common_objects':common,'paired_comparisons':comparisons,
    'whole_cohort':{n:{k:r[k] for k in ['AP','AP50','passenger_gt','localized','correct','conditional_accuracy','localized_fraction','correct_fraction_all_gt']} for n,r in reports.items()},
    'timings':{n:r['metadata']['forward_batch1_ms'] for n,r in reports.items()},
    'scope':'Fixed final EMA, two seeds, no tuning/selection on validation. Image bootstrap is not camera-disjoint; no AP uncertainty. Common objects exclude misses, so all-GT and all-class results must also be considered.'}
(ROOT/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))

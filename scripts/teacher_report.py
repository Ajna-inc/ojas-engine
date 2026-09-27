#!/usr/bin/env python3
"""Paired teacher comparison by ground-truth identity; never used for inference routing."""
import json
from pathlib import Path
import numpy as np

root = Path('evidence/teacher_benchmark_2026_09_20')
contract = json.loads((root/'contract.json').read_text())
names = ['rtdetr_s','rtdetr_x','dfine_x']
reports = {n:json.loads((root/f'{n}.json').read_text()) for n in names}
maps = {}
for n,r in reports.items():
    assert [im['id'] for im in r['per_image']] == [im['id'] for im in contract['images']]
    m = {}
    for im in r['per_image']:
        assert im['localized'] == len(im['matches'])
        for v in im['matches']:
            key = (im['id'],v['annotation_id'])
            assert key not in m
            m[key] = v
    maps[n] = m

def paired(keys, left, right):
    counts = {}
    for k in sorted(keys):
        a,b = maps[left][k],maps[right][k]
        assert a['truth'] == b['truth']
        counts.setdefault(k[0],np.zeros(2))[:] += np.array([1,int(b['predicted']==b['truth'])-int(a['predicted']==a['truth'])])
    a = np.array(list(counts.values()))
    rng = np.random.default_rng(20260920)
    samples = np.array([a[rng.integers(len(a),size=len(a))].sum(0) for _ in range(10000)])
    return {'objects':len(keys),'images':len(a),'delta_accuracy':float(a[:,1].sum()/a[:,0].sum()),
        'image_bootstrap_percentile95':np.percentile(samples[:,1]/samples[:,0],[2.5,97.5]).tolist()}

def comparison(which):
    keys = set.intersection(*(set(maps[n]) for n in which))
    result = {'objects_matched_by_all':len(keys),'models':{}}
    for n in which:
        cm = np.zeros((4,15),dtype=int)
        for k in keys:
            v=maps[n][k];cm[v['truth']-1,v['predicted']]+=1
        correct = sum(maps[n][k]['predicted']==maps[n][k]['truth'] for k in keys)
        result['models'][n]={'correct':correct,'accuracy':correct/len(keys),'confusion':cm.tolist()}
    result['oracle_correct_any_model'] = sum(any(maps[n][k]['predicted']==maps[n][k]['truth'] for n in which) for k in keys)
    result['oracle_accuracy'] = result['oracle_correct_any_model']/len(keys)
    result['paired_vs_small'] = {n:paired(keys,'rtdetr_s',n) for n in which if n!='rtdetr_s'}
    return result

summary = {'three_detectors_common_objects':comparison(names),
    'individual_detectors':{n:{k:v for k,v in r.items() if k not in ['metadata','per_image']} for n,r in reports.items()},
    'timings':{n:{k:r['metadata'][k] for k in ['parameters','forward_batch1_ms','peak_allocated_bytes','peak_reserved_bytes']} for n,r in reports.items()},
    'caveats':['Common-object comparison excludes objects missed by any included model; also report individual all-GT coverage.',
        'Bootstrap resamples images, not cameras; adjacent scenes may be correlated. No AP confidence interval.',
        'Oracle chooses using truth and is a diagnostic only, not a trained ensemble or attainable routing guarantee.',
        'All-class AP uses standard pycocotools on the same subset; older Ojas results used a different evaluator/backend.',
        'BF16 SigLIP2 and FP32 detector timing are different precision/operations and crop versus whole-frame workloads.']}
siglip = root/'siglip2.json'
if siglip.exists():
    report = json.loads(siglip.read_text())
    preds = json.loads((root/'siglip2_predictions.json').read_text())
    maps['siglip2'] = {}
    for p in preds:
        if p['truth'] not in [1,2,3,4]: continue
        key=(p['image_id'],p['annotation_id'])
        assert key not in maps['siglip2']
        maps['siglip2'][key]={'truth':p['truth'],'predicted':p['predicted']}
    keys=set(maps['siglip2'])
    assert keys <= set(maps['rtdetr_s'])
    summary['siglip2_same_baseline_crops']={'objects':len(keys),'correct':report['correct'],'accuracy':report['accuracy'],
        'small_correct_same_objects':report['detector_correct_same_objects'],
        'small_accuracy_same_objects':report['detector_correct_same_objects']/len(keys),
        'fixed':report['fixed'],'broken':report['broken'],'paired_vs_small':paired(keys,'rtdetr_s','siglip2')}
    summary['four_teachers_common_objects']=comparison(names+['siglip2'])
    summary['timings']['siglip2']={k:report[k] for k in ['total_parameters','vision_parameters','forward_batch1_ms','peak_allocated_bytes','peak_reserved_bytes']}
(root/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))

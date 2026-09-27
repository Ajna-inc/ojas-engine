#!/usr/bin/env python3
"""Compare whole-scene detection and fixed baseline-object cohorts without hiding misses."""
import argparse,json,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(root):
    contract=json.loads((root/'contract.json').read_text());names=['baseline']+[f'{arm}_{kind}' for arm in contract['arms'] for kind in ['selected','final']]
    reports={name:json.loads((root/f'{name}.json').read_text()) for name in names}
    baseline=reports['baseline'];maps={};groups={r['id']:r['group'] for r in contract['images']}
    for name,r in reports.items():
        assert [im['id'] for im in r['per_image']]==[im['id'] for im in contract['images']]
        assert r['metadata']['contract_sha256']==digest(root/'contract.json')
        assert r['metadata']['parameters']==baseline['metadata']['parameters']
        maps[name]={(im['id'],m['annotation_id']):m for im in r['per_image'] for m in im['matches']}
    crops=json.loads((root/'baseline_crops.json').read_text())
    fixed={(r['image_id'],r['annotation_id']) for r in crops if r['truth'] is not None and 1<=r['truth']<=4}
    assert fixed<=maps['baseline'].keys()
    def metrics(mapping,keys):
        confusion=np.zeros((4,16),dtype=int)
        for k in keys:
            truth=maps['baseline'][k]['truth'];pred=mapping[k]['predicted'] if k in mapping else 15
            confusion[truth-1,pred]+=1
        correct=sum(confusion[i,i+1] for i in range(4));n=int(confusion.sum())
        recall=[float(confusion[i,i+1]/max(1,confusion[i].sum())) for i in range(4)]
        f1=[float(2*confusion[i,i+1]/max(1,confusion[i].sum()+confusion[:,i+1].sum())) for i in range(4)]
        return dict(correct=int(correct),n=n,accuracy=correct/max(1,n),macro_f1=float(np.mean(f1)),per_class_recall=recall,confusion=confusion.tolist(),misses=int(confusion[:,15].sum()))
    def bootstrap(rows):
        grouped=defaultdict(lambda:np.zeros(2))
        for image_id,n,delta in rows:grouped[groups[image_id]]+=np.array([n,delta])
        a=np.array(list(grouped.values()));rng=np.random.default_rng(20260920)
        z=np.array([a[rng.integers(len(a),size=len(a))].sum(0) for _ in range(10000)])
        valid=z[:,0]>0
        return dict(delta=float(a[:,1].sum()/a[:,0].sum()),percentile95=np.percentile(z[valid,1]/z[valid,0],[2.5,97.5]).tolist(),groups=len(a))
    summary={'contract_sha256':digest(root/'contract.json'),'scope':'One seed per arm; consumed ST benchmark and automatic source groups, not independent camera confirmation. Fixed baseline routed-object cohort counts newly missed objects as wrong. FP32 PyTorch cohort may differ from earlier Ojas BF16 crop cohort. No AP uncertainty.','whole_scene':{},'fixed_baseline_routed_objects':{},'paired':{},'training':{}}
    for name,r in reports.items():
        summary['whole_scene'][name]={k:r[k] for k in ['AP','AP50','passenger_gt','localized','correct','conditional_accuracy','localized_fraction','correct_fraction_all_gt','per_class_ap']}
        summary['whole_scene'][name]['forward_batch1_ms']=r['metadata']['forward_batch1_ms']
        summary['fixed_baseline_routed_objects'][name]=metrics(maps[name],fixed)
        if name=='baseline':continue
        rows=[]
        for k in fixed:
            a=maps['baseline'][k];b=maps[name].get(k)
            rows.append((k[0],1,int(b is not None and b['predicted']==a['truth'])-int(a['predicted']==a['truth'])))
        common=maps['baseline'].keys()&maps[name].keys()
        summary['paired'][name]={'fixed_cohort_subtype':bootstrap(rows),'common_localized_baseline':metrics(maps['baseline'],common),'common_localized_candidate':metrics(maps[name],common),'all_passenger_gt_correctness':bootstrap([(a['id'],a['passenger_gt'],b['correct']-a['correct']) for a,b in zip(baseline['per_image'],r['per_image'])]),'AP_delta':r['AP']-baseline['AP']}
    for arm in contract['arms']:
        prov=json.loads((root/arm/'provenance.json').read_text());assert prov['completed'] and prov['updates']==contract['updates']
        assert prov['contract_sha256']==digest(root/'contract.json')
        assert reports[f'{arm}_final']['metadata']['checkpoint_sha256']==digest(root/arm/f"epoch{contract['epochs']}.pth")
        selected=json.loads((root/arm/'selection.json').read_text())
        if selected['epoch']:
            assert reports[f'{arm}_selected']['metadata']['checkpoint_sha256']==selected['checkpoint_sha256']
        else:assert reports[f'{arm}_selected']['metadata']['checkpoint_sha256']==baseline['metadata']['checkpoint_sha256']
        summary['training'][arm]={k:prov[k] for k in ['updates','history','selection','peak_allocated_bytes','elapsed_seconds','trainable_parameters','zero_update_max_absolute_difference','script_sha256']}
    assert len({p['script_sha256'] for p in summary['training'].values()})==1
    (root/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['training','whole_scene']},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();main(a.root)

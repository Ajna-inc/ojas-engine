#!/usr/bin/env python3
"""Flag ST/MV subtype disagreements for review; do not relabel either source."""
import argparse,json
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

def main(root):
    base=Path('/data/datasets/iisc-aim/UVH-26/UVH-26-Train')
    docs={s:json.loads((base/f'UVH-26-{s}-Train.json').read_text()) for s in ['ST','MV']}
    by={}
    for s,d in docs.items():
        names={r['id']:r['file_name'] for r in d['images']};a=defaultdict(list)
        for r in d['annotations']:
            if 1<=r['category_id']<=4 and not r.get('iscrowd',0):a[names[r['image_id']]].append(r)
        by[s]=a
    rows=[];matched=0
    for name in sorted(by['ST'].keys()&by['MV'].keys()):
        a,b=by['ST'][name],by['MV'][name];x=np.array([v['bbox'] for v in a]);y=np.array([v['bbox'] for v in b])
        inter=np.maximum(0,np.minimum(x[:,None,:2]+x[:,None,2:],y[None,:,:2]+y[None,:,2:])-np.maximum(x[:,None,:2],y[None,:,:2])).prod(2)
        iou=inter/np.maximum(x[:,2:].prod(1)[:,None]+y[:,2:].prod(1)[None,:]-inter,1e-8)
        # Only strong geometry matches qualify; maximize their count before overlap.
        qi,ti=linear_sum_assignment(-((iou>=.7)*1000+iou))
        for i,j in zip(qi,ti):
            if iou[i,j]<.7:continue
            matched+=1
            if a[i]['category_id']!=b[j]['category_id']:
                rows.append({'file_name':name,'st_annotation_id':a[i]['id'],'mv_annotation_id':b[j]['id'],'mv_image_id':b[j]['image_id'],'st_class':a[i]['category_id'],'mv_class':b[j]['category_id'],'iou':float(iou[i,j]),'st_box':a[i]['bbox'],'mv_box':b[j]['bbox']})
    splits={r['id']:s for s in ['train','dev'] for r in json.loads((root/f'{s}_manifest.json').read_text())}
    for r in rows:r['current_split']=splits.get(r['mv_image_id'],'excluded')
    (root/'label_disagreement_review.json').write_text(json.dumps(rows,indent=2)+'\n')
    report={'matched_passenger_pairs_iou_0_7':matched,'subtype_disagreements':len(rows),'fraction':len(rows)/matched,'by_current_split':dict(Counter(r['current_split'] for r in rows)),'policy':'One-to-one geometric association. Review candidates, not proven errors; no automatic relabeling. Canonical MV retained for this experiment.'}
    (root/'label_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();main(a.root)

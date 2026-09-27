#!/usr/bin/env python3
"""Diagnostic only: compare ST/MV labels on the same fixed baseline proposals."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from teacher_detector_bench import sha
r=Path('evidence/iisc_small_full_2026_09_20')
contract=json.loads((r/'contract.json').read_text());mv_path=Path(contract['annotations']).with_name('UVH-26-MV-Val.json');st=json.loads(Path(contract['annotations']).read_text());mv=json.loads(mv_path.read_text())
names_st={v['id']:v['file_name'] for v in st['images']};names_mv={v['id']:v['file_name'] for v in mv['images']}
a=defaultdict(list);b=defaultdict(list)
for item in st['annotations']:
 if 1<=item['category_id']<=4 and not item.get('iscrowd',0):a[names_st[item['image_id']]].append(item)
for item in mv['annotations']:
 if 1<=item['category_id']<=4 and not item.get('iscrowd',0):b[names_mv[item['image_id']]].append(item)
proposals=json.loads((r/'baseline_crops.json').read_text());fixed={(q['image_id'],q['annotation_id']):q for q in proposals if q['truth'] is not None and 1<=q['truth']<=4}
rows=[]
for info in contract['images']:
 name=info['file_name'];aa=a[name];bb=b[name]
 if not aa or not bb:continue
 x=np.array([v['bbox'] for v in aa]);y=np.array([v['bbox'] for v in bb]);inter=np.maximum(0,np.minimum(x[:,None,:2]+x[:,None,2:],y[None,:,:2]+y[None,:,2:])-np.maximum(x[:,None,:2],y[None,:,:2])).prod(2);iou=inter/np.maximum(x[:,2:].prod(1)[:,None]+y[:,2:].prod(1)[None,:]-inter,1e-8)
 qi,ti=linear_sum_assignment(-((iou>=.7)*1000+iou))
 for i,j in zip(qi,ti):
  k=(info['id'],aa[i]['id'])
  if iou[i,j]<.7 or k not in fixed:continue
  q=fixed[k];rows.append(dict(image_id=info['id'],st_annotation_id=aa[i]['id'],mv_annotation_id=bb[j]['id'],prediction=q['class'],st_class=aa[i]['category_id'],mv_class=bb[j]['category_id'],iou=float(iou[i,j])))
summary={'n':len(rows),'st_correct':sum(v['prediction']==v['st_class'] for v in rows),'mv_correct':sum(v['prediction']==v['mv_class'] for v in rows),'label_disagreements':sum(v['st_class']!=v['mv_class'] for v in rows),'mv_annotations_sha256':sha(mv_path),'scope':'Same unchanged predictions on one-to-one geometrically matched ST/MV passenger labels, restricted to frozen baseline routed cohort. Diagnostic label sensitivity, not a model improvement or permission to replace ST labels. ST remains the primary benchmark.'}
(r/'annotation_sensitivity.json').write_text(json.dumps(dict(summary,rows=rows),indent=2)+'\n');print(summary)

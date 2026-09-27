#!/usr/bin/env python3
"""Cache frozen IISc proposals and paired GT crops, retaining background audits."""
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from teacher_detector_bench import load_model, overlap, sha
from passenger_full_prepare import geometry, NAMES, SPLITS, BASE

ROOT = Path('evidence/passenger_proposals_2026_09_20').resolve()

class Frames(Dataset):
    def __init__(self, rows): self.rows=rows
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i][1]
        assert sha(r['cache_path'])==r['cache_sha256']
        x=np.load(r['cache_path'],allow_pickle=False)
        return torch.from_numpy(x.transpose(2,0,1).copy()).float()/255, i

def main():
    ROOT.mkdir(exist_ok=False)
    (ROOT/'crops').mkdir()
    annpath=BASE/'UVH-26/UVH-26-Train/UVH-26-MV-Train.json'
    coco=json.loads(annpath.read_text()); byimage=defaultdict(list)
    for a in coco['annotations']:
        if not a.get('iscrowd',0): byimage[a['image_id']].append(a)
    jobs=[(s,r) for s in ['train','dev'] for r in json.loads((SPLITS/f'{s}_manifest.json').read_text())]
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
    model,mapping,meta=load_model('rtdetr_s')
    assert mapping==list(range(15))
    records=[]; manifests={'proposal':[],'paired_gt':[]}; counts=Counter()
    loader=DataLoader(Frames(jobs),batch_size=8,num_workers=4,pin_memory=True,shuffle=False)
    with torch.inference_mode(), (ROOT/'all_proposals.jsonl').open('x') as audit:
        for bi,(x,indices) in enumerate(loader):
            out=model(x.cuda())
            logits=out['pred_logits'].float().cpu().numpy()
            boxes=out['pred_boxes'].float().cpu().numpy()
            assert np.isfinite(logits).all() and np.isfinite(boxes).all()
            for k,idx in enumerate(indices.tolist()):
                split,info=jobs[idx]; gt=byimage[info['id']]
                gb=np.array([[a['bbox'][0],a['bbox'][1],a['bbox'][0]+a['bbox'][2],a['bbox'][1]+a['bbox'][3]] for a in gt]).reshape(-1,4)
                xyxy=np.concatenate([boxes[k,:,:2]-boxes[k,:,2:]/2,boxes[k,:,:2]+boxes[k,:,2:]/2],1)*np.array([info['width'],info['height']]*2)
                scores=torch.from_numpy(logits[k,:,1:]).sigmoid().numpy()
                labels=scores.argmax(1)+1; conf=scores.max(1); used=set(); rgb=None
                for q in np.argsort(-conf,kind='stable'):
                    if conf[q]<.3: continue
                    b=xyxy[q]; match=None; maxiou=0.
                    if len(gb):
                        ious=overlap(b,gb); maxiou=float(ious.max())
                        for j in used: ious[j]=-1
                        j=int(ious.argmax())
                        if ious[j]>=.5: used.add(j); match=gt[j]
                    if not 1<=labels[q]<=4 or min(b[2:]-b[:2])<48: continue
                    truth=match['category_id'] if match else None
                    validity=1 if truth in [1,2,3,4] else (0 if match or maxiou<.1 else None)
                    record=dict(image_id=info['id'],split=split,group=info['group'],query=int(q),
                        box_xyxy=b.tolist(),detector_logits=logits[k,q].tolist(),detector_class=int(labels[q]),
                        score=float(conf[q]),annotation_id=match['id'] if match else None,truth=truth,
                        validity=validity,max_iou=maxiou,source_sha256=info['sha256'])
                    counts[f'{split}/validity_{validity}']+=1
                    # Retain every proposal in the audit; unmatched is never a subtype target.
                    audit.write(json.dumps(record)+'\n')
                    if validity!=1: continue
                    if min(match['bbox'][2:])<48:
                        counts[f'{split}/matched_gt_under48']+=1; continue
                    if rgb is None:
                        assert sha(info['path'])==info['sha256']
                        with Image.open(info['path']) as im: rgb=im.convert('RGB')
                    for arm,box in [('proposal',[b[0],b[1],b[2]-b[0],b[3]-b[1]]),('paired_gt',match['bbox'])]:
                        bounds=geometry(box,*rgb.size)
                        assert bounds[2]>bounds[0] and bounds[3]>bounds[1]
                        dest=ROOT/'crops'/f'{arm}_{info["id"]}_{q}.jpg'
                        rgb.crop(bounds).save(dest,quality=80,subsampling=0)
                        manifests[arm].append(dict(record,version=2,dataset='uvh-mv-train',
                            path=str(dest),label=truth-1,class_name=NAMES[truth-1],raw_category_id=truth,
                            box=list(map(float,box)),crop_xyxy=bounds,context=.15,source=info['path'],
                            width=bounds[2]-bounds[0],height=bounds[3]-bounds[1],crop_sha256=sha(dest)))
            if (bi+1)%64==0: print('cached frames',min((bi+1)*8,len(jobs)),'/',len(jobs),flush=True)
    # Identical group quarantine for both geometries; exact crop duplicates cannot cross splits.
    quarantine=set()
    for rows in manifests.values():
        devhash={r['crop_sha256'] for r in rows if r['split']=='dev'}
        quarantine.update(r['group'] for r in rows if r['split']=='train' and r['crop_sha256'] in devhash)
    for arm,rows in manifests.items():
        rows=[r for r in rows if r['group'] not in quarantine]
        path=ROOT/f'{arm}.jsonl';path.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
        for seed in [1,2]:
            config=dict(manifest=str(path),init=meta['checkpoint'],batch=16,size=224,epochs=8,
                epoch_samples=3961,lr=.0001,head_lr=.001,balanced=True,out=str(ROOT/f'{arm}_seed{seed}'),
                seed=seed,overfit_steps=0,backbone_stage=3,evaluate_train=True)
            (ROOT/f'{arm}_seed{seed}.json').write_text(json.dumps(config,indent=2)+'\n')
        counts.update({f'{arm}/{s}/{NAMES[c]}':sum(r['split']==s and r['label']==c for r in rows) for s in ['train','dev'] for c in range(4)})
    result=dict(metadata=meta,counts=dict(counts),quarantined_groups=sorted(quarantine),
        annotation_sha256=sha(annpath),script_sha256=sha(__file__),
        manifests={a:sha(ROOT/f'{a}.jsonl') for a in manifests},
        contract='Matched-budget paired GT versus actual proposals; subtype supervision only for matched passengers. Background/ambiguous proposals audited separately, not assigned a subtype. This stage does not train a validity head or authorize production fusion.',
        caveats=['Proposal errors on original pretraining sources may understate unseen errors.',
                 'Ground truth matching used only to create training targets, never inference routing.',
                 'FP32 PyTorch proposals; Ojas BF16 deployment evaluation remains required.'])
    (ROOT/'audit.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
    print(json.dumps(dict(counts),indent=2),flush=True)

if __name__=='__main__': main()

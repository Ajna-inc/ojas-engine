#!/usr/bin/env python3
"""Freeze full MV data, source groups, and a fixed consumed ST benchmark."""
import argparse, hashlib, io, json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def key(s): return hashlib.sha256(str(s).encode()).hexdigest()

def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    base=Path('/data/datasets/iisc-aim/UVH-26')
    tp=base/'UVH-26-Train/UVH-26-MV-Train.json'; vp=base/'UVH-26-Val/UVH-26-ST-Val.json'
    train=json.loads(tp.read_text()); val=json.loads(vp.read_text())
    assert {c['id']:c['name'] for c in train['categories']}=={c['id']:c['name'] for c in val['categories']}
    ids=set(json.loads(Path('evidence/passenger_sam_2026_09_20/evaluation_contract.json').read_text())['image_ids'])
    cache=root/'cache640';cache.mkdir()
    jobs=[]
    for domain,doc,sub in [('train',train,'UVH-26-Train'),('benchmark',val,'UVH-26-Val')]:
        paths={}
        for p in (base/sub/'data').rglob('*'):
            if p.is_file():
                assert p.name not in paths, p.name
                paths[p.name]=p
        for row in doc['images']:
            if domain=='benchmark' and row['id'] not in ids: continue
            jobs.append((domain,row,paths[row['file_name']]))
    def inspect(job):
        domain,r,p=job; data=p.read_bytes()
        with Image.open(io.BytesIO(data)) as im:
            if im.size!=(r['width'],r['height']): return {'excluded':True,'domain':domain,'id':r['id'],'actual_size':im.size,'declared_size':[r['width'],r['height']]}
            rgb=im.convert('RGB'); px=np.asarray(rgb.convert('L').resize((9,8)))
            dh=sum(int(v)<<i for i,v in enumerate((px[:,:-1]>px[:,1:]).ravel()))
            dest=cache/f"{domain}_{r['id']}.npy"
            if domain=='train': np.save(dest,np.asarray(rgb.resize((640,640),Image.Resampling.BILINEAR)),allow_pickle=False)
        return dict(r,domain=domain,path=str(p),sha256=hashlib.sha256(data).hexdigest(),dhash=f'{dh:016x}',cache_path=str(dest) if domain=='train' else None,cache_sha256=sha(dest) if domain=='train' else None)
    rows=[];excluded=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i,r in enumerate(pool.map(inspect,jobs)):
            (excluded if r.get('excluded') else rows).append(r)
            if (i+1)%500==0: print('audited',i+1,'/',len(jobs),flush=True)
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):
        a,b=find(i),find(j);parent[max(a,b)]=min(a,b)
    buckets=defaultdict(list);exact={}
    for i,r in enumerate(rows):
        if r['sha256'] in exact:union(i,exact[r['sha256']])
        exact[r['sha256']]=i;h=int(r['dhash'],16); candidates=set()
        for b in range(4):candidates.update(buckets[b,(h>>(16*b))&65535])
        for j in candidates:
            if (h^int(rows[j]['dhash'],16)).bit_count()<=3:union(i,j)
        for b in range(4):buckets[b,(h>>(16*b))&65535].append(i)
    benchmark_groups={find(i) for i,r in enumerate(rows) if r['domain']=='benchmark'}
    kept=[];dev=[];bench=[];overlap=[]
    for i,r in enumerate(rows):
        r['group']=f"uvh:{rows[find(i)]['domain']}:{rows[find(i)]['id']}"
        if r['domain']=='benchmark':bench.append(r)
        elif find(i) in benchmark_groups:overlap.append(r)
        elif int(key('20260920:'+r['group'])[:8],16)%20==0:dev.append(r)
        else:kept.append(r)
    # One canonical MV annotation source; never add overlapping ST labels as new examples.
    def write(name,value):(root/name).write_text(json.dumps(value,indent=2)+'\n')
    for name,data in [('train_manifest.json',kept),('dev_manifest.json',dev),('excluded_dimensions.json',excluded),('excluded_benchmark_overlap.json',overlap)]:write(name,data)
    ids_train={r['id'] for r in kept}; counts=Counter(a['category_id'] for a in train['annotations'] if a['image_id'] in ids_train and not a.get('iscrowd',0))
    write('audit.json',{'train_images':len(kept),'dev_images':len(dev),'benchmark_images':len(bench),'excluded_dimensions':len(excluded),'excluded_train_near_benchmark':len(overlap),'train_annotations_by_class':dict(counts),'passenger_train_annotations':sum(counts[i] for i in range(1,5)),'group_policy':'Exact SHA256 or dHash distance <=3 connected components; hash split 5% development. Not a camera identity guarantee. Development sources seen by original IISc pretraining; benchmark consumed.'})
    contract={'annotations':str(vp),'annotations_sha256':sha(vp),'images':bench,'training_manifest':str((root/'train_manifest.json').resolve()),'training_annotations':str(tp),'training_annotations_sha256':sha(tp),'model':'rtdetr_s','epochs':3,'seed':1,'micro_batch':4,'accumulation':2,'backbone_lr':1e-6,'other_lr':1e-5,'weight_decay':1e-4,'warmup_updates':100,'gradient_clip':.1,'ema_decay':.999,'selection':'Per-arm best development correct passenger fraction over ALL passenger GT, subject to all-class AP >= baseline AP - 0.005; baseline eligible. Three full epochs each, identical seeds and updates. Consumed ST benchmark evaluated only after selection.','arms':['continuation','robust'],'caveat':'Canonical MV labels not human-corrected; ST/MV conflicts remain review candidates. Full source training is continuation of a checkpoint already trained on this domain, not new independent data.'}
    write('contract.json',contract)
    droot=root/'development';droot.mkdir();dc={'annotations':str(tp),'annotations_sha256':sha(tp),'images':dev,'scope':'Grouped diagnostic development; included in IISc pretraining, excluded from current gradient updates.'};(droot/'contract.json').write_text(json.dumps(dc,indent=2)+'\n')
    print(json.dumps(json.loads((root/'audit.json').read_text()),indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();prepare(a.root.resolve())

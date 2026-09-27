from pathlib import Path
import sys,csv,json,collections
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path.insert(0,str(ROOT/'.tools/plotting'))
import numpy as np
run=ROOT/'build/map-capture/run-20260925-232109';shape=(342,342)
ground=np.empty(shape,dtype=object);reach=np.zeros(shape,dtype=bool);height=np.zeros(shape)
for r in csv.DictReader((run/'tww3_bai_map_capture_grid.csv').open()):
    i,j=int(r['iz']),int(r['ix']);ground[i,j]=r['ground'];reach[i,j]=r['reach_cavalry']=='1';height[i,j]=float(r['height'])
def components(mask):
    labels=np.full(shape,-1,dtype=np.int32);groups=[]
    for i,j in zip(*np.nonzero(mask)):
        if labels[i,j]>=0:continue
        k=len(groups);q=collections.deque([(int(i),int(j))]);labels[i,j]=k;pts=[]
        while q:
            a,b=q.popleft();pts.append((a,b))
            for x,z in ((a-1,b),(a+1,b),(a,b-1),(a,b+1)):
                if 0<=x<shape[0] and 0<=z<shape[1] and mask[x,z] and labels[x,z]<0:labels[x,z]=k;q.append((x,z))
        groups.append(pts)
    return labels,groups
dry=reach&~np.isin(ground,['deep_water','shallow_water']);land_labels,land=components(dry)
wet=reach&(ground=='shallow_water');wet_labels,wet_groups=components(wet)
results=[]
for k,pts in enumerate(wet_groups):
    banks={}
    for a,b in pts:
        for i,j in ((a-1,b),(a+1,b),(a,b-1),(a,b+1)):
            if 0<=i<342 and 0<=j<342 and land_labels[i,j]>=0:
                label=int(land_labels[i,j]);banks.setdefault(label,[]).append((i,j))
    major=[n for n in banks if len(land[n])>=100]
    if len(major)<2:continue
    pairs=[(a,b) for a in banks[major[0]] for b in banks[major[1]]]
    start,end=min(pairs,key=lambda p:(p[0][0]-p[1][0])**2+(p[0][1]-p[1][1])**2)
    allowed=set(pts)|{start,end};q=collections.deque([start]);prev={start:None}
    while q:
        a,b=q.popleft()
        if (a,b)==end:break
        for v in ((a-1,b),(a+1,b),(a,b-1),(a,b+1)):
            if v in allowed and v not in prev:prev[v]=(a,b);q.append(v)
    assert end in prev;path=[];v=end
    while v is not None:path.append(v);v=prev[v]
    path.reverse()
    def pos(p):return dict(ix=p[1],iz=p[0],x=-512+(p[1]+.5)*3,z=-512+(p[0]+.5)*3,height=float(height[p]))
    results.append(dict(id=len(results),cells=len(pts),bank_components=major,bank_sizes=[len(land[n]) for n in major],grid_path=[pos(v) for v in path],world_center={'x':float(np.mean([pos(v)['x'] for v in pts])),'z':float(np.mean([pos(v)['z'] for v in pts]))},bounds={'min_x':min(pos(v)['x'] for v in pts),'max_x':max(pos(v)['x'] for v in pts),'min_z':min(pos(v)['z'] for v in pts),'max_z':max(pos(v)['z'] for v in pts)}))
(HERE/'ford-candidates.json').write_text(json.dumps(results,indent=2))
np.savez_compressed(HERE/'ford-components.npz',land=land_labels,shallow=wet_labels,reach=reach)
print('dry_land_components',sorted([len(p) for p in land],reverse=True)[:10],'shallow_components',len(wet_groups))
print(json.dumps([{k:v for k,v in r.items() if k!='grid_path'}|{'endpoints':[r['grid_path'][0],r['grid_path'][-1]]} for r in results],indent=2))

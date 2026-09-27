"""Find sampled shallow-water connections between reachable dry-land components."""
from pathlib import Path
import sys,csv,gzip,io,json,hashlib,collections,argparse,math
import numpy as np

def components(mask):
    rows,cols=mask.shape;labels=np.full(mask.shape,-1,dtype=np.int32);groups=[]
    for i,j in zip(*np.nonzero(mask)):
        if labels[i,j]>=0:continue
        ident=len(groups);queue=collections.deque([(int(i),int(j))]);labels[i,j]=ident;group=[]
        while queue:
            a,b=queue.popleft();group.append((a,b))
            for iz,ix in ((a-1,b),(a+1,b),(a,b-1),(a,b+1)):
                if 0<=iz<rows and 0<=ix<cols and mask[iz,ix] and labels[iz,ix]<0:
                    labels[iz,ix]=ident;queue.append((iz,ix))
        groups.append(group)
    return labels,groups

def find_connections(ground,reachable,min_land_cells=100):
    assert ground.shape==reachable.shape
    dry=reachable&~np.isin(ground,['shallow_water','deep_water'])
    land,land_groups=components(dry);water,water_groups=components(reachable&(ground=='shallow_water'))
    connections=[];rows,cols=ground.shape
    for ident,group in enumerate(water_groups):
        adjacent=set()
        for a,b in group:
            for iz,ix in ((a-1,b),(a+1,b),(a,b-1),(a,b+1)):
                if 0<=iz<rows and 0<=ix<cols and land[iz,ix]>=0:adjacent.add(int(land[iz,ix]))
        banks=sorted(i for i in adjacent if len(land_groups[i])>=min_land_cells)
        if len(banks)>=2:
            connections.append(dict(water_component=ident,cells=len(group),bank_components=banks,bank_cells=[len(land_groups[i]) for i in banks],cell_bounds={'min_ix':min(b for a,b in group),'max_ix':max(b for a,b in group),'min_iz':min(a for a,b in group),'max_iz':max(a for a,b in group)}))
    return land,water,connections

def run(source,output,step,columns,min_land_cells):
    if not math.isfinite(step) or step<=0:raise ValueError('Step must be positive')
    raw=source.read_bytes();raw=gzip.decompress(raw) if source.suffix=='.gz' else raw
    records=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))));assert records,'Empty grid'
    nrow=max(int(r['iz']) for r in records)+1;ncol=max(int(r['ix']) for r in records)+1
    assert nrow*ncol==len(records),'Incomplete rectangular grid'
    origin_x=float(records[0]['x'])-(int(records[0]['ix'])+.5)*step
    origin_z=float(records[0]['z'])-(int(records[0]['iz'])+.5)*step
    ground=np.empty((nrow,ncol),dtype=object);reachable=np.zeros((nrow,ncol),dtype=bool);seen=set()
    for r in records:
        i,j=int(r['iz']),int(r['ix']);assert i>=0 and j>=0 and (i,j) not in seen;seen.add((i,j))
        assert abs(float(r['x'])-(origin_x+(j+.5)*step))<1e-6 and abs(float(r['z'])-(origin_z+(i+.5)*step))<1e-6
        assert r['inside_radar'] in ('0','1')
        assert all(r[c] in ('-1','0','1') for c in columns)
        ground[i,j]=r['ground'];reachable[i,j]=r['inside_radar']=='1' and all(r[c]=='1' for c in columns)
    land,water,connections=find_connections(ground,reachable,min_land_cells)
    output.mkdir(exist_ok=True,parents=True);np.savez_compressed(output/'components.npz',land=land,shallow=water,reachable=reachable)
    result={'source_csv_sha256':hashlib.sha256(raw).hexdigest(),'tool_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'shape':[nrow,ncol],'step_m':step,'min_x':origin_x,'min_z':origin_z,'reach_columns':columns,'minimum_bank_cells':min_land_cells,'adjacency':4,'connections':connections,'scope':'Grid-derived connection candidates. Adjacent reachable samples do not certify the entire connecting segment or formation clearance; movement validation is separate.'}
    (output/'connections.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--csv',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--step',required=True,type=float);p.add_argument('--reach-columns',nargs='+',default=['reach_cavalry','reach_infantry']);p.add_argument('--min-land-cells',type=int,default=100);a=p.parse_args()
    assert a.min_land_cells>0
    print(json.dumps(run(a.csv,a.output,a.step,a.reach_columns,a.min_land_cells),indent=2))

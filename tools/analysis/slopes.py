"""Derive geometric slopes from verified height samples; no game penalty model."""
import argparse,hashlib,json,math,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def derive(height,step):
    if height.ndim!=2 or min(height.shape)<3 or not np.isfinite(height).all():
        raise ValueError('Expected a complete finite height grid, at least 3 by 3')
    if not math.isfinite(step) or step<=0:raise ValueError('Invalid metre spacing')
    h=np.asarray(height,dtype=np.float64)
    gx=np.full_like(h,np.nan);gz=np.full_like(h,np.nan)
    gx[1:-1,1:-1]=(h[1:-1,2:]-h[1:-1,:-2])/(2*step)
    gz[1:-1,1:-1]=(h[2:,1:-1]-h[:-2,1:-1])/(2*step)
    slope=np.degrees(np.arctan(np.hypot(gx,gz)))
    # Preserve signed neighbour differences; central slope can hide a sharp crest.
    dx=h[:,1:]-h[:,:-1];dz=h[1:,:]-h[:-1,:]
    rise=np.full_like(h,np.nan)
    rise[1:-1,1:-1]=np.maximum.reduce([abs(dx[1:-1,:-1]),abs(dx[1:-1,1:]),abs(dz[:-1,1:-1]),abs(dz[1:,1:-1])])
    return dict(slope_degrees=slope,gradient_x=gx,gradient_z=gz,delta_x_m=dx,delta_z_m=dz,max_neighbor_delta_m=rise)

def render(height_file,metadata_file,output):
    meta=json.loads(metadata_file.read_text(encoding='utf-8'))
    raw=height_file.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==meta['height_npy_sha256'],'Height matrix hash mismatch'
    h=np.load(height_file,allow_pickle=False)
    assert h.shape==(meta['rows'],meta['columns'])
    arrays=derive(h,meta['step_m']);output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'slopes.npz',**arrays)
    slope=arrays['slope_degrees'];rise=arrays['max_neighbor_delta_m']
    stats={'slope_degrees':{k:float(v) for k,v in zip(['min','median','p95','max'],np.nanpercentile(slope,[0,50,95,100]))},'max_adjacent_height_difference_m':float(max(abs(arrays['delta_x_m']).max(),abs(arrays['delta_z_m']).max()))}
    for lang in ['en','ru']:
        titles=['Estimated slope, degrees','Largest difference to a neighbour, m'] if lang=='en' else ['Расчётный уклон, градусы','Наибольший перепад до соседа, м']
        fig,axs=plt.subplots(1,2,figsize=(15,7),layout='constrained')
        for ax,arr,title in zip(axs,[slope,rise],titles):
            cmap=plt.get_cmap('magma').copy();cmap.set_bad('#b8bec5')
            im=ax.imshow(arr,origin='lower',interpolation='nearest',extent=(meta['min_x'],meta['query_max_x'],meta['min_z'],meta['query_max_z']),cmap=cmap)
            ax.set(xlabel='X, m',ylabel='Z, m',title=title,xlim=(meta['min_x'],meta['max_x']),ylim=(meta['min_z'],meta['max_z']))
            fig.colorbar(im,ax=ax,shrink=.75)
        note=f"Geometry from {meta['step_m']:g} m samples; no movement or combat penalty inferred." if lang=='en' else f"Геометрия по замерам {meta['step_m']:g} м; штрафы движения и боя не вычислены."
        fig.suptitle(note,fontsize=14);fig.savefig(output/f'slopes.{lang}.png',dpi=150,bbox_inches='tight');plt.close(fig)
    report={'height_source_sha256':meta['source_csv_sha256'],'height_npy_sha256':meta['height_npy_sha256'],'tool_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'step_m':meta['step_m'],'shape':list(h.shape),'central_difference_baseline_m':2*meta['step_m'],'undefined_border_cells':int(np.isnan(slope).sum()),'stats':stats,'units':{'gradient':'m/m','slope':'degrees','delta':'metres'},'derived_offline':True,'engine_penalty_verified':False,'outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.suffix in ('.npz','.png')}}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--height',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(render(a.height,a.metadata,a.output),indent=2))

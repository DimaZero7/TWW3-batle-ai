from pathlib import Path
import sys,json,math,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path.insert(0,str(ROOT/'.tools/plotting'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
land=np.load(HERE/'ford-components.npz')['land'];candidates=json.loads((HERE/'ford-candidates.json').read_text())
run=ROOT/'build/map-capture/run-20260925-232645';es=[json.loads(s) for s in (run/'tww3_bai_map_capture_events.jsonl').read_text().splitlines()]
def label(e):return int(land[math.floor((e['z']+512)/3),math.floor((e['x']+512)/3)])
trials=[];routes=[]
for side in (1,2):
    for ford in range(3):
        ps=[e for e in es if e['event']=='ford_position' and e['side']==side and e['ford']==ford]
        # Teleport is applied on the following engine tick. Remove stale prior-trial points.
        start=next(i for i,e in enumerate(ps) if label(e)==0)
        ps=ps[start:];water=next(i for i,e in enumerate(ps) if e['ground']=='shallow_water')
        crossed=next((e for e in ps[water+1:] if label(e)==1 and e['ground'] not in ('shallow_water','deep_water')),None)
        end=next(e for e in es if e['event']=='ford_trial_done' and e['side']==side and e['ford']==ford)
        assert crossed is not None
        trials.append({'side':side,'ford':ford,'started_on_bank':0,'ended_on_bank':label(ps[-1]),'shallow_water_observed':True,'reached_other_dry_bank':True,'first_other_dry_bank_ms':crossed['elapsed_ms'],'original_8m_goal_result':end,'stale_initial_samples_removed':start})
        routes.append((side,ford,ps))
out=HERE/'crossings';out.mkdir(exist_ok=True)
result={'scope':'Unit-centre trajectories, two complete units, three crossings. No per-soldier trajectories or capacity measurement.','run':run.name,'trials':trials,'caveats':['Teleport applied asynchronously; stale samples removed by first observed source-bank point.','Cavalry crossed and stopped about 12 m from the order coordinate; original 8 m arrival criterion timed out. Do not relabel it as reached.','Crossing is independently confirmed by source bank -> shallow_water -> opposite dry bank sequence.','No inference of exact ford width or universal unit clearance.']}
(out/'summary.json').write_text(json.dumps(result,indent=2))
water=np.load(HERE/'run-20260925-232109/surface-masks.npz')['water']
for lang in ['en','ru']:
    fig,ax=plt.subplots(figsize=(10,10),layout='constrained')
    ax.imshow(water,origin='lower',extent=(-512,514,-512,514),interpolation='nearest',cmap=ListedColormap(['#bbbbbb','#efda85','#62c7ec','#174a92']),norm=BoundaryNorm([-1.5,-.5,.5,1.5,2.5],4))
    for side,ford,ps in routes:
        ax.plot([p['x'] for p in ps],[p['z'] for p in ps],color='#e63031' if side==1 else '#181e27',lw=2,alpha=.85,label=((['Cavalry','Infantry'] if lang=='en' else ['Кавалерия','Пехота'])[side-1]) if ford==0 else None)
    for r in candidates:
        c=r['world_center'];ax.annotate(str(r['id']+1),(c['x'],c['z']),xytext=(20,20),textcoords='offset points',fontsize=16,weight='bold',bbox={'facecolor':'white','alpha':.9},arrowprops={'arrowstyle':'->'})
    ax.set(xlim=(-512,512),ylim=(-512,512),xlabel='X, m',ylabel='Z, m',title='Three ford crossings verified by movement' if lang=='en' else 'Три брода: проход проверен движением отрядов')
    ax.legend(loc='upper left');fig.savefig(out/f'crossings.{lang}.png',dpi=150,bbox_inches='tight');plt.close(fig)
print('Six bank-to-bank crossings verified; original cavalry arrival timeouts preserved.')

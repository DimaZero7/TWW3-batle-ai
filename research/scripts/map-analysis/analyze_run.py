from pathlib import Path
import sys,csv,json,collections,hashlib,gzip
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'.tools/plotting'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch
run=Path(sys.argv[1]);out=HERE/run.name;out.mkdir(exist_ok=True)
events=[json.loads(s) for s in (run/'tww3_bai_map_capture_events.jsonl').read_text().splitlines()]
frame=next(e for e in events if e['event']=='frame');grid=next(e for e in events if e['event']=='grid_begin')
shape=(grid['rows'],grid['columns']);water=np.full(shape,-1,dtype=np.int8);blocked=np.full(shape,-1,dtype=np.int8);height=np.zeros(shape);counts=collections.Counter()
grounds=np.empty(shape,dtype=object)
with (run/'tww3_bai_map_capture_grid.csv').open() as f:
    for r in csv.DictReader(f):
        i,j=int(r['iz']),int(r['ix']);height[i,j]=float(r['height']);grounds[i,j]=r['ground']
        if int(r['inside_radar']):
            counts[r['ground']]+=1;water[i,j]={'shallow_water':1,'deep_water':2}.get(r['ground'],0);blocked[i,j]=1-int(r['clear'])
np.savez_compressed(out/'surface-masks.npz',water=water,restricted=blocked,height=height)
buildings=[e for e in events if e['event']=='building'];reach=[e for e in events if e['event']=='reach_sample']
inside=[b for b in buildings if frame['min_x']<=b['x']<frame['max_x'] and frame['min_z']<=b['z']<frame['max_z']]
db=json.loads((HERE/'battlefield_buildings_tables.json').read_text())[0];byname={b['key']:b for b in db['rows']};matches=[byname[n] for n in sorted({b['name'] for b in buildings}) if n in byname]
ab=json.loads((HERE/'battlefield_buildings_to_special_abilities_tables.json').read_text())[0]
effects=[r for r in ab['rows'] if r['battlefield_building'] in {b['name'] for b in buildings}]
(out/'buildings.json').write_text(json.dumps(buildings,indent=2));(out/'reach-samples.json').write_text(json.dumps(reach,indent=2))
(out/'matching-db-records.json').write_text(json.dumps({'building_table_source':db['source'],'building_table_sha256':db['sha256'],'roundtrip':db['roundtrip'],'rows':matches,'ability_table_source':ab['source'],'ability_table_sha256':ab['sha256'],'ability_matches':effects},indent=2))
summary={'run':run.name,'frame':frame,'grid':grid,'ground_inside_counts':dict(counts),'building_count':len(buildings),'building_pivots_inside_count':len(inside),'native_categories':dict(collections.Counter(b['category'] for b in buildings)),'db_matched_unique_names':len(matches),'db_unmatched_names':sorted({b['name'] for b in buildings}-byname.keys()),'ability_matches':effects,'reach_sample_count':len(reach),'reach_counts':[{'ground':k[0],'cavalry':k[1],'infantry':k[2],'count':v} for k,v in collections.Counter((r['ground'],r.get('reach_cavalry'),r.get('reach_infantry')) for r in reach).items()],'context':[e for e in events if e['event'] in ('context_summary','context_variants','unit_origin','probe_done','grid_done','navigation_done')],'status':json.loads((run/'status.json').read_text(encoding='utf-8-sig'))}
(out/'summary.json').write_text(json.dumps(summary,indent=2))
ex=(frame['min_x'],grid['query_max_x'],frame['min_z'],grid['query_max_z'])
for lang in ['en','ru']:
    fig,axs=plt.subplots(1,2,figsize=(15,8),layout='constrained')
    watercols=['#aaaeb3','#efda85','#62c7ec','#174a92'];restcols=['#aaaeb3','#7caf70','#b84f45']
    axs[0].imshow(water,origin='lower',extent=ex,interpolation='nearest',cmap=ListedColormap(watercols),norm=BoundaryNorm([-1.5,-.5,.5,1.5,2.5],4))
    axs[1].imshow(blocked,origin='lower',extent=ex,interpolation='nearest',cmap=ListedColormap(restcols),norm=BoundaryNorm([-1.5,-.5,.5,1.5],3))
    axs[1].scatter([b['x'] for b in inside],[b['z'] for b in inside],s=5,c='#172132',alpha=.65)
    titles=['Water classes from Lua','Area restriction + object origins'] if lang=='en' else ['Типы воды из Lua','Ограничения области и точки объектов']
    waterlabels=['Unknown edge','Other ground','Shallow water','Deep water'] if lang=='en' else ['Край без замера','Другая земля','Мелководье','Глубокая вода']
    restlabels=['Unknown edge','Area query: clear','Area query: restricted'] if lang=='en' else ['Край без замера','Запрос области: свободно','Запрос области: ограничение']
    for ax,title in zip(axs,titles):ax.set(title=title,xlabel='X, m',ylabel='Z, m',xlim=(frame['min_x'],frame['max_x']),ylim=(frame['min_z'],frame['max_z']))
    axs[0].legend(handles=[Patch(color=c,label=l) for c,l in zip(watercols,waterlabels)],loc='upper center',bbox_to_anchor=(.5,-.09),ncol=2,fontsize=9)
    axs[1].legend(handles=[Patch(color=c,label=l) for c,l in zip(restcols,restlabels)],loc='upper center',bbox_to_anchor=(.5,-.09),ncol=1,fontsize=9)
    fig.suptitle(('3 m cells; water class is not depth; dots are not collision shapes.' if lang=='en' else 'Клетка 3 м; тип воды не равен глубине; точки не описывают форму препятствий.'),fontsize=14)
    fig.savefig(out/f'water-obstacles.{lang}.png',dpi=150,bbox_inches='tight');plt.close(fig)
print(json.dumps({k:summary[k] for k in ('run','ground_inside_counts','building_count','building_pivots_inside_count','ability_matches','reach_sample_count')},indent=2))

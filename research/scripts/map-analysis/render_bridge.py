from pathlib import Path
import sys,csv,json
H=Path('tmp/map-research/field-features');sys.path.insert(0,'.tools/plotting')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch
R=Path('build/map-capture/run-20260925-234152');Q=Path('build/map-capture/run-20260925-234830');O=H/Q.name
rows=list(csv.DictReader((R/'tww3_bai_map_capture_grid.csv').open(encoding='utf-8')));water=np.full((342,342),-1);reach=water.copy()
for r in rows:
 i,j=int(r['iz']),int(r['ix'])
 if r['inside_radar']=='1':water[i,j]={'shallow_water':1,'deep_water':2}.get(r['ground'],0);reach[i,j]=int(r['reach_cavalry'])
es=[json.loads(l) for l in (Q/'tww3_bai_map_capture_events.jsonl').read_text(encoding='utf-8').splitlines()]
for lang in ['en','ru']:
 fig,axs=plt.subplots(1,2,figsize=(13,6),layout='constrained')
 colors=['#aaaaaa','#f1dc86','#6bc6eb','#194c91'];labels=['Unknown','Other ground','Shallow','Deep'] if lang=='en' else ['Нет замера','Другая земля','Мелководье','Глубокая вода']
 axs[0].imshow(water,origin='lower',extent=(-512,514,-512,514),interpolation='nearest',cmap=ListedColormap(colors),norm=BoundaryNorm([-1.5,-.5,.5,1.5,2.5],4))
 axs[1].imshow(reach,origin='lower',extent=(-512,514,-512,514),interpolation='nearest',cmap=ListedColormap(['#aaaaaa','#6d737b','#88b884']),norm=BoundaryNorm([-1.5,-.5,.5,1.5],3))
 for ax in axs:
  ax.set(xlim=(-430,-120),ylim=(-360,-200),xlabel='X, m',ylabel='Z, m');ax.scatter([-274.438],[-276.252],s=90,marker='D',c='black',label='Bridge origin' if lang=='en' else 'Точка объекта моста')
  for side,color in [(1,'#e82d32'),(2,'#59299e')]:
   ps=[e for e in es if e['event']=='bridge_position' and e['side']==side and e['stage']==1]
   ax.plot([e['x'] for e in ps],[e['z'] for e in ps],lw=3,c=color,label=(['Cavalry','Infantry'] if lang=='en' else ['Кавалерия','Пехота'])[side-1])
  ax.legend(loc='lower right',fontsize=8)
 axs[0].set_title('Ground classes, 3 m' if lang=='en' else 'Тип поверхности, шаг 3 м');axs[1].set_title('Point query: green = true' if lang=='en' else 'Доступность точки: зелёный = да')
 fig.suptitle('Bridge detected; traversal was not demonstrated' if lang=='en' else 'Мост распознан; проход не подтверждён',fontsize=15)
 fig.legend(handles=[Patch(color=c,label=l) for c,l in zip(colors[1:],labels[1:])],loc='outside lower center',ncol=3,fontsize=9)
 fig.savefig(O/f'bridge-check.{lang}.png',dpi=150,bbox_inches='tight');plt.close(fig)

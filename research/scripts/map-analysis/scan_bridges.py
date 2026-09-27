from pathlib import Path
import json,struct,time
from common import index,extract,GAME,HERE
db=json.loads((HERE/'battlefield_buildings_tables.json').read_text())[0]['rows']
keys=[r['key'] for r in db if r['category']=='bridge'];patterns={k:struct.pack('<H',len(k))+k.encode() for k in keys}
start=time.monotonic();hits=[];files=0;bytes_read=0
for pack in sorted(GAME.glob('tile*.pack')):
    entries=index(pack)
    for e in entries:
        if not e['path'].endswith('bmd_data.bin'):continue
        # Terrain records only; no settlement maps are launched by this static search.
        raw=extract(e);files+=1;bytes_read+=len(raw)
        found=[k for k,p in patterns.items() if p in raw]
        if found:hits.append({'source':e,'keys':found});print('HIT',e['path'],found,flush=True)
    if time.monotonic()-start>180:print('Stopped bounded scan at 180 seconds');break
(HERE/'bridge-category-hits.json').write_text(json.dumps({'files':files,'decompressed_bytes':bytes_read,'elapsed_s':time.monotonic()-start,'bridge_keys':keys,'hits':hits},indent=2))
print('DONE',files,'files',len(hits),'hits',round(time.monotonic()-start,1),'seconds')

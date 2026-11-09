import csv, gzip, collections, sys, math
import numpy as np
meta = {}
for row in csv.reader(open('data/battleterrain_vegetation_trees_tree_metadata.txt', encoding='utf-8', errors='replace'), delimiter='\t'):
    if len(row) > 5 and row[0].lower().endswith('.rigid_model_v2'):
        try: meta[row[0].lower()] = (float(row[1]), float(row[3]), float(row[4]), float(row[5]))
        except ValueError: pass
pts = list(csv.DictReader(gzip.open(sys.argv[1], 'rt')))
miss = collections.Counter(); rs = []; hs = []; sc = []
per = collections.defaultdict(list)
for p in pts:
    m = meta.get(p['model'].lower())
    if m is None: miss[p['model']] += 1; continue
    s = float(p['scale']); rs.append(m[3]); hs.append(m[2]); sc.append(s)
    per[p['model'].split('/')[-1]].append((m[3], m[2], s))
print('points', len(pts), 'with meta', len(rs), 'missing', sum(miss.values()), miss.most_common(5))
for k, v in sorted(per.items(), key=lambda kv: -len(kv[1]))[:16]:
    a = np.array(v); print(f'{len(v):5d} {k:45s} R={a[0,0]} H={a[0,1]} scale {a[:,2].min():.2f}-{a[:,2].max():.2f} mean {a[:,2].mean():.2f}')
r = np.array(rs); s = np.array(sc)
print('R mean', r.mean(), 'median', np.median(r), 'R*scale mean', (r*s).mean())
print('area sum m2 (pi R^2)', (math.pi*r**2).sum(), ' (R*scale)', (math.pi*(r*s)**2).sum())

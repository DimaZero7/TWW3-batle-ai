import csv, gzip, math, sys
import numpy as np
meta = {}
for row in csv.reader(open('data/battleterrain_vegetation_trees_tree_metadata.txt', encoding='utf-8', errors='replace'), delimiter='\t'):
    if len(row) > 5 and row[0].lower().endswith('.rigid_model_v2'):
        try: meta[row[0].lower()] = (float(row[4]), float(row[5]))   # height, radius
        except ValueError: pass
E = r"C:/Users/dima/Desktop/Файлы/Программы/TWW3-batle-ai/research/evidence/map/forest-trees-20260925/"
pts = list(csv.DictReader(gzip.open(E + 'tree-points.csv.gz', 'rt')))
F = np.load(E + 'forest-3m.npy')   # [iz, ix], 3 m, -512..
R = []; H = []; X = []; Z = []
for p in pts:
    m = meta.get(p['model'].lower())
    if not m: continue
    s = float(p['scale']); r = (m[1] if m[1] != 0 else 0.85) * s
    R.append(min(r, 2.25)); H.append(m[0] * s); X.append(float(p['x'])); Z.append(float(p['z']))
R = np.array(R); H = np.array(H); X = np.array(X); Z = np.array(Z)
ix = np.clip(((X + 512) // 3).astype(int), 0, F.shape[1]-1); iz = np.clip(((Z + 512) // 3).astype(int), 0, F.shape[0]-1)
inF = F[iz, ix] == 1
fa = (F == 1).sum() * 9.0
print('colliders', len(R), 'in forest', inF.sum(), 'forest area m2', fa)
print('radius: mean %.2f median %.2f p10 %.2f p90 %.2f max %.2f' % (R.mean(), np.median(R), *np.percentile(R, [10, 90]), R.max()))
print('height: median %.1f, share < 2 m %.2f, < 0.3 m %d' % (np.median(H), (H < 2).mean(), (H < 0.3).sum()))
d = inF.sum() / fa
print('density in forest: %.4f per m2 -> one tree per %.0f m2, mean spacing %.1f m' % (d, 1/d, 1/math.sqrt(d)))
print('trunk area share in forest: %.3f' % ((math.pi * R[inF]**2).sum() / fa))
tall = inF & (H > 2.0)
for rp in (0.05, 0.3):
    lam = (2 * (R[tall] + rp)).sum() / fa
    print('proj radius %.2f: interception per m of path below crowns %.4f -> mean free path %.0f m; P(hit) over 10/20/40 m: %.2f %.2f %.2f' % (rp, lam, 1/lam, 1-math.exp(-10*lam), 1-math.exp(-20*lam), 1-math.exp(-40*lam)))
for rs in (0.5,):
    lam = (2 * (R[inF] + rs)).sum() / fa
    print('soldier radius %.1f: trunks met per m walked %.4f (one per %.0f m)' % (rs, lam, 1/lam))
# formation slots: share of 1 m spaced slots lying inside a trunk + soldier radius
A = (math.pi * (R[inF] + 0.4)**2).sum() / fa
print('share of formation places blocked (trunk + 0.4 m body): %.3f' % A)
# per 3 m cell occupancy
occ = np.zeros_like(F, dtype=float)
np.add.at(occ, (iz[inF], ix[inF]), math.pi * R[inF]**2)
sh = occ[F == 1] / 9.0
print('per 3 m forest cell trunk share: mean %.3f, p90 %.3f, max %.3f, cells with a trunk %.2f' % (sh.mean(), np.percentile(sh, 90), sh.max(), (sh > 0).mean()))

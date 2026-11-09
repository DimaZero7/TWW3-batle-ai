# predicted share of arrows stopped by trunks on the probe lanes (Kislev), from tree list + tree_metadata
import csv, gzip, math
import numpy as np
meta = {}
for row in csv.reader(open('data/battleterrain_vegetation_trees_tree_metadata.txt', encoding='utf-8', errors='replace'), delimiter='\t'):
    if len(row) > 5 and row[0].lower().endswith('.rigid_model_v2'):
        try: meta[row[0].lower()] = (float(row[4]), float(row[5]))
        except ValueError: pass
E = r"C:/Users/dima/Desktop/Файлы/Программы/TWW3-batle-ai/"
pts = list(csv.DictReader(gzip.open(E + 'research/evidence/map/forest-trees-20260925/tree-points.csv.gz', 'rt')))
T = []
for p in pts:
    m = meta.get(p['model'].lower())
    if not m: continue
    s = float(p['scale']); r = min((m[1] or 0.85) * s, 2.25)
    T.append((float(p['x']), float(p['y']), float(p['z']), r, m[0] * s))
T = np.array(T)
print('tree x range', T[:,0].min(), T[:,0].max(), 'y', T[:,1].min(), T[:,1].max())
d = np.load(E + 'build/maps/ksl_plains_01/heights.npz'); H = d['height']; x0 = float(d['x0']); z0 = float(d['z0']); st = float(d['step'])
def h(x, z):
    i = np.clip(np.round((x - x0) / st).astype(int), 0, H.shape[1]-1); j = np.clip(np.round((z - z0) / st).astype(int), 0, H.shape[0]-1)
    return H[j, i]
g = 9.8; v = 45.0; vit = 0.93; rp = 0.1
lanes = {"strip_57": (160.6, -366.1, 64, -392), "strip_7": (8.0, 324.0, 8, 224), "strip_11": (336.0, 380.0, 336, 480),
         "grass_1": (-217.4, -2.0, -304, 48), "grass_2": (-140.0, -224.0, -240, -224), "grass_3": (-354.0, -198.6, -304, -112)}
rng = np.random.default_rng(1)
for name, (sx, sz, tx, tz) in lanes.items():
    dx, dz = tx - sx, tz - sz; L = math.hypot(dx, dz); ux, uz = dx / L, dz / L; nx, nz = uz, -ux
    stopped = 0; N = 400; skipped_m = []
    for k in range(N):
        a = rng.uniform(-14, 14); b = a + rng.uniform(-6, 6)          # shooter / target lateral offsets
        ax, az = sx + nx * a + ux * rng.uniform(-3, 3), sz + nz * a
        bx, bz = tx + nx * b + ux * rng.uniform(-3, 3), tz + nz * b
        D = math.hypot(bx - ax, bz - az); y0 = h(np.array([ax]), np.array([az]))[0] + 1.7; y1 = h(np.array([bx]), np.array([bz]))[0] + 0.75 * 1.6
        dy = y1 - y0
        disc = v**4 - g * (g * D * D + 2 * dy * v * v)
        if disc < 0: continue
        th = math.atan((v * v - math.sqrt(disc)) / (g * D)); vh = v * math.cos(th); tf = D / vh
        ts = np.linspace(0, tf, 300)
        px = ax + (bx - ax) * ts / tf; pz = az + (bz - az) * ts / tf; py = y0 + v * math.sin(th) * ts - 0.5 * g * ts**2
        live = ts > vit
        # trees near the path
        hit = False
        for (x, y, z, r, ht) in T[(np.abs((T[:,0]-ax)*nx + (T[:,2]-az)*nz) < 20) & (((T[:,0]-ax)*ux + (T[:,2]-az)*uz) > -5) & (((T[:,0]-ax)*ux + (T[:,2]-az)*uz) < D + 5)]:
            dd = np.hypot(px - x, pz - z)
            m = live & (dd < r + rp) & (py >= y - 0.5) & (py <= y + ht)
            if m.any(): hit = True; break
        stopped += hit
    print(f'{name}: L={L:.0f} m, predicted stopped {stopped / N:.2f}')

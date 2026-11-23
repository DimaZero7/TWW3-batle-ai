"""S0-1's ground (tmp/drill_s1/design_merged.md section 4, design_opus section 5 'Ground'): the captured maps as they are
and synthetic Gaussian ridges - pools of maps written once per pool entry into build/s1_synth/<name>/ (out of git,
deterministic from the name) in sim/maps' own format, and registered at run time through what the simulator offers:
sim.terrain.load_npz (its registry) under the name in net/matchup/duel's map cache - sim/** is not touched.

    POOLS                   {pool: (seed0, n entries)} - one pool per split (whole maps in one split: a ridge never
                            seen in training reaches V, the probe or the exam)
    entry(pool, k)          the k-th map's recipe {base, phi, cx, cz, sigma, length, slope, low, shift_m}
    name(pool, k, variant)  'synth:<pool>:<k>:<variant>', variant ridge | none | shift | low
    folder(name)            the map's folder (written on first use)
    ground(name)            crest.py's grids of any map name ('flat', a captured map, a synthetic one)
    register(names)         every synthetic name in net/matchup/duel's cache (sim.terrain.load_npz), before a build
A ridge: h = A exp(-u^2 / 2 sigma^2) w(v) on the base, u along its facing phi (toward the enemy), v along the crest,
w a flat-topped window of the length with Gaussian ends of END_M; A = slope x sigma / exp(-1/2) (the Gaussian's peak
slope is A / sigma x 0.607); cannot stand where the slope is >= CANNOT_STAND (design_opus 5: the captured maps' 50 %
point, tmp/drill_s1/opus/slope_blocked.txt), or where the captured base says so. Grid STEP_M 2 m (513 x 513 over the
captured maps' -512..512): a pool of 4 variants x 24 entries is ~300 MB of grids, not 1.2 GB at 1 m.
"""
import json
import math
import random
from functools import lru_cache

import numpy as np

from net import ROOT

STEP_M = 2.0
SIZE = 513
X0 = -512.0
CANNOT_STAND = 0.5
END_M = 30.0
BASES = (("flat", 0.5), ("ksl_plains_01", 0.5))
_BASE_WHY = ("design_merged 4: half on a flat base, half on a captured one; the captured base is the plains (the river is "
             "61-80 % water inside its zones: a ridge there is mostly illegal); kho_wastes_01 is held ground, never a base")
RIDGE = {"sigma": (25.0, 70.0), "length": (200.0, 600.0), "slope": (0.15, 0.45), "back_m": 180.0, "jitter_m": 40.0,
         "low_band": (0.02, 0.045), "shift_m": 200.0}
RIDGE_HELD_STEEP = {"slope": (0.45, 0.58), "sigma": (25.0, 40.0)}
_RIDGE_WHY = ("design_opus 5: sigma 25-70 m, length 200-600 m, peak slope <= 0.45 (the captured maps' own range); the "
              "ridge back_m behind the map's middle along its facing so the zone pair (fronts 192 m apart) fits the map; "
              "low twins Delta band 0.02-0.045 (the generator's 'low' twin, design_merged 4); shift twins: the ridge 200 m "
              "forward - a crest within the zone's depth (+-88 m) lands 112-288 m ahead of the zone's middle, beyond its "
              "front (ASSUMPTION for design_opus's '100-150 m beyond the front'); the held 'crest Delta band 0.30-0.42' "
              "axis needs a steeper ridge (pool held_band: slope 0.45-0.58, sigma 25-40, the cannot-stand part illegal)")
POOLS = {"train": (910_000, 24), "dev": (920_000, 8), "V": (930_000, 8), "probe_train": (940_000, 16),
         "probe_held": (950_000, 12), "taught": (960_000, 16), "held": (970_000, 12), "held_band": (975_000, 12),
         "G": (980_000, 1)}
_POOLS_WHY = ("one pool a split (design_merged 4, from Astra: whole families in one split - here whole maps); the seeds "
              "are Python streams of this module only (generator 's1.ground', registry entries in tmp/drill_s1/c/NEEDS_A.md)")
VARIANTS = ("ridge", "none", "shift", "low")


def _captured(name):
    from net.strategist import crest as CR
    return CR.ground(name)


def entry(pool, k):
    s0, n = POOLS[pool]
    assert 0 <= k < n, (pool, k)
    rnd = random.Random(s0 + k)
    base = rnd.choices([b for b, _ in BASES], weights=[w for _, w in BASES])[0]
    R = dict(RIDGE)
    if pool == "held_band":
        R.update(RIDGE_HELD_STEEP)
    phi = rnd.uniform(0.0, 360.0)
    a = math.radians(phi)
    back = R["back_m"]
    cx = -back * math.sin(a) + rnd.uniform(-R["jitter_m"], R["jitter_m"])
    cz = -back * math.cos(a) + rnd.uniform(-R["jitter_m"], R["jitter_m"])
    return {"pool": pool, "k": k, "base": base, "phi": phi, "cx": cx, "cz": cz, "sigma": rnd.uniform(*R["sigma"]),
            "length": rnd.uniform(*R["length"]), "slope": rnd.uniform(*R["slope"]),
            "low_band": rnd.uniform(*R["low_band"]), "shift_m": R["shift_m"]}


def name(pool, k, variant="ridge"):
    assert variant in VARIANTS, variant
    return f"synth:{pool}:{k}:{variant}"


def parse(nm):
    _, pool, k, variant = nm.split(":")
    return pool, int(k), variant


def is_synth(nm):
    return isinstance(nm, str) and nm.startswith("synth:")


def _ridge(e, variant):
    """[SIZE, SIZE] the ridge's heights (0 for 'none')."""
    if variant == "none":
        return np.zeros((SIZE, SIZE))
    xs = X0 + STEP_M * np.arange(SIZE)
    X, Z = np.meshgrid(xs, xs)                                 # (row = z, column = x)
    a = math.radians(e["phi"])
    fx, fz = math.sin(a), math.cos(a)
    shift = e["shift_m"] if variant == "shift" else 0.0
    dx, dz = X - (e["cx"] + shift * fx), Z - (e["cz"] + shift * fz)
    u = dx * fx + dz * fz
    v = dx * fz - dz * fx
    half = e["length"] / 2
    w = np.exp(-0.5 * (np.clip(np.abs(v) - half, 0, None) / END_M) ** 2)
    A = e["slope"] * e["sigma"] / math.exp(-0.5)
    if variant == "low":
        A *= e["low_band"] / max(0.6 * e["slope"], 1e-6)       # (band ~ 0.6 x the peak slope: Delta band ~ low_band)
    return A * np.exp(-0.5 * (u / e["sigma"]) ** 2) * w


def recipe_sha(nm):
    """The sha of everything a synthetic map's grids are made of (its entry, variant and this module's constants): its
    folder is keyed by it, so a changed recipe never reads an old grid (Astra s1_step2_review 7)."""
    import hashlib
    pool, k, variant = parse(nm)
    r = {"entry": entry(pool, k), "variant": variant, "STEP_M": STEP_M, "SIZE": SIZE, "X0": X0,
         "CANNOT_STAND": CANNOT_STAND, "END_M": END_M, "v": 2}
    return hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()[:10]


def _write(nm):
    pool, k, variant = parse(nm)
    e = entry(pool, k)
    d = ROOT / "build" / "s1_synth" / f"{nm.replace(':', '_')}__{recipe_sha(nm)}"
    if (d / "meta.json").exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    if e["base"] == "flat":
        h = np.full((SIZE, SIZE), 200.0)
        g = np.zeros((SIZE, SIZE), np.uint8)
        names = ["grass"]
        blk = np.zeros((SIZE, SIZE), np.uint8)
    else:
        B = _captured(e["base"])
        h = B["h"][::2, ::2][:SIZE, :SIZE].copy()
        g = B["g"][::2, ::2][:SIZE, :SIZE].copy()
        b1 = B["blk"]
        pad = np.pad(b1, ((0, 1), (0, 1)), mode="edge")
        blk = np.maximum.reduce([pad[0::2, 0::2], pad[1::2, 0::2], pad[0::2, 1::2], pad[1::2, 1::2]])[:SIZE, :SIZE]
        meta = json.loads((ROOT / "sim" / "maps" / e["base"] / "meta.json").read_text(encoding="utf-8"))
        names = meta["ground_names"]
    h = h + _ridge(e, variant)
    gz, gx = np.gradient(h, STEP_M)
    slope = np.hypot(gx, gz)
    blk = np.maximum(blk, (slope >= CANNOT_STAND).astype(np.uint8))
    np.savez(d / "heights.npz", height=h.astype(np.float32), clear=np.ones_like(g, dtype=np.uint8),
             ground=g.astype(np.uint8), x0=X0, z0=X0, step=STEP_M)
    np.savez(d / "passable.npz", blocked=blk.astype(np.uint8), x0=X0, z0=X0, step=STEP_M)
    (d / "meta.json").write_text(json.dumps({"name": nm, "ground_names": names, "entry": e, "variant": variant,
                                             "frame": {"min_x": -512, "max_x": 512, "min_z": -512, "max_z": 512},
                                             "note": "S0-1 synthetic ridge (net/strategist/s1_ground.py)"}, indent=1),
                                 encoding="utf-8")
    return d


def folder(nm):
    return _write(nm)


@lru_cache(maxsize=None)
def _flat():
    from net.strategist import crest as CR
    return CR.ground_of(np.zeros((3, 3)), -1e5, -1e5, step=1e5)


@lru_cache(maxsize=256)
def ground(nm):
    """crest.py's grids of a map name."""
    from net.strategist import crest as CR
    if nm in (None, "flat"):
        return _flat()
    if not is_synth(nm):
        return CR.ground(nm)
    d = folder(nm)
    H = np.load(d / "heights.npz")
    P = np.load(d / "passable.npz")
    names = json.loads((d / "meta.json").read_text(encoding="utf-8"))["ground_names"]
    code = {n: i for i, n in enumerate(names)}
    return CR.ground_of(H["height"], float(H["x0"]), float(H["z0"]), H["ground"], P["blocked"],
                        [code[n] for n in ("forest",) if n in code],
                        [code[n] for n in ("shallow_water", "deep_water") if n in code], float(H["step"]))


def register(names):
    """Every synthetic map of names into net/matchup/duel's map cache (sim.terrain.load_npz with its layers) - the
    simulator's run-time map registry; duel.build then finds it by name (torch side only)."""
    from net.matchup import duel as DU
    from sim import terrain
    for nm in dict.fromkeys(n for n in names if is_synth(n)):
        if nm not in DU._MAPS:
            DU._MAPS[nm] = terrain.load_npz(folder(nm) / "heights.npz")
    return DU._MAPS

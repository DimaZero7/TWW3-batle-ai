"""A real map's height grid for the network (build/maps/<name>/): one capture a map through the map-capture target
(src/entries/map_capture.lua: bm:get_terrain_height over the radar frame, the multiples of the step - with step 1 or 2
the engine's own 2 m nodes are among them - plus clear / ground type), then heights.npz (float32 [rows, columns], row = z,
column = x, origin x0 / z0, step), meta.json (map, frame, deployment zones of the capture scenario, source run, date,
timing) and preview.png (dark lows, light hills, the zones outlined). One more layer on the same grid (build/diag_map/
scan.md): passable.npz - blocked uint8 1 = a unit cannot stand there (the node is not reachable for the capture's units
after deployment: unit:can_reach_position; outside that read - not bm:is_area_clear); passable.png shows it over the
water and forest ground types (heights.npz ground: the game's own 'in the water' / 'in the forest', 2 m cells).

    python -m tools.nn.map_grid capture --name ksl_plains_01 [--step 1]   # build, launch, convert (needs the game free)
    python -m tools.nn.map_grid convert build/map-capture/runs/<run> --name ksl_plains_01
    python -m tools.nn.map_grid layers build/map-capture/runs/<run> --name ksl_plains_01   # only passable
    python -m tools.nn.map_grid trees --name ksl_plains_01 --points <tree-points.csv.gz> --meta <tree_metadata.txt>

trees.npz (the trunks: x, z, base y, radius r, height h - float32; tree_bodies) comes from the map's tree list (the map
file's *.tree_list.bin, read once: research/evidence/map/forest-trees-20260925/tree-points.csv.gz for Kislev) and
vegetation.pack's tree_metadata.txt; the game gives no tree to scripts (build/diag_g49/terrain/trees.md 3).

The scenario (scenarios/map_capture.xml) decides the map; the zones are that scenario's deployment areas, not the map's
own (the game gives a map's zones to a battle only through the battle's XML)."""
import argparse
import csv
import datetime
import io
import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from tools import config as project

RUNS = project.ROOT / "build" / "map-capture" / "runs"
MAPS = project.ROOT / "build" / "maps"
SCENARIO = project.ROOT / "scenarios" / "map_capture.xml"
LOCK = project.ROOT / "build" / "steps" / "probe_game.lock"
HUMAN = project.ROOT / "build" / "steps" / "HUMAN_PLAYING"


def parse_grid(text):
    """The capture's grid.csv -> (heights float32 [rows, columns], clear uint8, ground codes uint8, ground names, x0, z0,
    step). Every cell must be there once; a missing or non-finite height is an error."""
    rows = list(csv.DictReader(io.StringIO(text)))
    assert rows, "empty grid"
    ix = np.array([int(r["ix"]) for r in rows])
    iz = np.array([int(r["iz"]) for r in rows])
    x = np.array([float(r["x"]) for r in rows])
    z = np.array([float(r["z"]) for r in rows])
    h = np.array([float(r["height"]) for r in rows])
    assert np.isfinite(h).all(), "non-finite height"
    nx, nz = ix.max() + 1, iz.max() + 1
    assert len(rows) == nx * nz and len(set(zip(ix.tolist(), iz.tolist()))) == len(rows), "missing or repeated cells"
    step = float(np.median(np.diff(np.unique(x)))) if nx > 1 else float(np.median(np.diff(np.unique(z))))
    x0, z0 = float(x[(ix == 0)][0]), float(z[(iz == 0)][0])
    assert np.allclose(x, x0 + ix * step, atol=1e-3) and np.allclose(z, z0 + iz * step, atol=1e-3), "irregular grid"
    names = sorted({r["ground"] for r in rows})
    code = {n: i for i, n in enumerate(names)}
    H = np.zeros((nz, nx), np.float32)
    C = np.zeros((nz, nx), np.uint8)
    G = np.zeros((nz, nx), np.uint8)
    H[iz, ix] = h
    C[iz, ix] = [int(r["clear"]) for r in rows]
    G[iz, ix] = [code[r["ground"]] for r in rows]
    return H, C, G, names, x0, z0, step


def parse_reach(text, shape):
    """The capture's reach columns (map-capture --features: unit:can_reach_position of each side's first unit after
    deployment) -> (reach side 1, reach side 2) int8 [rows, columns]: 1 yes, 0 no, -1 not read (outside the radar);
    None when the grid has no reach columns."""
    reader = csv.DictReader(io.StringIO(text))
    if "reach_side_1" not in (reader.fieldnames or []):
        return None
    R = np.full((2,) + tuple(shape), -1, np.int8)
    for r in reader:
        iz, ix = int(r["iz"]), int(r["ix"])
        R[0, iz, ix] = int(r["reach_side_1"])
        R[1, iz, ix] = int(r["reach_side_2"])
    return R[0], R[1]


def passable_layers(C, G, names, reach=None):
    """-> blocked uint8 [rows, columns] (1 = a unit cannot stand there) and counts. Where the reach was read: reachable
    for none of the captured units (the game's point query: it agrees with is_area_clear but for the 1-node ring of a
    blocked area, where the 1 x 1 m box touches it and the point itself is free); elsewhere: not clear
    (bm:is_area_clear of the node's 1 x 1 m box). Water and forest are not layers of their own: the ground types
    (heights.npz ground) already are them; info counts the blocked nodes by ground type."""
    blocked = C == 0
    info = {"cells": int(C.size)}
    if reach is not None:
        r1, r2 = reach
        known = (r1 >= 0) & (r2 >= 0)
        no = known & (r1 == 0) & (r2 == 0)
        blocked = np.where(known, no, blocked)
        info.update(reach_read=int(known.sum()), reach_differs=int((known & (r1 != r2)).sum()),
                    clear_not_reach=int((known & (C == 1) & no).sum()),
                    not_clear_reach=int((known & (C == 0) & ~no).sum()),
                    source="unit:can_reach_position (both units), bm:is_area_clear outside the read")
    else:
        info["source"] = "bm:is_area_clear (no reach read)"
    info.update(blocked=int(blocked.sum()), blocked_share=round(float(blocked.mean()), 4),
                blocked_by_ground={n: [int((blocked & (G == i)).sum()), int((G == i).sum())]
                                   for i, n in enumerate(names)})
    return blocked.astype(np.uint8), info


def write_layers(dst, blocked, x0, z0, step, info, run=None):
    """passable.npz next to heights.npz (the same grid: x0, z0, step) and meta.json's 'passable' entry (the rest of
    meta.json kept)."""
    dst = Path(dst)
    np.savez_compressed(dst / "passable.npz", blocked=blocked, x0=np.float64(x0), z0=np.float64(z0),
                        step=np.float64(step))
    mp = dst / "meta.json"
    meta = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
    meta["passable"] = {"file": "passable.npz", "layout": "blocked[iz, ix] uint8, 1 = a unit cannot stand there",
                        **{k: info[k] for k in ("source", "blocked", "blocked_share", "cells", "reach_read",
                                                "reach_differs", "clear_not_reach", "not_clear_reach") if k in info},
                        "blocked_by_ground": {k: f"{v[0]} of {v[1]}" for k, v in info["blocked_by_ground"].items()}}
    if run is not None:
        meta["passable"]["run"] = str(run)
    mp.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    return meta


def preview_layers(path, blocked, G, names, x0, z0, step, title):
    """The picture: cannot stand black over the ground types forest / shallow / deep water."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cls = np.zeros(blocked.shape, np.uint8)
    for k, n in ((4, "forest"), (2, "shallow_water"), (3, "deep_water")):
        if n in names:
            cls[G == names.index(n)] = k
    cls[blocked > 0] = 1
    ext = (x0 - step / 2, x0 + (blocked.shape[1] - 0.5) * step, z0 - step / 2, z0 + (blocked.shape[0] - 0.5) * step)
    fig, ax = plt.subplots(figsize=(8, 7.2), dpi=120)
    ax.imshow(cls, origin="lower", extent=ext, interpolation="nearest", vmin=-0.5, vmax=4.5,
              cmap=ListedColormap(["#f4f4f4", "#202020", "#7fc4ff", "#1f4e9c", "#6fbf73"]))
    ax.set_xlabel("x, m")
    ax.set_ylabel("z, m")
    ax.set_title(title + "\nblack: cannot stand; ground: light blue shallow water, dark blue deep water, green forest",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def layers(run, name, out_root=MAPS):
    """Only passable.npz / passable.png (and meta.json's entry) for a map already converted: the run must be the same
    grid (shape, origin, step) with the same heights and clear layer; heights.npz is not touched."""
    run = Path(run)
    text = (run / "tww3_bai_map_capture_grid.csv").read_text(encoding="utf-8")
    H, C, G, names, x0, z0, step = parse_grid(text)
    dst = Path(out_root) / name
    have = np.load(dst / "heights.npz")
    assert have["height"].shape == H.shape and float(have["step"]) == step, "another grid"
    assert float(have["x0"]) == x0 and float(have["z0"]) == z0, "another origin"
    assert np.abs(have["height"] - H).max() < 1e-3 and (have["clear"] == C).all(), "another map or capture"
    blocked, info = passable_layers(C, G, names, parse_reach(text, H.shape))
    rel = run.resolve().relative_to(project.ROOT) if run.resolve().is_relative_to(project.ROOT) else run
    meta = write_layers(dst, blocked, x0, z0, step, info, rel)
    preview_layers(dst / "passable.png", blocked, G, names, x0, z0, step, f"{name}: cannot stand")
    return dst, meta


def zones(xml_text):
    """Deployment areas of a battle XML: [{alliance, x, z, width, depth, bearing_deg, corners [[x, z] x4]}]. The
    battle XML's y is the world z; orientation radians is the facing (0 = +z, clockwise)."""
    out = []
    root = ET.fromstring(xml_text)
    for k, al in enumerate(root.findall("alliance")):
        for dep in al.iter("deployment_area"):
            c, w, d, o = dep.find("centre"), dep.find("width"), dep.find("height"), dep.find("orientation")
            x, z = float(c.get("x")), float(c.get("y"))
            wm, dm = float(w.get("metres")), float(d.get("metres"))
            b = float(o.get("radians")) if o is not None else 0.0
            f = np.array([math.sin(b), math.cos(b)])
            s = np.array([math.cos(b), -math.sin(b)])
            corners = [[round(float(v), 2) for v in np.array([x, z]) + su * wm / 2 * s + fu * dm / 2 * f]
                       for su, fu in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            out.append({"alliance": int(al.get("id", k)), "x": x, "z": z, "width": wm, "depth": dm,
                        "bearing_deg": round(math.degrees(b) % 360, 2), "corners": corners})
    return out


def read_xml(path):
    """A battle XML: the game writes UTF-16 with a byte-order mark, our scenarios UTF-8."""
    raw = Path(path).read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig")


def map_of(xml_text):
    root = ET.fromstring(xml_text)
    d = root.find("battle_map_definition")
    return (d.findtext("name") or "").strip(), (d.findtext("catchment_area") or "").strip()


def events(run):
    out = {}
    path = Path(run) / "tww3_bai_map_capture_events.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out.setdefault(r.get("event"), []).append(r)
    return out


def preview(path, H, x0, z0, step, zone_list, frame, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ext = (x0 - step / 2, x0 + (H.shape[1] - 0.5) * step, z0 - step / 2, z0 + (H.shape[0] - 0.5) * step)
    fig, ax = plt.subplots(figsize=(8, 7.2), dpi=120)
    im = ax.imshow(H, origin="lower", extent=ext, cmap="gray", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="height, m (world Y)")
    for zn, col in zip(zone_list, ("tab:blue", "tab:red", "tab:green", "tab:orange")):
        cs = np.array(zn["corners"] + [zn["corners"][0]])
        ax.plot(cs[:, 0], cs[:, 1], color=col, lw=1.5, label=f"deployment, alliance {zn['alliance']}")
    if frame:
        ax.plot([frame["min_x"], frame["max_x"], frame["max_x"], frame["min_x"], frame["min_x"]],
                [frame["min_z"], frame["min_z"], frame["max_z"], frame["max_z"], frame["min_z"]], "y--", lw=1,
                label="radar frame")
    ax.set_xlabel("x, m")
    ax.set_ylabel("z, m")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def convert(run, name, out_root=MAPS):
    run = Path(run)
    H, C, G, names, x0, z0, step = parse_grid((run / "tww3_bai_map_capture_grid.csv").read_text(encoding="utf-8"))
    ev = events(run)
    frame = {k: ev["frame"][0][k] for k in ("min_x", "max_x", "min_z", "max_z")} if ev.get("frame") else None
    done = (ev.get("grid_done") or [{}])[0]
    xml_path = run / "tww3_bai_map_capture_ready.xml"
    xml_text = read_xml(xml_path) if xml_path.exists() else read_xml(SCENARIO)
    zone_list = zones(xml_text)
    mp, catchment = map_of(xml_text)
    dst = Path(out_root) / name
    dst.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst / "heights.npz", height=H, clear=C, ground=G, x0=np.float64(x0), z0=np.float64(z0),
                        step=np.float64(step))
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8")) if (run / "manifest.json").exists() else {}
    meta = {"name": name, "map": mp, "catchment": catchment, "frame": frame,
            "grid": {"x0": x0, "z0": z0, "step": step, "columns": int(H.shape[1]), "rows": int(H.shape[0]),
                     "layout": "height[iz, ix] at x = x0 + ix * step, z = z0 + iz * step (world metres)",
                     "nodes": bool(manifest.get("config", {}).get("nodes"))},
            "height_m": {"min": round(float(H.min()), 3), "max": round(float(H.max()), 3),
                         "mean": round(float(H.mean()), 3)},
            "clear_share": round(float(C.mean()), 4), "ground_names": names,
            "ground_share": {n: round(float((G == i).mean()), 4) for i, n in enumerate(names)},
            "deployment_zones": zone_list,
            "zones_note": "the capture scenario's deployment areas (scenarios/map_capture.xml), not the map's own",
            "source": {"run": str(run.relative_to(project.ROOT)) if run.is_relative_to(project.ROOT) else str(run),
                       "call": "bm:get_terrain_height(x, z) (src/apps/map/adapter.lua read_batch)",
                       "cells": done.get("cells"), "clock_elapsed_s": done.get("clock_elapsed_s"),
                       "build": manifest.get("build")},
            "date": datetime.date.today().isoformat()}
    (dst / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    preview(dst / "preview.png", H, x0, z0, step, zone_list, frame, f"{name}: {mp} ({step:g} m grid)")
    text = (run / "tww3_bai_map_capture_grid.csv").read_text(encoding="utf-8")
    blocked, info = passable_layers(C, G, names, parse_reach(text, H.shape))
    meta = write_layers(dst, blocked, x0, z0, step, info, meta["source"]["run"])
    preview_layers(dst / "passable.png", blocked, G, names, x0, z0, step, f"{name}: cannot stand")
    return dst, meta


def capture(name, step=1, python=sys.executable):
    """Builds map-capture (nodes, the step, --features: objects and every node's reach after deployment, ~25 s more)
    and launches it, then converts the new run. The probe lock is held; not when
    the user plays."""
    if HUMAN.exists():
        raise SystemExit("the user plays (build/steps/HUMAN_PLAYING)")
    if LOCK.exists():
        raise SystemExit(f"the probe lock is held: {LOCK.read_text(encoding='utf-8', errors='replace').strip()}")
    LOCK.write_text(f"map_grid {name} {datetime.datetime.now().isoformat(timespec='seconds')}\n", encoding="utf-8")
    try:
        before = set(p.name for p in RUNS.glob("*")) if RUNS.exists() else set()
        code = subprocess.run([python, "-m", "tools.build", "map-capture", "--step", str(step), "--nodes",
                               "--features"],
                              cwd=project.ROOT, stdout=subprocess.DEVNULL).returncode
        if code:
            raise SystemExit(f"build failed ({code})")
        code = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                               str(project.ROOT / "tools" / "launcher" / "launch.ps1"), "-Target", "map-capture"],
                              cwd=project.ROOT).returncode
        new = sorted(set(p.name for p in RUNS.glob("*")) - before)
        if code or not new:
            raise SystemExit(f"launch failed ({code}), runs {new}")
    finally:
        LOCK.unlink(missing_ok=True)
    return convert(RUNS / new[-1], name)


TRUNK_MAX_M = 2.25          # the game's cap on a tree body's radius (build/diag_g49/terrain/trees.md 3)
TRUNK_ZERO_M = 0.85         # a model whose CollRadius is 0 gets this radius before the scale


def tree_bodies(points, meta_rows):
    """The trunks of a map: points - the map's tree list rows (model, x, y, z, scale: the tree_list.bin read, world
    metres), meta_rows - vegetation.pack tree_metadata.txt rows (Filename, CollHeight, CollRadius). Each tree is a
    vertical cylinder from its base y: radius min((CollRadius or 0.85) x scale, 2.25), height CollHeight x scale; a
    model without a metadata row has no body (the game builds none). -> dict of float32 arrays x, z, y, r, h."""
    meta = {}
    for row in meta_rows:
        name = (row.get("Filename") or "").strip().lower()
        try:
            meta[name] = (float(row["CollHeight"]), float(row["CollRadius"]))
        except (KeyError, TypeError, ValueError):
            continue
    out = {k: [] for k in ("x", "z", "y", "r", "h")}
    for p in points:
        m = meta.get(str(p["model"]).strip().lower())
        if m is None:
            continue
        s = float(p["scale"])
        out["x"].append(float(p["x"]))
        out["z"].append(float(p["z"]))
        out["y"].append(float(p["y"]))
        out["r"].append(min((m[1] or TRUNK_ZERO_M) * s, TRUNK_MAX_M))
        out["h"].append(m[0] * s)
    return {k: np.asarray(v, np.float32) for k, v in out.items()}


def trees(points_csv, meta_txt, name, out_root=MAPS):
    """trees.npz next to the map's heights.npz: the trunks (tree_bodies) of the map's tree list (a csv[.gz] with model,
    x, y, z, scale - research/evidence/map/forest-trees-20260925/tree-points.csv.gz for Kislev) and tree_metadata.txt
    (vegetation.pack battleterrain/vegetation/trees/tree_metadata.txt, tab separated)."""
    import gzip
    opener = gzip.open if str(points_csv).endswith(".gz") else open
    with opener(points_csv, "rt", encoding="utf-8") as f:
        points = list(csv.DictReader(f))
    with open(meta_txt, encoding="utf-8", errors="replace") as f:
        meta_rows = list(csv.DictReader(f, delimiter="\t"))
    body = tree_bodies(points, meta_rows)
    dst = Path(out_root) / name
    dst.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst / "trees.npz", **body)
    return dst / "trees.npz", {"points": len(points), "trunks": int(len(body["x"])),
                               "r_median": round(float(np.median(body["r"])), 3) if len(body["r"]) else None,
                               "h_median": round(float(np.median(body["h"])), 2) if len(body["h"]) else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("trees")
    tr.add_argument("--name", required=True)
    tr.add_argument("--points", type=Path, required=True)
    tr.add_argument("--meta", type=Path, required=True)
    c = sub.add_parser("capture")
    c.add_argument("--name", required=True)
    c.add_argument("--step", type=int, choices=(1, 2), default=1)
    v = sub.add_parser("convert")
    v.add_argument("run", type=Path)
    v.add_argument("--name", required=True)
    ly = sub.add_parser("layers")
    ly.add_argument("run", type=Path)
    ly.add_argument("--name", required=True)
    ly.add_argument("--out-root", type=Path, default=MAPS)
    args = ap.parse_args(argv)
    if args.cmd == "layers":
        dst, meta = layers(args.run, args.name, args.out_root)
        print(dst)
        print(json.dumps(meta["passable"], indent=1))
        return 0
    if args.cmd == "trees":
        path, info = trees(args.points, args.meta, args.name)
        print(path)
        print(json.dumps(info))
        return 0
    dst, meta = capture(args.name, args.step) if args.cmd == "capture" else convert(args.run, args.name)
    print(dst)
    print(json.dumps({k: meta[k] for k in ("map", "grid", "height_m", "source")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

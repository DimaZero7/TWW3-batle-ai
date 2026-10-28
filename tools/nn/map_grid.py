"""A real map's height grid for the network (build/maps/<name>/): one capture a map through the map-capture target
(src/entries/map_capture.lua: bm:get_terrain_height over the radar frame, the multiples of the step - with step 1 or 2
the engine's own 2 m nodes are among them - plus clear / ground type), then heights.npz (float32 [rows, columns], row = z,
column = x, origin x0 / z0, step), meta.json (map, frame, deployment zones of the capture scenario, source run, date,
timing) and preview.png (dark lows, light hills, the zones outlined).

    python -m tools.nn.map_grid capture --name ksl_plains_01 [--step 1]   # build, launch, convert (needs the game free)
    python -m tools.nn.map_grid convert build/map-capture/runs/<run> --name ksl_plains_01
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
    return dst, meta


def capture(name, step=1, python=sys.executable):
    """Builds map-capture (nodes, the step) and launches it, then converts the new run. The probe lock is held; not when
    the user plays."""
    if HUMAN.exists():
        raise SystemExit("the user plays (build/steps/HUMAN_PLAYING)")
    if LOCK.exists():
        raise SystemExit(f"the probe lock is held: {LOCK.read_text(encoding='utf-8', errors='replace').strip()}")
    LOCK.write_text(f"map_grid {name} {datetime.datetime.now().isoformat(timespec='seconds')}\n", encoding="utf-8")
    try:
        before = set(p.name for p in RUNS.glob("*")) if RUNS.exists() else set()
        code = subprocess.run([python, "-m", "tools.build", "map-capture", "--step", str(step), "--nodes"],
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
    args = ap.parse_args(argv)
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

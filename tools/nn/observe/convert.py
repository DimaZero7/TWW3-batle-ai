"""Recorded battles -> demonstrations for training (one file a battle: build/observe/data/<run>.npz).

    python -m tools.nn.observe.convert --human build/human/runs/20261010-165656 [...]
    python -m tools.nn.observe.convert --list build/observe/aivai_ok.txt [--limit 5]
        [--ckpt build/v2/itV2.pt] [--out build/observe/data] [--workers 4]

For every side of another player (record.py: a human's side 1 in tools.build human --enemy-ai net; both sides of the
game's AI against the game's AI, build/nn-arena/runs) and every recorded second: the network's input of that side (its
own view, the companion's path), the critic's value on the side's critic input, the advantage (advantage.py: GAE of
the reward v2 to the battle's end minus V, and the plain return minus V, and the window20 fallback), and the side's
new orders in the network's language (labels.py). Keys s<side>_<name>; "meta" a JSON string. The network's v2 inputs
the recording has no reading of: every unit that takes orders is free (no commitment), commit 0 (store.py adds them).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from tools import config as project
from tools.nn.observe import advantage as adv
from tools.nn.observe import labels as lab
from tools.nn.observe import record
from tools.nn.train import checkpoint

OUT = project.BUILD / "observe" / "data"
CKPT = project.BUILD / "v2" / "itV2.pt"
RUNS = project.BUILD / "nn-arena" / "runs"
CRITIC_KEYS = ("tokens", "ctx", "own", "attend", "pos")
F16 = ("tokens", "ctx", "abil")
BOOLS = ("own", "attend", "ctrl", "target_ok", "abil_ok")


def critic_values(critic, obs_list):
    out = []
    for a in range(0, len(obs_list), 256):
        chunk = obs_list[a:a + 256]
        d = {k: torch.as_tensor(np.concatenate([np.asarray(getattr(o, k)) for o in chunk], 0)) for k in CRITIC_KEYS}
        with torch.no_grad():
            out.append(critic(d).float().numpy())
    return np.concatenate(out)


def convert(run_dir, actor_cfg, critic, ckpt_name=""):
    """One battle -> (dict of arrays, meta)."""
    from tools.nn.model import sectors
    rec = record.load(run_dir)
    pipe = record.Pipeline(rec)
    T = len(rec.frames)
    sides = pipe.sides
    obs = {s: [] for s in sides}
    crit = {s: [] for s in sides}
    raws, gold, t = [], [], []
    for k in range(T):
        st = pipe.step(k)
        for s in sides:
            obs[s].append(st.obs[s])
            crit[s].append(st.critic[s])
        raws.append(st.raw)
        gold.append(st.gold)
        t.append(rec.frames[k][0] / 1000.0)
    raw = {f: np.stack([r[f] for r in raws]) for f in record.RAW}
    gold, t = np.asarray(gold), np.asarray(t)
    side = np.asarray(rec.battle.side)
    missile = np.nan_to_num(pipe.b.shape["range"]) > 0
    widths = [rec.widths.get(n) for n in rec.battle.names]
    spacing = float(pipe.b.shape["spacing"])
    out, meta_sides = {}, {}
    for s in sides:
        o = obs[s]
        cat = lambda k, dt=None: np.concatenate([np.asarray(getattr(x, k)) for x in o], 0).astype(dt or np.float32)
        arrays = {k: cat(k, np.float16 if k in F16 else (bool if k in BOOLS else np.float32))
                  for k in ("tokens", "ctx", "own", "attend", "ctrl", "target_ok", "pos", "abil", "abil_ok")
                  if getattr(o[0], k) is not None}
        frame = o[0].frame
        geo = sectors.geo(frame, torch.as_tensor(pipe.b.setup.bounds, dtype=torch.float32))[0].numpy()
        lb = lab.read(raw, side, s, arrays["ctrl"], arrays["target_ok"], missile, human=(rec.human == s),
                      widths=widths, spacing=spacing)
        point = np.full(lb["kind"].shape, -1, np.int64)
        has = np.isfinite(lb["px"])
        err = np.zeros(0)
        if has.any():
            cells = lab.grid_cells(actor_cfg, geo)
            point[has] = lab.cell_of(frame, lb["px"][has], lb["pz"][has], cells)
            # how far the cell's centre lies from the (clipped) point: the grid covers +-grid_half m in the frame
            centre = dict(zip(cells[0].tolist(), cells[1]))
            fl = np.array([centre[int(i)] for i in point[has]])
            px, pz = record.clip_points(lb["px"][has], lb["pz"][has], pipe.bounds)
            wx, wz = frame.world(fl[None, :, 0], fl[None, :, 1])
            err = np.hypot(wx[0] - px, wz[0] - pz)
        v = critic_values(critic, crit[s]) if critic is not None else np.zeros(T)
        v[-1] = 0.0
        r = adv.rewards(gold, rec.winner, s)
        a_gae, ret = adv.gae(r, v)
        a_mc, _ = adv.gae(r, v, lam=1.0)
        w_trade, w_adv = adv.window(gold, t, s)
        arrays.update(geo=geo.astype(np.float32), t=t.astype(np.float32), value=v.astype(np.float32),
                      ret=ret.astype(np.float32), adv_critic=a_gae.astype(np.float32), adv_mc=a_mc.astype(np.float32),
                      w20_trade=w_trade.astype(np.float32), w20_adv=w_adv.astype(np.float32),
                      reward=r.astype(np.float32), kind=lb["kind"].astype(np.int8), target=lb["target"].astype(np.int16),
                      point=point.astype(np.int32), run=lb["run"], run_known=lb["run_known"], alt=lb["alt"])
        for k, x in arrays.items():
            out[f"s{s}_{k}"] = x
        rows = np.nonzero(lb["kind"] >= 0)
        per = arrays["kind"][rows]
        a_at = a_gae[rows[0]]
        meta_sides[s] = {"orders": int(len(per)), "hold": int((per == 0).sum()), "move": int((per == 1).sum()),
                         "attack": int((per == 2).sum()), "run_set": int(lb["alt"].sum()),
                         "good_critic": int((a_at > 0).sum()), "good_mc": int((a_mc[rows[0]] > 0).sum()),
                         "good_w20": int(((w_adv[rows[0]] > 0) & (w_trade[rows[0]] > 0)).sum()),
                         "good_both": int(((a_at > 0) & (w_adv[rows[0]] > 0) & (w_trade[rows[0]] > 0)).sum()),
                         "point_err_p50": float(np.median(err)) if len(err) else 0.0,
                         "point_err_p95": float(np.percentile(err, 95)) if len(err) else 0.0,
                         "point_far": int((err > 25).sum()),
                         "adv_mean": float(a_at.mean()) if len(a_at) else 0.0,
                         "adv_std": float(a_at.std()) if len(a_at) else 0.0,
                         "won": rec.winner == s, "human": rec.human == s}
    meta = {"run": rec.run, "path": str(rec.path), "T": T, "N": len(rec.battle.names), "sides": list(sides),
            "winner": rec.winner, "ckpt": ckpt_name, "per_side": meta_sides,
            "cfg": {"sectors": actor_cfg.sectors, "fine": actor_cfg.fine, "grid_half": actor_cfg.grid_half}}
    out["meta"] = np.asarray(json.dumps(meta))
    return out, meta


_NETS = {}


def _job(job):
    run_dir, out_dir, ckpt = job
    torch.set_num_threads(1)
    if ckpt not in _NETS:
        _NETS[ckpt] = (checkpoint.load_policy(ckpt, "cpu").cfg, checkpoint.load_critic(ckpt, "cpu").eval())
    cfg, critic = _NETS[ckpt]
    t0 = time.time()
    try:
        arrays, meta = convert(run_dir, cfg, critic, str(ckpt))
    except Exception as e:  # noqa: BLE001 - reported, the others go on
        import traceback
        traceback.print_exc()
        return f"{Path(run_dir).name}: {type(e).__name__}: {e}", None
    np.savez_compressed(Path(out_dir) / f"{Path(run_dir).name}.npz", **arrays)
    return f"{Path(run_dir).name}: T {meta['T']}, {time.time() - t0:.0f} s, " + "; ".join(
        f"side {s} {'won ' if d['won'] else ''}{'(human) ' if d['human'] else ''}orders {d['orders']} "
        f"(hold {d['hold']} move {d['move']} attack {d['attack']}), A>0 {d['good_critic']}, w20 {d['good_w20']}, "
        f"both {d['good_both']}, cell from the point p50 {d['point_err_p50']:.0f} / p95 {d['point_err_p95']:.0f} m "
        f"(> 25 m: {d['point_far']})" for s, d in meta["per_side"].items()), meta


def summary(metas):
    tot = {}
    for m in metas:
        for d in m["per_side"].values():
            for k, v in d.items():
                if isinstance(v, (int, bool)):
                    tot[k] = tot.get(k, 0) + int(v)
    sides = sum(len(m["per_side"]) for m in metas)
    seconds = sum(m["T"] * len(m["sides"]) for m in metas)
    n = max(1, tot.get("orders", 0))
    return (f"{len(metas)} battles, {sides} sides, {seconds} side-seconds; orders {tot.get('orders', 0)} "
            f"({tot.get('orders', 0) / max(1, seconds):.2f} a side-second; hold {tot.get('hold', 0)}, move "
            f"{tot.get('move', 0)}, attack {tot.get('attack', 0)}, running-move set {tot.get('run_set', 0)}); "
            f"A>0 by the critic {tot.get('good_critic', 0)} ({tot.get('good_critic', 0) / n:.2f}), by the plain return "
            f"{tot.get('good_mc', 0) / n:.2f}, by window20 {tot.get('good_w20', 0) / n:.2f}, both critic and window20 "
            f"{tot.get('good_both', 0) / n:.2f}; move cells > 25 m from the point {tot.get('point_far', 0)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--human", nargs="*", default=[], help="run folders of human-vs-network battles")
    ap.add_argument("--list", help="a file of nn-arena run names (build/observe/aivai_ok.txt): game AI vs game AI")
    ap.add_argument("--limit", type=int, default=0, help="the first that many of --list")
    ap.add_argument("--ckpt", default=str(CKPT), help="the network whose critic judges and whose grid labels")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--force", action="store_true", help="convert again a battle already in --out")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = [Path(h) for h in a.human]
    if a.list:
        names = [n.strip() for n in Path(a.list).read_text(encoding="utf-8").split() if n.strip()]
        runs += [RUNS / n for n in (names[:a.limit] if a.limit else names)]
    jobs = [(str(r), str(out), a.ckpt) for r in runs if a.force or not (out / f"{r.name}.npz").exists()]
    print(f"{len(jobs)} battles to convert ({len(runs) - len(jobs)} already in {out}), {a.workers} workers", flush=True)
    t0, metas = time.time(), []
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.workers) as pool:
            for n, (msg, meta) in enumerate(pool.imap_unordered(_job, jobs), 1):
                print(f"[{n}/{len(jobs)}] {msg} ({time.time() - t0:.0f} s)", flush=True)
                metas += [meta] if meta else []
    else:
        for n, j in enumerate(jobs, 1):
            msg, meta = _job(j)
            print(f"[{n}/{len(jobs)}] {msg} ({time.time() - t0:.0f} s)", flush=True)
            metas += [meta] if meta else []
    if metas:
        print(summary(metas), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

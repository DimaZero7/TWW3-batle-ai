"""How well the student guesses the game AI (tools/nn/student, docs/en/training/student.md), and how it plays.

    python -m tools.nn.student.eval --checkpoint build/student/s12/best.pt [--split val,exam] [--max-sides 40]
        [--sim 32 --opponents nearest,ai_like] [--out build/student/s12/eval.json]

* The guesses (objective.py's accuracies) on the held-out starts (val: the AI-vs-AI starts the training never saw;
  exam: the AI's side of the exam battles, 3 starts) and the point's error: the most likely sector, then its most
  likely cell, against the AI's cell (m between the centres).
* Beside them the simple baselines on the same seconds (baselines()): "repeat" - the unit's last order again (the kind,
  target, cell and run of its last new order; hold before any), and "when" always keep; "nearest" - attack the
  nearest seen enemy, its cell for a move, run, and "when" always a new order; "majority" - the most common kind
  (move), running.
* --sim N: the student in the simulator against the scripts (evaluate.play: N generated battles an opponent in
  swapped pairs, its orders sampled as in training): wins, gold, orders a minute, the order kinds, contact.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from tools.nn.model import heads as hd
from tools.nn.model import observation as ob
from tools.nn.model import sectors as sc
from tools.nn.student import dataset as ds
from tools.nn.student import objective as obj
from tools.nn.train import checkpoint

KEEP, HOLD, MOVE, ATTACK, WITHDRAW = ds.KEEP, ds.HOLD, ds.MOVE, ds.ATTACK, ds.WITHDRAW


def cell_centre(cfg, point):
    """(forward, lateral) m of grid cells (numpy or torch point indices)."""
    nk = cfg.sectors * cfg.fine
    step = 2 * cfg.grid_half / nk
    return (-cfg.grid_half + (point // nk + 0.5) * step, -cfg.grid_half + (point % nk + 0.5) * step)


def cell_at(cfg, f, l):
    """The grid cell of frame points (numpy)."""
    nk = cfg.sectors * cfg.fine
    step = 2 * cfg.grid_half / nk
    r = np.clip(np.floor((f + cfg.grid_half) / step), 0, nk - 1).astype(np.int64)
    c = np.clip(np.floor((l + cfg.grid_half) / step), 0, nk - 1).astype(np.int64)
    return r * nk + c


def point_error(cfg, p, q):
    fa, la = cell_centre(cfg, p)
    fb, lb = cell_centre(cfg, q)
    return np.hypot(fa - fb, la - lb)


def errors_text(e):
    if not len(e):
        return "point: none"
    e = np.asarray(e)
    return (f"point error p50 {np.median(e):.0f} m, within 25 m {np.mean(e <= 25 + 1e-6):.3f}, 50 m "
            f"{np.mean(e <= 50 + 1e-6):.3f}, 100 m {np.mean(e <= 100 + 1e-6):.3f}")


@torch.no_grad()
def accuracy(actor, sides, B=16, T=32, device="cpu", N=None):
    """(summary of objective.py's accuracies, point errors [m]) over these sides."""
    cfg = actor.cfg
    actor.eval()
    st = ds.Streams(sides, B=B, T=T, N=N, shuffle=False)
    acc, h, err = {}, None, []
    for steps in st:
        b = ds.to_batch(steps, device)
        if h is None:
            h = actor.initial({k: v[0] for k, v in b["obs"].items()})
        _, s, h, aux = obj.run(actor, b, h)
        obj.add(acc, s)
        L, lg = aux["L"], aux["logits"]
        place = (L["kind"] >= 0) & ((L["k1"] == MOVE) | (L["k1"] == WITHDRAW))
        if bool(place.any()):
            ps = lg["sector"].argmax(-1)
            a = hd.Action(L["k1"], sc.point_index(cfg, ps, torch.zeros_like(ps)), L["target"], L["run"], None,
                          L["commit"])
            lg2, _ = actor._finish(aux["x"], aux["pack"], aux["flat"], a, False, 1.0, True, False)
            pred = sc.point_index(cfg, ps, lg2["fine"].argmax(-1))
            err.append(point_error(cfg, pred[place].cpu().numpy(), L["point"][place].cpu().numpy()))
    return obj.summary(acc), (np.concatenate(err) if err else np.zeros(0))


def baselines(sides, cfg, N=None):
    """{name: (summary as objective.summary's keys that apply, point errors)} on the same labelled seconds."""
    N = N or max(s.N for s in sides)
    loader = ds.Loader(N)
    names = ("repeat", "nearest", "majority")
    acc = {n: {} for n in names}
    err = {n: [] for n in names}
    for side in sides:
        a = loader.load(side)
        kind, target, point, run = a["kind"], a["target"], a["point"], a["run"]
        T = kind.shape[0]
        last = np.full(N, HOLD)
        last_t, last_p, last_r = np.full(N, -1), np.full(N, -1), np.zeros(N, bool)
        pos = a["pos"].astype(np.float64) * ob.POS
        for k in range(T):
            lab = kind[k] >= 0
            new = lab & (kind[k] != KEEP)
            ok = a["target_ok"][k]
            # the nearest seen enemy of every unit
            d = np.hypot(pos[k, :, None, 0] - pos[k, None, :, 0], pos[k, :, None, 1] - pos[k, None, :, 1])
            d = np.where(ok[None, :], d, np.inf)
            near = d.argmin(1)
            has = np.isfinite(d.min(1))
            near_cell = cell_at(cfg, pos[k, near, 0], pos[k, near, 1])
            alt = a["alt"][k]
            att = new & (kind[k] == ATTACK)
            place = new & ((kind[k] == MOVE) | (kind[k] == WITHDRAW))
            runk = new & a["run_known"][k] & ((kind[k] == MOVE) | (kind[k] == ATTACK))
            guess = {"repeat": (last, last_t, last_p, last_r),
                     "nearest": (np.full(N, ATTACK), np.where(has, near, -1), near_cell, np.ones(N, bool)),
                     "majority": (np.full(N, MOVE), np.full(N, -1), np.full(N, -1), np.ones(N, bool))}
            for n, (gk, gt, gp, gr) in guess.items():
                c = acc[n]
                c["labels"] = c.get("labels", 0) + lab.sum()
                c["new"] = c.get("new", 0) + new.sum()
                c["kind_ok"] = c.get("kind_ok", 0) + (new & ((gk == kind[k]) | (alt & (gk == WITHDRAW)))).sum()
                c["attack"] = c.get("attack", 0) + att.sum()
                c["target_ok"] = c.get("target_ok", 0) + (att & (gt == target[k])).sum()
                c["place"] = c.get("place", 0) + place.sum()
                good = place & (gp >= 0)
                sp, sl = _sector(cfg, np.where(gp >= 0, gp, 0)), _sector(cfg, point[k].clip(0))
                c["sector_ok"] = c.get("sector_ok", 0) + (good & (sp == sl)).sum()
                c["run_n"] = c.get("run_n", 0) + runk.sum()
                c["run_ok"] = c.get("run_ok", 0) + (runk & (gr == run[k])).sum()
                e = point_error(cfg, np.where(gp >= 0, gp, 0)[place], point[k][place])
                err[n].append(np.where(gp[place] >= 0, e, np.inf))
            # the repeat baseline's memory: the last new order of every unit
            last = np.where(new, kind[k], last)
            last_t = np.where(new, np.where(kind[k] == ATTACK, target[k], -1), last_t)
            last_p = np.where(place, point[k], last_p)
            last_r = np.where(new, run[k], last_r)
    out = {}
    for n in names:
        c = acc[n]
        d = lambda x, y: float(c.get(x, 0)) / max(1.0, float(c.get(y, 0)))
        out[n] = ({"kind_acc": d("kind_ok", "new"), "target_acc": d("target_ok", "attack"),
                   "sector_acc": d("sector_ok", "place"), "run_acc": d("run_ok", "run_n"),
                   "when_recall": 0.0 if n == "repeat" else 1.0,
                   "when_precision": float("nan") if n == "repeat" else d("new", "labels")},
                  np.concatenate(err[n]) if err[n] else np.zeros(0))
    return out


def _sector(cfg, point):
    nk, k = cfg.sectors * cfg.fine, cfg.fine
    row, col = point // nk, point % nk
    return (row // k) * cfg.sectors + col // k


def base_text(n, sm, e):
    return (f"  {n:8} when: recall {sm['when_recall']:.2f} precision {sm['when_precision']:.2f}; kind "
            f"{sm['kind_acc']:.3f}, target {sm['target_acc']:.3f}, sector {sm['sector_acc']:.3f}, run "
            f"{sm['run_acc']:.3f}; {errors_text(e[np.isfinite(e)])} (no point {np.mean(~np.isfinite(e)) if len(e) else 0:.2f})")


def sim(actor, n, opponents, device, limit_s=3600.0):
    """The student's battles against the scripts (evaluate.play, module doc) -> the result dict, printed."""
    from tools.nn.train import evaluate
    from tools.nn.train.run import show
    t = time.time()
    res = evaluate.play(actor, opponents=tuple(opponents), device=device, limit_s=limit_s, generated=n,
                        paired=True, baseline=False, measure=("behaviour", "liveliness"))
    print(f"sim: {time.time() - t:.0f} s", flush=True)
    show("student", res)
    for name, o in res["by_opponent"].items():
        print(f"  {name}: win {o['win_rate']:.3f} ({o['wins']}/{o['games']}), gold trade {o.get('gold_trade', 0):+.3f}, "
              f"orders/min {o['orders_per_minute']:.1f}, contact share {o.get('contact_share', float('nan')):.2f}, "
              f"first contact {o.get('first_contact_s') or float('nan'):.0f} s, timeouts {o['timeouts']:.2f}", flush=True)
    return res


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", nargs="*", default=[str(d) for d in ds.DIRS])
    ap.add_argument("--split", default="val,exam", help="val, exam, train (comma-separated; empty: none)")
    ap.add_argument("--val-share", type=float, default=None, help="default: the checkpoint's")
    ap.add_argument("--exam-train", default=None, help="default: the checkpoint's")
    ap.add_argument("--max-sides", type=int, default=0, help="at most this many sides a split (0: all)")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--sim", type=int, default=0, help="generated battles an opponent in the simulator (0: none)")
    ap.add_argument("--opponents", default="nearest,ai_like")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    device = torch.device(a.device)
    actor = checkpoint.load_policy(a.checkpoint, device)
    meta = checkpoint.meta(a.checkpoint)
    vs = a.val_share if a.val_share is not None else float(meta.get("val_share", 0.15))
    et = a.exam_train if a.exam_train is not None else str(meta.get("exam_train", ""))
    sides = ds.discover([Path(d) for d in a.data], actor.cfg)
    parts = ds.split(sides, vs, [int(s) for s in et.split(",") if s])
    N = max(s.N for s in sides)
    res = {"checkpoint": a.checkpoint}
    from tools.nn.student.train import pick
    for name in filter(None, a.split.split(",")):
        part = pick(parts[name], a.max_sides, a.seed)
        t = time.time()
        sm, err = accuracy(actor, part, a.B, a.T, device, N)
        print(f"{name}: {len(part)} sides ({len({s.group for s in part})} starts), {time.time() - t:.0f} s\n"
              f"  student  {obj.text(sm)}; {errors_text(err)}", flush=True)
        res[name] = {"student": dict(sm, point_p50=float(np.median(err)) if len(err) else None,
                                     point_25=float(np.mean(err <= 25 + 1e-6)) if len(err) else None)}
        if not a.no_baselines:
            for n, (bs, e) in baselines(part, actor.cfg, N).items():
                print(base_text(n, bs, e), flush=True)
                res[name][n] = bs
    if a.sim:
        r = sim(actor, a.sim, a.opponents.split(","), device)
        res["sim"] = {k: {kk: v.get(kk) for kk in ("win_rate", "games", "gold_trade", "orders_per_minute", "kinds",
                                                    "contact_share", "first_contact_s", "timeouts")}
                      for k, v in r["by_opponent"].items()}
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

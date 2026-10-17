"""Who limits the blows in a crowd, the strikers or the men struck: the crowd probe (tools/nn/charge_probe.py plan
'crowd': swordsmen on clanrats cut to 5 / 15 / 25 men or full, one swordsmen unit or three - two more on the flanks;
samples and the soldiers' places every 0.1 s) read (numpy, no torch).

  python -m tools.nn.fighters.crowddata build/charge-probe/runs/<crowd run> [--from 3 --to 30]

Per lane and side struck, over the seconds from..to after the first contact, while neither side routs: the HP the side
loses a second against the formula of one striker (p x mean blow / interval, swingdata) times
  S        the striking men within reach (3.15 m) of a man of the side struck
  D        the struck men with a striker within reach
  cap K    sum over the struck men of min(n_i, K), n_i the strikers within reach of man i
  match K  the most strikers that can be paired with struck men in reach, each struck man taking at most K (greedy)
A rule the game follows gives the same ratio (~1) on every lane: few men struck by many strikers (the cut clanrats)
tell 'the strikers' (S) from 'the men struck' (D, cap, match)."""
import argparse
import json
from pathlib import Path

import numpy as np

from tools.nn import charge_probe as cp
from tools.nn.fighters import shape, swingdata as sw

KS = (1, 2, 3)


def match(inr, k):
    """Greedy pairing: strikers (rows) to struck men (columns) in reach, each struck man at most k; the strikers with
    the fewest choices first, each to the struck man in reach with the fewest strikers so far."""
    if not inr.size:
        return 0
    load = np.zeros(inr.shape[1], int)
    n = 0
    for i in np.argsort(inr.sum(1)):
        c = np.flatnonzero(inr[i] & (load < k))
        if len(c):
            load[c[np.argmin(load[c])]] += 1
            n += 1
    return n


def counts(A, T):
    if not len(A) or not len(T):
        return None
    inr = np.sqrt(((A[:, None, :] - T[None, :, :]) ** 2).sum(-1)) <= shape.REACH_M
    n_i = inr.sum(0)
    out = {"S": int(inr.any(1).sum()), "D": int(inr.any(0).sum())}
    for k in KS:
        out[f"cap{k}"] = int(np.minimum(n_i, k).sum())
        out[f"match{k}"] = match(inr, k)
    return out


def lane_rows(ln, t_from, t_to):
    sp = ln["spec"]
    c = (ln.get("contacts") or {}).get(1)
    if c is None or not ln["men"]:
        return []
    t = np.array([x[0] for x in ln["samples"]])
    ex_men = {round(tm, 2): e for tm, e in ln.get("extra_men", [])}
    rows = []
    # (struck side, its hp series, the striking unit key, the struck key)
    hp = {"tg": cp.series(ln["samples"], "tg", "hp"), "a": cp.series(ln["samples"], "a", "hp")}

    def bad(i):
        a, g = ln["samples"][i][1] or {}, ln["samples"][i][2] or {}
        return any("rout" in (x.get("st") or "") or "shatter" in (x.get("st") or "") for x in (a, g))
    for j in range(len(ln["men"]) - 1):
        tm, fa, ft = ln["men"][j]
        tn = ln["men"][j + 1][0]
        if not (c + t_from <= tm < c + t_to) or tn - tm > 0.5:
            continue
        i0, i1 = int(np.argmin(np.abs(t - tm))), int(np.argmin(np.abs(t - tn)))
        if bad(i0) or bad(i1):
            continue
        A = shape.xy(fa)
        for e in ex_men.get(round(tm, 2), []):
            A = np.concatenate([A, shape.xy(e)]) if len(A) else shape.xy(e)
        T = shape.xy(ft)
        dt = tn - tm
        # the attackers' side strikes the clanrats; the clanrats strike the attacker unit only (the extras' HP apart)
        for who, S_, T_, sk, tk in (("tg", A, T, sp["a_key"], sp["t_key"]), ("a", T, shape.xy(fa), sp["t_key"], sp["a_key"])):
            h0, h1 = cp.at(t, hp[who], tm), cp.at(t, hp[who], tn)
            cnt = counts(S_, T_)
            if cnt is None or not np.isfinite(h0 + h1):
                continue
            hit, _ = sw.mean_hit(sk, tk)
            p, iv = sw.p_hit(sk, tk)
            rows.append(dict(cnt, struck=who, lost=max(0.0, h0 - h1), dt=dt, f=p * hit / iv))
    return rows


def summary(rows):
    keys = ["S", "D"] + [f"cap{k}" for k in KS] + [f"match{k}" for k in KS]
    lost = sum(r["lost"] for r in rows)
    secs = sum(r["dt"] for r in rows)
    out = {"secs": round(secs, 1), "hp_s": round(lost / secs, 2) if secs else None}
    for k in keys:
        den = sum(r["f"] * r[k] * r["dt"] for r in rows)
        out[k] = round(sum(r[k] * r["dt"] for r in rows) / secs, 1) if secs else None
        out[f"game/{k}"] = round(lost / den, 2) if den > 0 else None
    return out


WINDOWS = ((0, 3), (3, 10), (10, 30))


def hp_rates(ln):
    """{(struck, window): HP lost a second} after the first contact, until either side routs."""
    c = (ln.get("contacts") or {}).get(1)
    if c is None:
        return {}
    t = np.array([x[0] for x in ln["samples"]])
    rout = [any("rout" in ((x[k] or {}).get("st") or "") or (x[k] or {}).get("r") for k in (1, 2))
            for x in ln["samples"]]
    end = next((t[i] for i, r in enumerate(rout) if r and t[i] > c), t[-1])
    out = {}
    for who in ("tg", "a"):
        hp = cp.series(ln["samples"], who, "hp")
        for lo, hi in WINDOWS:
            t0, t1 = c + lo, min(c + hi, end)
            if t1 - t0 < 1.0:
                continue
            h0, h1 = cp.at(t, hp, t0), cp.at(t, hp, t1)
            if np.isfinite(h0 + h1):
                out[(who, f"{lo}-{hi}")] = (h0 - h1) / (t1 - t0)
    return out


def lab_compare(run_dirs, copies, device, lab_sets=()):
    """The lab's twin (probes.twin, fighters) on the crowd lanes: HP lost a second, lab / game, per lane and window."""
    from tools.nn.fighters import hook, probes
    for kv in lab_sets:
        k, v = kv.split("=", 1)
        *path, last = k.split(".")
        d = hook.LAB
        for p in path:
            d = d.setdefault(p, {})
        d[last] = json.loads(v)
    lanes = [ln for d in run_dirs for ln in cp.load_run(Path(d)) if (ln.get("contacts") or {}).get(1) is not None]
    figh = probes.twin(lanes, True, copies, device)
    rows = []
    for i, ln in enumerate(lanes):
        g = hp_rates(ln)
        sims = [hp_rates(x) for x in figh[i * copies:(i + 1) * copies]]
        for key, gv in g.items():
            sv = [s[key] for s in sims if key in s]
            if sv:
                rows.append({"lane": ln["spec"]["name"], "kind": ln["spec"].get("kind"), "struck": key[0],
                             "window": key[1], "game": gv, "lab": float(np.mean(sv))})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--from", dest="t_from", type=float, default=3.0)
    ap.add_argument("--to", dest="t_to", type=float, default=30.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--lab", action="store_true", help="also the lab's twin on the lanes (torch)")
    ap.add_argument("--set", action="append", default=[], help="section.key=json over tools/nn/fighters/lab.json")
    ap.add_argument("--copies", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    if a.lab:
        rows = lab_compare(a.runs, a.copies, a.device, a.set)
        print("HP lost a second after contact: game / lab (lab / game)")
        for r in rows:
            print(f"  {r['lane']} {r['kind']:<11} struck {r['struck']:<2} {r['window']:>5} s: {r['game']:7.1f} / "
                  f"{r['lab']:7.1f} ({r['lab'] / r['game'] if r['game'] > 0 else float('nan'):.2f})")
        for who in ("tg", "a"):
            for w in ("0-3", "3-10", "10-30"):
                rs = [r for r in rows if r["struck"] == who and r["window"] == w]
                if rs:
                    print(f"  sum struck {who} {w}: lab / game {sum(r['lab'] for r in rs) / sum(r['game'] for r in rs):.2f}")
        if a.out:
            Path(a.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")
        return
    res = []
    for d in a.runs:
        for ln in cp.load_run(Path(d)):
            rows = lane_rows(ln, a.t_from, a.t_to)
            for who in ("tg", "a"):
                rs = [r for r in rows if r["struck"] == who]
                if rs:
                    res.append(dict(summary(rs), lane=ln["spec"]["name"], kind=ln["spec"].get("kind"),
                                    struck="clanrats" if who == "tg" else "swordsmen (lane's own)"))
    cols = ["S", "D", "match1", "match2", "cap2"]
    print(f"{a.t_from}-{a.t_to} s after contact: HP/s, mean men, and game / (formula x men)")
    print("lane kind struck | secs | HP/s | " + " | ".join(cols) + " | " + " | ".join(f"g/{c}" for c in cols))
    for r in res:
        print(f"{r['lane']} {r['kind']} {r['struck']} | {r['secs']} | {r['hp_s']} | " +
              " | ".join(str(r[c]) for c in cols) + " | " + " | ".join(str(r[f'game/{c}']) for c in cols))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

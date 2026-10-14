"""The step replay's comparison (numpy only): per window and unit, what the simulator did against what the game
recorded, in cells of the situation at the window start (docs/en/training/simulator.md, "The step replay").

One row per (battle, window, unit) the game shows on the field at the window start (not dead, not gone):
    g_loss / s_loss   health lost over the window, share of the unit's start (the simulator: its copies' mean); a unit
                      leaving the map loses nothing (the game keeps its health)
    g_mel / s_mel     of it in melee (the game: seconds with its melee flag on this or the last row; the simulator: HP
                      its melee blows took), the rest from missiles
    g_rout / s_rout   a rout began in the window (not routing at the start): 0/1 and the copies' share
    g_rally / s_rally a router rallied in the window
    g_dmp / s_dmp     morale change over the window (MoralePercent), units standing through it in the game
Situation at the start (the game's state; the contact count and side from the simulator's own contact test at the game's
places - the recording names an attacker in only some of its melee seconds): see situation().
Ratios are sim / game of sums over a cell; the 95 % interval resamples battles (the cluster: windows of one battle are
not independent). python -m tools.nn.sim.stepcells build/stepreplay/h10 [...] --md out.md
"""
import argparse
from pathlib import Path

import numpy as np

BOOT = 1000
SEED = 7
SETTLE_S = 5         # seconds of a window after which its loss counts as settled (the late columns)


def load(path, src=""):
    """One step-replay npz -> {name: [rows]} (one row per window and unit on the field at the window start)."""
    a = np.load(path, allow_pickle=False)
    st = a["starts"].astype(int)
    H = int(a["H"])
    W = len(st)
    n = len(a["names"])
    if W == 0:
        return None
    g_hp = a["g_hp"].astype(float)
    g_men = a["g_men"].astype(float)
    T = len(g_hp)
    hp = g_hp.copy()
    for k in range(1, T):
        bad = ~np.isfinite(hp[k])
        hp[k, bad] = hp[k - 1, bad]
    hp = np.nan_to_num(hp, nan=1.0)
    men = np.nan_to_num(g_men)
    gone = np.maximum.accumulate((men <= 0) & (hp > 0.003), axis=0)
    dead = hp <= 0.003
    r = a["g_r"].astype(bool)
    m = a["g_m"].astype(bool)
    fire = a["g_fire"].astype(bool)
    tgt = a["g_target"].astype(int)
    side = a["side"].astype(int)
    mp = a["g_mp"].astype(float)
    ms = a["g_ms"].astype(float)
    on = ~dead[st] & ~gone[st] & (men[st] > 0)                                  # [W, n]
    end = st + H
    g0 = hp[st]
    g_loss = np.clip(g0 - hp[end], 0, None)
    # the game's loss by second: melee where the melee flag is on (this row or the last)
    per = np.clip(hp[:-1] - hp[1:], 0, None)                                   # [T-1, n]: row k -> k+1
    melr = m[1:] | m[:-1]
    g_mel = np.stack([(per[s:s + H] * melr[s:s + H]).sum(0) for s in st])
    s_hp = a["sim_hp"].astype(float)                                           # [W, H, C, n]
    s_loss = np.clip(g0[:, None, :] - s_hp[:, -1], 0, None).mean(1)
    hp0 = a["u_hp0"].astype(float)
    s_mel = (a["sim_mel"].astype(float).sum(1) / hp0[None, None, :]).mean(1)
    s_mis = (a["sim_mis"].astype(float).sum(1) / hp0[None, None, :]).mean(1)
    # the per-second loss curve (cumulative share by second of the window)
    g_cum = np.clip(g0[:, None, :] - np.stack([hp[s + 1:s + H + 1] for s in st]), 0, None)        # [W, H, n]
    s_cum = np.clip(g0[:, None, None, :] - s_hp, 0, None).mean(2)                                    # [W, H, n]
    r0 = r[st]
    g_rout = ~r0 & np.stack([r[s + 1:s + H + 1].any(0) for s in st])
    s_r = a["sim_r"].astype(bool)
    s_rout = (~r0[:, None, :] & s_r.any(1)).mean(1)
    g_rally = r0 & np.stack([(~r[s + 1:s + H + 1]).any(0) for s in st]) & on
    s_rally = (r0[:, None, :] & (~s_r).any(1)).mean(1)
    stand_g = ~r0 & ~np.stack([r[s + 1:s + H + 1].any(0) for s in st])
    g_dmp = mp[end] - mp[st]
    s_dmp = a["sim_mp"].astype(float)[:, -1].mean(1) - mp[st]
    # the game's morale when a rout began (its first routing row) and the simulator's morale at that second
    first = np.full((W, n), -1)
    for w, s in enumerate(st):
        rr = r[s + 1:s + H + 1]
        hit = rr.any(0) & ~r0[w]
        first[w] = np.where(hit, rr.argmax(0), -1)
    s_mp_at = np.full((W, n), np.nan)
    smp = a["sim_mp"].astype(float).mean(2)                                    # [W, H, n]
    for w in range(W):
        for i in np.flatnonzero(first[w] >= 0):
            s_mp_at[w, i] = smp[w, max(first[w, i] - 1, 0), i]
    # situation at the start
    enemy = side[:, None] != side[None, :]
    att = np.zeros((W, n), int)
    shot_at = np.zeros((W, n), bool)
    for w, s in enumerate(st):
        tg = tgt[s]
        for j in range(n):
            if tg[j] >= 0 and enemy[j, tg[j]]:
                if m[s, j] and not r[s, j]:
                    att[w, tg[j]] += 1
                if fire[s, j]:
                    shot_at[w, tg[j]] = True
    # contact clock from the clocks
    contact_s = a["c_contact_s"].astype(float)
    # contact begins in the window (the game: melee flag off at the start, on later; standing)
    m_in = np.stack([m[s + 1:s + H + 1].any(0) for s in st])
    begins = ~m[st] & m_in & ~r0
    f = a["g_f"].astype(bool)
    ran = np.stack([f[max(s - 2, 0):s + 1].any(0) | f[s + 1:s + 4].any(0) for s in st])
    t_n = a["t_n"].astype(float)[:, 0]
    t_sec = a["t_sec"].astype(int)[:, 0]
    t_gap = a["t_gap"].astype(float)[:, 0]
    rng = a["u_range"].astype(float)
    lord = a["u_lord"].astype(bool)
    large = a["u_large"].astype(bool)
    own_ai = str(a["own_ai"])
    cols = {
        "on": on, "g_loss": g_loss, "s_loss": s_loss, "g_mel": g_mel, "s_mel": s_mel, "s_mis": s_mis,
        "g_rout": g_rout, "s_rout": s_rout, "g_rally": g_rally, "s_rally": s_rally, "stand": stand_g,
        "g_dmp": g_dmp, "s_dmp": s_dmp, "s_mp_at": s_mp_at,
        "hp": g0, "mp": mp[st], "ms": ms[st], "r": r0, "m": m[st], "s": a["g_s"].astype(bool)[st],
        "att": att, "shot_at": shot_at, "contact_s": contact_s, "begins": begins, "ran": ran,
        "t_n": t_n, "t_sec": t_sec, "t_gap": t_gap,
        "waver_s": a["c_waver_s"].astype(float), "rout_count": a["c_rout_count"].astype(float),
        "rally_s": a["c_rally_s"].astype(float), "charge": a["c_charge"].astype(float),
        "recent": a["c_recent"].astype(float) / hp0[None, :], "extended": a["c_extended"].astype(float) / hp0[None, :],
        "lf": a["g_lf"].astype(bool)[st] | a["g_rf"].astype(bool)[st], "bf": a["g_bf"].astype(bool)[st],
    }
    if "sim_dl" in a.files and "g_dl" in a.files:
        # HP dealt in melee over the window (absolute; the game's by stepclock's attribution) and the unit's recorded
        # order at the start (tools/nn/sim/orders.py kinds)
        gdl = a["g_dl"].astype(float)
        cols["g_dlt"] = np.stack([gdl[s + 1:s + H + 1].sum(0) for s in st])
        cols["s_dlt"] = a["sim_dl"].astype(float).sum(1).mean(1)
        cols["okind"] = a["o_kind"].astype(int)[st]
    for k in range(H):
        cols[f"g_cum{k + 1}"] = g_cum[:, k]
        cols[f"s_cum{k + 1}"] = s_cum[:, k]
    # the loss after the window's first SETTLE_S seconds (the reset's own transient left out: stepreplay.py)
    k0 = min(SETTLE_S, H - 1)
    cols["g_late"] = g_cum[:, -1] - g_cum[:, k0 - 1]
    cols["s_late"] = s_cum[:, -1] - s_cum[:, k0 - 1]
    for q in range(a["terms1"].shape[0]):
        cols[f"term1_{q}"] = a["terms1"][q].astype(float)
    bcast = {"side": side, "lord": lord, "missile": rng > 0, "large": large, "net": (side == 1) & (own_ai == "net")}
    for k, v in bcast.items():
        cols[k] = np.broadcast_to(v[None, :], (W, n))
    keep = on.reshape(-1)
    out = {k: np.asarray(v).reshape(-1)[keep] for k, v in cols.items()}
    out["battle"] = np.full(keep.sum(), Path(path).stem, dtype=object)
    out["src"] = np.full(keep.sum(), src, dtype=object)
    out["H"] = H
    return out


def load_dir(d, src=""):
    """Every npz of a directory -> one table (rows concatenated)."""
    parts = [p for p in (load(f, src) for f in sorted(Path(d).glob("*.npz"))) if p is not None]
    H = parts[0]["H"]
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != "H"}
    out["H"] = H
    return out


def ratio_ci(sim, game, battle, mask, boot=BOOT, seed=SEED):
    """sim / game of sums over mask, its 95 % interval by battle resampling; (sum sim, sum game, ratio, lo, hi, rows,
    battles)."""
    mask = np.asarray(mask, bool)
    b = battle[mask]
    if not len(b):
        return (0.0, 0.0, np.nan, np.nan, np.nan, 0, 0)
    ids, inv = np.unique(b, return_inverse=True)
    S = np.bincount(inv, sim[mask], len(ids))
    G = np.bincount(inv, game[mask], len(ids))
    ratio = S.sum() / G.sum() if G.sum() > 0 else np.nan
    rs = np.random.default_rng(seed)
    wts = rs.multinomial(len(ids), np.full(len(ids), 1.0 / len(ids)), size=boot)
    gs = wts @ G
    ss = wts @ S
    with np.errstate(divide="ignore", invalid="ignore"):
        rr = np.where(gs > 0, ss / gs, np.nan)
    lo, hi = np.nanpercentile(rr, [2.5, 97.5]) if np.isfinite(rr).any() else (np.nan, np.nan)
    return (float(S.sum()), float(G.sum()), float(ratio), float(lo), float(hi), int(mask.sum()), len(ids))


def mean_ci(x, battle, mask, boot=BOOT, seed=SEED):
    """Mean of x over mask and its 95 % interval by battle resampling: (mean, lo, hi, rows)."""
    mask = np.asarray(mask, bool) & np.isfinite(x)
    if not mask.any():
        return (np.nan, np.nan, np.nan, 0)
    ids, inv = np.unique(battle[mask], return_inverse=True)
    S = np.bincount(inv, x[mask], len(ids))
    N = np.bincount(inv, None, len(ids)).astype(float)
    rs = np.random.default_rng(seed)
    wts = rs.multinomial(len(ids), np.full(len(ids), 1.0 / len(ids)), size=boot)
    mm = (wts @ S) / np.maximum(wts @ N, 1e-9)
    lo, hi = np.percentile(mm, [2.5, 97.5])
    return (float(S.sum() / N.sum()), float(lo), float(hi), int(mask.sum()))


def situations(t):
    """{name: mask} the situations at the window start (rows of a table from load / load_dir). Melee: the game's flag
    on, standing; its contact count and the side it is touched from by the simulator's contact test at the game's places
    (t_n, t_sec)."""
    stand = ~t["r"]
    mel = t["m"] & stand
    out = {
        "all on the field": np.ones(len(t["r"]), bool),
        "standing, not in melee, no contact in window": stand & ~t["m"] & ~t["begins"] & ~t["shot_at"],
        "standing, shot at (an enemy shooting it)": stand & ~t["m"] & t["shot_at"],
        "contact begins in window, ran in": t["begins"] & t["ran"],
        "contact begins in window, walked in": t["begins"] & ~t["ran"],
        "melee, first 10 s of the fight": mel & (t["contact_s"] <= 10),
        "melee, 10-30 s": mel & (t["contact_s"] > 10) & (t["contact_s"] <= 30),
        "melee, 30 s+": mel & (t["contact_s"] > 30),
        "melee, sim touches 1 enemy": mel & (t["t_n"] == 1),
        "melee, sim touches 2+ enemies": mel & (t["t_n"] >= 2),
        "melee, sim touches nobody (game flag on)": mel & (t["t_n"] == 0),
        "melee, touched in front": mel & (t["t_n"] > 0) & (t["t_sec"] == 0),
        "melee, touched in flank": mel & (t["t_n"] > 0) & (t["t_sec"] == 1),
        "melee, touched in rear": mel & (t["t_n"] > 0) & (t["t_sec"] == 2),
        "melee, health < 30 %": mel & (t["hp"] < 0.3),
        "melee, health >= 30 %": mel & (t["hp"] >= 0.3),
        "routing, in melee (pursued)": t["r"] & t["m"],
        "routing, free": t["r"] & ~t["m"],
        "standing, health < 30 %": stand & (t["hp"] < 0.3),
    }
    return out


def units(t):
    """{name: mask} the unit kinds and sides."""
    lord, mis, large = t["lord"], t["missile"], t["large"]
    return {"lord": lord, "missile": mis & ~lord, "large (not lord)": large & ~lord & ~mis,
            "infantry melee": ~lord & ~mis & ~large, "network's side": t["net"], "the other side": ~t["net"]}


def morale_bands(t):
    """{name: mask} the game's morale state at the start (standing units)."""
    ms, mp = t["ms"], t["mp"]
    stand = ~t["r"]
    return {"eager / confident (ms 1-2)": stand & (ms <= 2), "steady (3)": stand & (ms == 3),
            "shaken (4)": stand & (ms == 4), "wavering (5)": stand & (ms == 5),
            "mp < 0.25": stand & (mp < 0.25), "mp 0.25-0.5": stand & (mp >= 0.25) & (mp < 0.5),
            "mp 0.5-0.75": stand & (mp >= 0.5) & (mp < 0.75), "mp >= 0.75": stand & (mp >= 0.75)}


def fmt(x, nd=2):
    return "-" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def loss_table(t, groups, title):
    rows = [f"### {title}", "", "| situation | unit-windows | battles | game HP lost | sim HP lost | sim/game | 95 % CI |"
            f" after {SETTLE_S} s: sim/game | 95 % CI | share of the total deficit |", "|---|---|---|---|---|---|---|---|---|---|"]
    total_def = (t["s_loss"] - t["g_loss"]).sum()
    for name, mask in groups.items():
        s, g, r, lo, hi, n, nb = ratio_ci(t["s_loss"], t["g_loss"], t["battle"], mask)
        if n == 0:
            continue
        _, _, r2, lo2, hi2, _, _ = ratio_ci(t["s_late"], t["g_late"], t["battle"], mask)
        share = (s - g) / total_def if total_def else np.nan
        rows.append(f"| {name} | {n} | {nb} | {g:.1f} | {s:.1f} | {fmt(r)} | {fmt(lo)}-{fmt(hi)} | {fmt(r2)} | "
                    f"{fmt(lo2)}-{fmt(hi2)} | {fmt(share)} |")
    return rows + [""]


def rout_table(t, groups, title, field="rout"):
    stand = ~t["r"] if field == "rout" else t["r"]
    rows = [f"### {title}", "", f"| situation | unit-windows | battles | game {field}s | sim {field}s | sim/game | 95 % CI |",
            "|---|---|---|---|---|---|---|"]
    for name, mask in groups.items():
        mk = mask & stand
        s, g, r, lo, hi, n, nb = ratio_ci(t[f"s_{field}"], t[f"g_{field}"].astype(float), t["battle"], mk)
        if n == 0 or g + s < 3:
            continue
        rows.append(f"| {name} | {n} | {nb} | {g:.0f} | {s:.1f} | {fmt(r)} | {fmt(lo)}-{fmt(hi)} |")
    return rows + [""]


def morale_table(t, groups, title):
    rows = [f"### {title}", "", "| situation | unit-windows | game dMP | sim dMP | sim - game | 95 % CI |", "|---|---|---|---|---|---|"]
    for name, mask in groups.items():
        mk = mask & t["stand"] & np.isfinite(t["g_dmp"]) & np.isfinite(t["s_dmp"])
        if mk.sum() < 20:
            continue
        g = float(np.mean(t["g_dmp"][mk]))
        s = float(np.mean(t["s_dmp"][mk]))
        d, lo, hi, n = mean_ci(t["s_dmp"] - t["g_dmp"], t["battle"], mk)
        rows.append(f"| {name} | {n} | {g:+.3f} | {s:+.3f} | {d:+.3f} | {lo:+.3f} - {hi:+.3f} |")
    return rows + [""]


def curve_lines(t, mask, H):
    """The cumulative loss by second of the window (game, sim, ratio) over mask."""
    g = [t[f"g_cum{k}"][mask].sum() for k in range(1, H + 1)]
    s = [t[f"s_cum{k}"][mask].sum() for k in range(1, H + 1)]
    return "| s | " + " | ".join(str(k) for k in range(1, H + 1)) + " |", \
        "| sim/game | " + " | ".join(fmt(b / a if a else np.nan) for a, b in zip(g, s)) + " |"


def dealt_table(t):
    """HP dealt in melee by the dealer's recorded order at the window start (standing units in melee): the game's
    is stepclock's attribution (the named attacker, else the nearest enemy in melee) - an estimate."""
    names = {0: "hold", 1: "move", 2: "attack", 3: "withdraw"}
    rows = ["### HP dealt in melee by the dealer's recorded order (the game's: attributed, an estimate)", "",
            "| dealer | unit-windows | battles | game HP dealt | sim HP dealt | sim/game | 95 % CI |", "|---|---|---|---|---|---|---|"]
    mel = t["m"] & ~t["r"]
    groups = {f"{nm}, {kind}": mel & (t["okind"] == k) & mk for k, nm in names.items()
              for kind, mk in (("formation", ~t["lord"] & ~t["missile"]), ("missile unit", t["missile"] & ~t["lord"]),
                               ("lord", t["lord"]))}
    for name, mask in groups.items():
        s_, g, r, lo, hi, n, nb = ratio_ci(t["s_dlt"], t["g_dlt"], t["battle"], mask)
        if n < 30:
            continue
        rows.append(f"| {name} | {n} | {nb} | {g:.0f} | {s_:.0f} | {fmt(r)} | {fmt(lo)}-{fmt(hi)} |")
    return rows + [""]


def report(t, title):
    """Markdown lines: the loss, rout, rally and morale tables of one table t."""
    H = t["H"]
    L = [f"## {title}", ""]
    tot = ratio_ci(t["s_loss"], t["g_loss"], t["battle"], np.ones(len(t["r"]), bool))
    rt = ratio_ci(t["s_rout"], t["g_rout"].astype(float), t["battle"], ~t["r"])
    L += [f"Windows of {H} s; {len(set(t['battle']))} battles, {len(t['r'])} unit-windows. HP lost sim/game "
          f"{fmt(tot[2])} ({fmt(tot[3])}-{fmt(tot[4])}); rout starts sim/game {fmt(rt[2])} ({fmt(rt[3])}-{fmt(rt[4])}), "
          f"game {rt[1]:.0f}, sim {rt[0]:.1f}.", ""]
    mel = t["m"] & ~t["r"]
    a, b = curve_lines(t, mel, H)
    L += ["Melee units, the loss by second of the window (sim/game of the cumulative loss):", "", a,
          "|" + "---|" * (H + 1), b, ""]
    sit = situations(t)
    L += loss_table(t, sit, "HP lost by situation")
    L += loss_table(t, units(t), "HP lost by unit kind")
    mu = {f"{k}, {kk}": v & vv for k, v in units(t).items() if k in ("lord", "missile", "infantry melee", "large (not lord)")
          for kk, vv in (("melee", t["m"] & ~t["r"]), ("routing", t["r"]))}
    L += loss_table(t, mu, "HP lost by unit kind and state")
    L += rout_table(t, morale_bands(t), "Rout starts by the game's morale at the start")
    L += rout_table(t, {**sit, **units(t)}, "Rout starts by situation and unit kind")
    L += rout_table(t, {"all routers": np.ones(len(t["r"]), bool), "routers in melee": t["m"], "routers free": ~t["m"]},
                    "Rallies (routing at the start)", field="rally")
    L += morale_table(t, {**sit, **morale_bands(t)}, "Morale change over the window (standing through it in the game)")
    if "okind" in t:
        L += dealt_table(t)
    return L


def subset(t, keep):
    """The rows of table t where keep (a mask over rows)."""
    return {k: (v[keep] if isinstance(v, np.ndarray) and v.shape[:1] == keep.shape else v) for k, v in t.items()}


def main(argv=None):
    import json
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--md", default=None)
    ap.add_argument("--group", action="append", default=[],
                    help="NAME=FILE: a json with a 'runs' list reported apart (the rest: 'other battles')")
    a = ap.parse_args(argv)
    L = []
    for d in a.dirs:
        t = load_dir(d)
        if not a.group:
            L += report(t, f"{d}")
            continue
        rest = np.ones(len(t["battle"]), bool)
        for item in a.group:
            name, path = item.split("=", 1)
            runs = list(json.loads(Path(path).read_text(encoding="utf-8"))["runs"])
            keep = np.isin(t["battle"].astype(str), runs)
            rest &= ~keep
            if keep.any():
                L += report(subset(t, keep), f"{d}: {name}")
        if rest.any():
            L += report(subset(t, rest), f"{d}: other battles")
    text = "\n".join(L)
    if a.md:
        Path(a.md).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

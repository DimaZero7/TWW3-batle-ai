"""The step replay's numbers for the rectangle and the lab side by side, each against the game (numpy, no torch).

  python -m tools.nn.fighters.card build/fighters/sr_rect build/fighters/sr_fighters [--out build/fighters/card.json]

Windows (10 s from the recorded state, the simulator's 4 copies 9 s on; tools/nn/sim/stepreplay.py) and units: not a
lord, 40+ men at the start, not routing or shattered in the game nor in a copy through the window. Chosen by the game
alone (the same windows for both models):
  melee window  the game's melee flag on over half of its seconds
  shot window   the melee flag off all through and health lost (projectiles: the battles have no other source)
Numbers (sim / game, 95 % CI by battles bootstrapped):
  men per HP    men lost / HP lost in men's worth (HP lost / HP a man) - 1: every hit kills a man; by the unit's health
                at the window's start (>85 / 60-85 / 40-60 / 20-40 / <20 %), melee and shot windows
  HP by health  HP lost in melee windows by the same health bins (a damaged unit: below 30 %)
  contact       the seconds the game's melee flag is on (not routing): the share with the copies' flag on (and the
                share of the game's off seconds with the copies' on)
  routs         routs begun in the windows"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

J = 9
BINS = [(0.85, 1.01), (0.6, 0.85), (0.4, 0.6), (0.2, 0.4), (0.0, 0.2)]
rng = np.random.default_rng(0)


def load(d):
    """Per battle: dict of record arrays."""
    out = {}
    for f in sorted(glob.glob(str(Path(d) / "*.npz"))):
        z = np.load(f, allow_pickle=True)
        out[Path(f).stem] = {k: z[k] for k in ("starts", "u_men0", "u_lord", "u_hp0", "g_men", "g_hp", "g_r", "g_s",
                                              "g_m", "sim_men", "sim_hp", "sim_r", "sim_s", "sim_gone", "sim_m", "keys")}
    return out


def records(d):
    """[n, 13] per (window, unit): hp0 share, men share, men0, game HP lost (men's worth), game men lost, sim HP lost
    (men's worth, copy mean), sim men lost, game melee share, game rout begun, sim routs begun (copy mean), hp a man,
    two-handed (greatswords), key index; and the contact seconds [m, 3]: game m, sim m share, valid."""
    T = d["g_men"].shape[0]
    s = d["starts"]
    ok_w = s + J < T
    s = s[ok_w]
    if not len(s):
        return np.zeros((0, 13)), np.zeros((0, 2))
    sm, sh, sr, ss, sg, smm = (d[k][ok_w] for k in ("sim_men", "sim_hp", "sim_r", "sim_s", "sim_gone", "sim_m"))
    idx = np.clip(s[:, None] + np.arange(J + 1)[None, :], 0, T - 1)
    g_men, g_hp = d["g_men"][idx], d["g_hp"][idx]
    g_rs = (d["g_r"][idx] | d["g_s"][idx])
    g_m = d["g_m"][idx]
    men0 = d["u_men0"][None, :].astype(float)
    hp_man = d["u_hp0"] / np.maximum(d["u_men0"], 1)
    gm0, gm1, gh0, gh1 = g_men[:, 0], g_men[:, J], g_hp[:, 0], g_hp[:, J]
    sim_bad = (sr[:, :J] | ss[:, :J] | sg[:, :J]).any(1).any(1)                  # [W, n]
    keep = (~d["u_lord"][None, :]) & (men0 >= 40) & (gm0 > 0) & ~g_rs.any(1) & ~sim_bad
    keep &= np.isfinite(gm0 + gm1 + gh0 + gh1)
    # sim_*[W, H, C, n]: second h ends at row s + h + 1; the window's start is the recorded row s (sim 'hp' at h = -1
    # is the game's): HP lost over rows s .. s + J = game hp0 - sim hp at h = J - 1
    sd_hp = ((gh0[:, None, :] - sh[:, J - 1]) * men0[:, None, :]).mean(1)
    sd_m = (gm0[:, None, :] - sm[:, J - 1]).mean(1)
    two = np.array(["greatsword" in str(k) for k in d["keys"]])[None, :]
    keyi = np.arange(len(d["keys"]))[None, :]
    g_rout = (~g_rs[:, 0]) & g_rs.any(1)
    s_rout = ((sr[:, :J] | ss[:, :J]).any(1)).mean(1)
    rec = np.stack([gh0, gm0 / men0, np.broadcast_to(men0, gh0.shape), (gh0 - gh1) * men0, gm0 - gm1, sd_hp, sd_m,
                    g_m.mean(1), g_rout.astype(float), s_rout, np.broadcast_to(hp_man[None, :], gh0.shape),
                    np.broadcast_to(two, gh0.shape).astype(float), np.broadcast_to(keyi, gh0.shape)], -1)
    # routs: every unit not routing at the start (lords too), whatever its losses
    rr = (~g_rs[:, 0]) & (gm0 > 0) & np.isfinite(gm0)
    rout = np.stack([g_rout[rr].astype(float), s_rout[rr]], -1)
    # contact seconds: rows s+1 .. s+J of the game against the copies' second h = 0 .. J-1
    gm_ = g_m[:, 1:]                                                            # [W, J, n]
    ok_s = ~g_rs[:, 1:] & (g_men[:, 1:] > 0) & ~(sr[:, :J] | ss[:, :J] | sg[:, :J]).any(2)
    con = np.stack([gm_[ok_s].astype(float), smm[:, :J].mean(2)[ok_s]], -1)
    return rec[keep], (con, rout)


def collect(d):
    rows, cons, routs = [], [], []
    for name, b in load(d).items():
        r, (c, ro) = records(b)
        rows.append(r)
        cons.append(c)
        routs.append(ro)
    return rows, cons, routs


def _ratio(rows, sel, num_s, den_s, num_g, den_g, n_boot=300):
    """(sim ratio / game ratio, CI) over battles: (sum num_s / sum den_s) / (sum num_g / sum den_g)."""
    def one(rs):
        a = np.concatenate(rs) if rs else np.zeros((0, 13))
        a = a[sel(a)]
        ds, dg = den_s(a).sum(), den_g(a).sum()
        if ds <= 0 or dg <= 0 or num_g(a).sum() <= 0:
            return np.nan
        return (num_s(a).sum() / ds) / (num_g(a).sum() / dg)
    v = one(rows)
    bs = [one([rows[i] for i in rng.integers(0, len(rows), len(rows))]) for _ in range(n_boot)]
    bs = [x for x in bs if np.isfinite(x)]
    return v, (np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan))


def card(d):
    rows, cons, routs = collect(d)
    out = {}
    melee = lambda a: a[:, 7] > 0.5
    shot = lambda a: (a[:, 7] == 0) & (a[:, 3] > 0)
    for kind, kfn in (("melee", melee), ("shot", shot)):
        for lo, hi in BINS + [(0.0, 1.01)]:
            sel = lambda a, lo=lo, hi=hi, kfn=kfn: kfn(a) & (a[:, 0] >= lo) & (a[:, 0] < hi)
            # men per HP: (sim men / sim HP) / (game men / game HP); HP in men's worth (HP lost x men0 / ... = share x men0)
            eta = _ratio(rows, sel, lambda a: a[:, 6], lambda a: a[:, 5], lambda a: a[:, 4], lambda a: a[:, 3])
            hp = _ratio(rows, sel, lambda a: a[:, 5], lambda a: np.ones(len(a)), lambda a: a[:, 3], lambda a: np.ones(len(a)))
            a = np.concatenate(rows)
            n = int(sel(a).sum())
            g_eta = a[sel(a)][:, 4].sum() / max(a[sel(a)][:, 3].sum(), 1e-9)
            out[f"{kind} {lo:.2f}-{min(hi, 1):.2f}"] = {"n": n, "eta": eta[0], "eta_ci": list(eta[1]), "hp": hp[0],
                                                       "hp_ci": list(hp[1]), "game_eta": g_eta}
    two = lambda a: shot(a) & (a[:, 11] > 0)
    eta = _ratio(rows, two, lambda a: a[:, 6], lambda a: a[:, 5], lambda a: a[:, 4], lambda a: a[:, 3])
    out["shot greatswords"] = {"n": int(two(np.concatenate(rows)).sum()), "eta": eta[0], "eta_ci": list(eta[1])}
    dmg = lambda a: melee(a) & (a[:, 0] < 0.3)
    hp = _ratio(rows, dmg, lambda a: a[:, 5], lambda a: np.ones(len(a)), lambda a: a[:, 3], lambda a: np.ones(len(a)))
    out["melee damaged <30% HP"] = {"n": int(dmg(np.concatenate(rows)).sum()), "hp": hp[0], "hp_ci": list(hp[1])}
    c = np.concatenate(cons)
    on = c[:, 0] > 0.5
    out["contact"] = {"game_on_sim_on": float(c[on, 1].mean()), "game_off_sim_on": float(c[~on, 1].mean()),
                      "seconds": int(on.sum())}
    r = np.concatenate(routs)
    out["routs"] = {"game": float(r[:, 0].sum()), "sim": float(r[:, 1].sum()),
                    "ratio": float(r[:, 1].sum() / max(r[:, 0].sum(), 1e-9))}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dirs", nargs="+", help="step-replay folders (the first: the rectangle, the next: the lab)")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    cards = {Path(d).name: card(d) for d in a.dirs}
    names = list(cards)
    keys = list(cards[names[0]])
    print("sim / game [95 % CI by battles]; " + " | ".join(names))
    for k in keys:
        cells = []
        for nm in names:
            v = cards[nm].get(k, {})
            if "eta" in v:
                s = f"men/HP {v['eta']:.2f} [{v['eta_ci'][0]:.2f}-{v['eta_ci'][1]:.2f}]"
                if "hp" in v:
                    s += f" HP {v['hp']:.2f} [{v['hp_ci'][0]:.2f}-{v['hp_ci'][1]:.2f}]"
                cells.append(s)
            elif "hp" in v:
                cells.append(f"HP {v['hp']:.2f} [{v['hp_ci'][0]:.2f}-{v['hp_ci'][1]:.2f}]")
            elif "game_on_sim_on" in v:
                cells.append(f"on {v['game_on_sim_on']:.3f} off->on {v['game_off_sim_on']:.3f}")
            elif "ratio" in v:
                cells.append(f"{v['sim']:.0f} / {v['game']:.0f} = {v['ratio']:.2f}")
        n = cards[names[0]].get(k, {}).get("n", "")
        game = cards[names[0]].get(k, {}).get("game_eta")
        g = f" game men/HP {game:.3f}" if game is not None else ""
        print(f"  {k:24} n {n}{g} | " + " | ".join(cells))
    if a.out:
        Path(a.out).write_text(json.dumps(cards, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()

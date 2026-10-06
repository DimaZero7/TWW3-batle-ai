"""How well the simulator matches the game, scored as a probabilistic forecast (docs/en/training/simulator.md,
"Match with the game: forecast scores").

Every recorded battle is replayed m times from starts moved by up to 2 m (tools/nn/sim/check.py): the m copies are
the simulator's forecast of a quantity (HP lost by 120 s after contact, the winner, routs, ...), the recording is the
outcome. A case = (battle, quantity): the game's value y and the copies' values x_1..x_m. Only numpy.

coverage   the share of game values inside the copies' central 90 % interval. The interval is read from the
           randomised probability integral transform (PIT): the game's rank among the m copies, ties spread evenly
           (Czado, Gneiting, Held 2009, Biometrics 65:1254). If the game behaves like one more copy (the copies and
           the game exchangeable: the simulator matches the game, noise included), the rank is uniform on 0..m and
           the PIT uniform on (0, 1), so the coverage is 90 % for any m (Hamill 2001, MWR 129:550, rank histograms).
           With m = 19 the interval is exactly the copies' range: a new draw lands outside the range of 19 others
           with probability 2 / 20 (the nonparametric prediction interval of order statistics).
below / above   the share of game values below the copies' 5 % / above their 95 % (5 % each when calibrated):
           which way the simulator is off.
CRPSS      continuous ranked probability skill score 1 - CRPS(sim) / CRPS(reference) per quantity, summed over
           battles (Gneiting & Raftery 2007, JASA 102:359: the CRPS is a strictly proper score, it rewards a right
           centre and an honest spread at once; Hersbach 2000 for ensembles). The CRPS of the m copies is the fair
           estimator (Ferro 2014, QJRMS 140:1917): unbiased for the CRPS of the copies' distribution, so 8 and 19
           copies score alike. For a yes/no quantity the CRPS is the Brier score. Reference: leave-one-out
           climatology - the game's values of the same quantity in the other battles of the family (the forecast
           of someone who knows only the typical game). 100 % = exact, 0 % = no better than the typical game value,
           below 0 = worse. A perfect simulator of a noisy game stays below 100 %: the coverage is the measure of
           correspondence, the CRPSS of usefulness and the guard against copies spread wide to cover the game.
           A family's CRPSS is the median over its quantities (the mean is kept as crpss_mean): a quantity the game
           hardly varies (a share near 1 in every battle) has a reference near 0 and a skill of -700 % or -1300 %
           for a small bias, and such one would decide the mean.
spread     per quantity sqrt((m + 1) / m x mean variance of the copies) / root mean square error of the copies' mean:
           1 when the copies spread as far as the simulator misses the game (Fortin et al. 2014, QJRMS 140:1708,
           "why should ensemble spread match the RMSE of the ensemble mean"); below 1: the copies are too alike
           (the simulator is surer than it should be: coverage drops on both sides).
CIs        95 % percentile bootstrap over battles (Efron & Tibshirani 1993): battles drawn with replacement, every
           quantity of a drawn battle with it (the quantities of one battle are not independent).

    python -m tools.nn.simskill build/nn-sim/check.json            # rescore the cases a check saved (no torch)
    python -m tools.nn.simskill build/nn-sim/check.json --all      # every quantity, not the 10 worst
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

LEVEL = 0.9          # central interval of the coverage
BOOT = 2000          # bootstrap draws
CI = 0.95


def crps(ens, y, fair=True):
    """CRPS of the ensemble `ens` (1-d) for the outcome y: mean |x_i - y| - sum_ij |x_i - x_j| / (2 m (m - 1))
    (fair, Ferro 2014) or / (2 m^2) (the ensemble's own empirical distribution). One member: |x - y|."""
    x = np.asarray(ens, float)
    m = len(x)
    a = np.abs(x - y).mean()
    if m < 2:
        return float(a)
    d = np.abs(x[:, None] - x[None, :]).sum()
    return float(a - d / (2 * m * (m - 1) if fair else 2 * m * m))


def pit(ens, y, level=LEVEL):
    """(inside, below, above): the probabilities that the randomised PIT of y among the copies lies in the central
    `level` interval, below it, above it. Rank r = number of copies below y, plus a uniform share of the copies equal
    to it; the PIT is uniform on [r / (m + 1), (r + 1) / (m + 1)] given r."""
    x = np.asarray(ens, float)
    m = len(x)
    lo_r = int((x < y).sum())
    hi_r = lo_r + int((x == y).sum())            # the rank is uniform on lo_r..hi_r
    a, b = lo_r / (m + 1), (hi_r + 1) / (m + 1)  # the PIT is uniform on [a, b]
    lo, hi = (1 - level) / 2, (1 + level) / 2
    w = b - a
    below = max(0.0, min(b, lo) - a) / w
    above = max(0.0, b - max(a, hi)) / w
    return 1.0 - below - above, below, above


def case(battle, quantity, game, sims):
    """A case for score(): None when the game has no value or fewer than two copies have one."""
    if game is None or not np.isfinite(float(game)):
        return None
    xs = [float(s) for s in sims if s is not None and np.isfinite(float(s))]
    if len(xs) < 2:
        return None
    return {"battle": battle, "q": quantity, "game": float(game), "sims": np.array(xs)}


def score(cases, level=LEVEL):
    """Per case: inside / below / above, the fair CRPS of the copies and of the leave-one-out climatology (None
    when fewer than two other battles have the quantity). Adds the keys in place; returns cases."""
    by_q = {}
    for c in cases:
        by_q.setdefault(c["q"], []).append(c)
    for q, cs in by_q.items():
        ys = np.array([c["game"] for c in cs])
        bs = [c["battle"] for c in cs]
        for c in cs:
            c["inside"], c["below"], c["above"] = pit(c["sims"], c["game"], level)
            c["crps"] = crps(c["sims"], c["game"])
            ref = ys[[b != c["battle"] for b in bs]]
            c["crps_ref"] = crps(ref, c["game"]) if len(ref) >= 2 else None
    return cases


def _skill(cs):
    """Per quantity 1 - sum CRPS / sum CRPS_ref over cases with a reference; their mean over quantities (equal
    weight). A quantity whose reference is 0 (the game's value never varies) has none."""
    num, den = {}, {}
    for c in cs:
        if c["crps_ref"] is None:
            continue
        num[c["q"]] = num.get(c["q"], 0.0) + c["crps"]
        den[c["q"]] = den.get(c["q"], 0.0) + c["crps_ref"]
    per = {q: 1 - num[q] / den[q] for q in num if den[q] > 1e-12}
    return (float(np.median(list(per.values()))) if per else None), per


class _Arrays:
    """The scored cases as arrays, for the bootstrap: a draw is a weight per battle (how often it was drawn)."""

    def __init__(self, cs, battles):
        bi = {b: i for i, b in enumerate(battles)}
        qs = sorted({c["q"] for c in cs})
        qi = {q: i for i, q in enumerate(qs)}
        self.nb, self.nq = len(battles), len(qs)
        self.b = np.array([bi[c["battle"]] for c in cs])
        self.q = np.array([qi[c["q"]] for c in cs])
        self.inside = np.array([c["inside"] for c in cs])
        self.below = np.array([c["below"] for c in cs])
        self.above = np.array([c["above"] for c in cs])
        has = np.array([c["crps_ref"] is not None for c in cs])
        self.crps = np.where(has, [c["crps"] for c in cs], 0.0)
        self.ref = np.array([c["crps_ref"] if c["crps_ref"] is not None else 0.0 for c in cs])

    def stats(self, wb):
        w = wb[self.b]
        n = w.sum()
        num = np.bincount(self.q, w * self.crps, self.nq)
        den = np.bincount(self.q, w * self.ref, self.nq)
        ok = den > 1e-12
        skill = 1 - num[ok] / den[ok]
        return {"coverage": float(w @ self.inside / n), "below": float(w @ self.below / n),
                "above": float(w @ self.above / n),
                "crpss": float(np.median(skill)) if ok.any() else None,
                "crpss_mean": float(np.mean(skill)) if ok.any() else None}


def summary(cases, boot=BOOT, seed=0, ci=CI):
    """The family's numbers: {battles, cases, copies (median), coverage / below / above / crpss: [estimate, lo, hi],
    per_q: [{q, n, game, sim, coverage, below, above, crpss}] worst coverage first}. cases: scored (score())."""
    cs = [c for c in cases if c is not None]
    if not cs:
        return None
    battles = sorted({c["battle"] for c in cs}, key=str)
    arr = _Arrays(cs, battles)
    point = arr.stats(np.ones(arr.nb))
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in point}
    for _ in range(boot):
        s = arr.stats(np.bincount(rng.integers(0, arr.nb, arr.nb), minlength=arr.nb).astype(float))
        for k in point:
            if s[k] is not None:
                draws[k].append(s[k])
    a = (1 - ci) / 2
    out = {"battles": len(battles), "cases": len(cs), "copies": int(np.median([len(c["sims"]) for c in cs]))}
    for k, v in point.items():
        d = draws[k]
        out[k] = [v, float(np.quantile(d, a)) if d else None, float(np.quantile(d, 1 - a)) if d else None]
    _, per = _skill(cs)
    rows = []
    for q in sorted({c["q"] for c in cs}):
        qc = [c for c in cs if c["q"] == q]
        rows.append({"q": q, "n": len(qc), "game": float(np.mean([c["game"] for c in qc])),
                     "sim": float(np.mean([c["sims"].mean() for c in qc])),
                     "coverage": float(np.mean([c["inside"] for c in qc])),
                     "below": float(np.mean([c["below"] for c in qc])),
                     "above": float(np.mean([c["above"] for c in qc])), "crpss": per.get(q),
                     "spread": spread(qc)})
    known = [r["spread"] for r in rows if r["spread"] is not None]
    out["spread_median"] = float(np.median(known)) if known else None
    out["per_q"] = sorted(rows, key=lambda r: (r["coverage"], r["crpss"] if r["crpss"] is not None else 0))
    return out


def spread(cs):
    """The spread-error ratio of one quantity's cases (None when the copies' mean never misses)."""
    var = np.mean([c["sims"].var(ddof=1) * (len(c["sims"]) + 1) / len(c["sims"]) for c in cs])
    mse = np.mean([(c["sims"].mean() - c["game"]) ** 2 for c in cs])
    return float(np.sqrt(var / mse)) if mse > 1e-12 else None


def pct(x):
    return "-" if x is None else f"{100 * x:.0f}"


def lines(name, s, worst=8):
    """Printable lines of a family's summary()."""
    if s is None:
        return [f"{name}: no cases"]
    cov, crp = s["coverage"], s["crpss"]
    out = [f"{name}: {s['battles']} battles, {s['cases']} cases, {s['copies']} copies a battle",
           f"  game inside the sim's 90 % interval: {pct(cov[0])} % (95 % CI {pct(cov[1])}-{pct(cov[2])}; "
           f"a simulator matching the game: 90); below it {pct(s['below'][0])} %, above it {pct(s['above'][0])} % "
           f"(5 / 5 when calibrated)",
           f"  CRPS skill vs the typical game value (median over quantities): {pct(crp[0])} % (95 % CI "
           f"{pct(crp[1])}-{pct(crp[2])}; 100 exact, 0 no better than the typical game value; mean "
           f"{pct(s['crpss_mean'][0])} %)",
           f"  copies' spread / their miss (median over quantities; 1 when calibrated): "
           f"{'-' if s.get('spread_median') is None else format(s['spread_median'], '.2f')}",
           f"  {'quantity':34} {'n':>4} {'game':>9} {'sim':>9} {'inside':>7} {'below':>6} {'above':>6} {'CRPSS':>6} "
           f"{'spread':>6}"]
    for r in s["per_q"][:worst] if worst else s["per_q"]:
        out.append(f"  {r['q']:34} {r['n']:>4} {r['game']:>9.3g} {r['sim']:>9.3g} {pct(r['coverage']):>6}% "
                   f"{pct(r['below']):>5}% {pct(r['above']):>5}% {pct(r['crpss']):>5}% "
                   f"{'-' if r['spread'] is None else format(r['spread'], '.2f'):>6}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("check", type=Path, help="a check.json of tools.nn.sim.check (its 'cases')")
    ap.add_argument("--all", action="store_true", help="every quantity")
    ap.add_argument("--boot", type=int, default=BOOT)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    doc = json.loads(args.check.read_text(encoding="utf-8"))
    for key, rows in (doc.get("cases") or {}).items():
        cases = score([case(r["battle"], r["q"], r["game"], r["sims"]) for r in rows])
        title = (doc.get("forecast") or {}).get(key, {}).get("title", key)
        print(chr(10).join(lines(title, summary(cases, boot=args.boot), worst=0 if args.all else 10)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

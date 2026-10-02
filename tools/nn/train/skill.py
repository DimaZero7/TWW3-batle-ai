"""The network's skill apart from the faction matchup (docs/en/training/training.md "Network
evaluation: fair metrics").

Plain numpy (no torch): tools.nn.gate uses it in the project's .venv.

The factions are unequal (rock-paper-scissors), so a raw win rate mixes the network's skill with the
matchups it happened to get. Measures that take the matchup out:

    pairs(won, pair)            every battle twice on the same armies, the network on side A, then on
                                side B: won both / split / lost both, pair_score = P(both) - P(neither)
    pair_gold(pair, side, destroyed, lost, budget, margin)
                                the continuous side of a pair (most pairs are split, which hides the
                                skill): the gold both pairs of hands got out of the same two armies
    advantage(net, script, ..)  the same battles played by the opponent script against itself (the
                                matchup's natural edge): the network's win rate and gold exchange minus
                                the script's, same armies, same sides, per matchup
    fit(won, own, enemy, attacks, opponent)
                                one rating: P(win) = sigmoid(skill_net - skill_opp + faction_edge(ours,
                                theirs) + role), a ridge logistic (Bradley-Terry) fit; the network's
                                rating per opponent and overall (their mean) with a 95% interval
    margin(winner, side, lost, start)
                                the winner's share of its army's gold left at the end, + when we win,
                                - when we lose: a continuous score
"""
import numpy as np

from tools.nn.train import matchups

Z95 = 1.959964
PRIOR_SD = 3.0          # the ridge: a Gaussian prior (logits) on every parameter, against perfect separation


def pairs(won, pair, void=None):
    """won [n] bool, pair [n] the pair id of each battle (-1: not paired), void [n] bool (no result).
    -> {pairs, won_both, split, lost_both (shares of the complete pairs), pair_score, incomplete}.
    A pair is complete with exactly two battles, both with a result."""
    won, pair = np.asarray(won, dtype=bool), np.asarray(pair)
    void = np.zeros(len(won), bool) if void is None else np.asarray(void, dtype=bool)
    counts = [0, 0, 0]                                   # won both, split, lost both
    incomplete = 0
    for p in np.unique(pair[pair >= 0]) if len(pair) else ():
        sel = pair == p
        if sel.sum() != 2 or void[sel].any():
            incomplete += 1
            continue
        counts[2 - int(won[sel].sum())] += 1
    n = sum(counts)
    out = {"pairs": n, "incomplete": incomplete}
    if n:
        out.update(won_both=counts[0] / n, split=counts[1] / n, lost_both=counts[2] / n,
                   pair_score=(counts[0] - counts[2]) / n)
    else:
        out.update(won_both=None, split=None, lost_both=None, pair_score=None)
    return out


def _mean_ci(x):
    """{value, se, lo, hi} (95%) of the mean of x, or None when empty; se 0 for one value."""
    x = np.asarray(x, dtype=float)
    if not len(x):
        return None
    se = float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
    return _est(x.mean(), se)


def _ratio(num, den):
    return float(num / den) if den > 0 else None


def pair_gold(pair, side, destroyed, lost, budget, margin, void=None):
    """The gold balance of the swapped pairs: in a pair the network plays army A (side 1) in one battle
    and army B in the other, the opponent the other way round, so both pairs of hands get the same two
    armies. [n] each: pair id (-1: none), the side (army) the network played, enemy gold destroyed and
    own gold lost (the network's), budget, the signed margin (ours). A pair counts with exactly two
    battles, sides 1 and 2, both with a result.
    -> {pairs, incomplete,
        pair_gold: mean ± 95% of (destroyed in both - lost in both) / budget (the opponent destroyed what
                   we lost; = our trade with an army minus the opponent's with the same army, either army),
        exchange:  the same as a factor: exp(mean ± 95% of log(our destroyed/lost with an army) - log(the
                   opponent's with it)), per pair the same for both armies (gold floored at 1% budget),
        net_ratio, opp_ratio: all destroyed / all lost by each pair of hands,
        weak, strong: {pairs, trade_net, trade_opp (mean (destroyed - lost) / budget with that army),
                   ratio_net, ratio_opp (its destroyed / lost summed over the pairs)}; the weak army is the
                   one with the lower margin over its two battles (ours with it - ours against it), i.e. the
                   one we did worse with: in a split pair the army that lost both; tied pairs left out}."""
    pair, side = np.asarray(pair), np.asarray(side, dtype=int)
    d, lo, bud, m = (np.asarray(x, dtype=float) for x in (destroyed, lost, budget, margin))
    void = np.zeros(len(pair), bool) if void is None else np.asarray(void, dtype=bool)
    rows, incomplete = [], 0                       # (battle on side 1, battle on side 2) per complete pair
    for p in np.unique(pair[pair >= 0]) if len(pair) else ():
        idx = np.nonzero(pair == p)[0]
        if len(idx) != 2 or void[idx].any() or sorted(side[idx]) != [1, 2] or not np.isfinite(
                np.r_[d[idx], lo[idx], m[idx]]).all():
            incomplete += 1
            continue
        rows.append(tuple(idx[np.argsort(side[idx])]))
    out = {"pairs": len(rows), "incomplete": incomplete}
    if not rows:
        return dict(out, pair_gold=None, exchange=None, net_ratio=None, opp_ratio=None, weak=None, strong=None)
    a, b = (np.array(x) for x in zip(*rows))
    B = (bud[a] + bud[b]) / 2
    out["pair_gold"] = _mean_ci((d[a] + d[b] - lo[a] - lo[b]) / np.maximum(B, 1e-9))
    eps = 0.01 * np.maximum(B, 1e-9)
    log = np.log(d[a] + eps) + np.log(d[b] + eps) - np.log(lo[a] + eps) - np.log(lo[b] + eps)
    e = _mean_ci(log)
    out["exchange"] = {k: float(np.exp(e[k])) for k in ("value", "lo", "hi")}
    out["net_ratio"] = _ratio((d[a] + d[b]).sum(), (lo[a] + lo[b]).sum())
    out["opp_ratio"] = _ratio((lo[a] + lo[b]).sum(), (d[a] + d[b]).sum())
    # weak army: ours in the battle with the lower margin (ties by the trade); the opponent's in the other
    key_a, key_b = np.c_[m[a], (d[a] - lo[a]) / B], np.c_[m[b], (d[b] - lo[b]) / B]
    a_weak = (key_a[:, 0] < key_b[:, 0]) | ((key_a[:, 0] == key_b[:, 0]) & (key_a[:, 1] < key_b[:, 1]))
    tied = (key_a == key_b).all(1)
    for name, ours_is_a in (("weak", a_weak), ("strong", ~a_weak)):
        k = ~tied
        us = np.where(ours_is_a, a, b)[k]                 # our battle with that army
        them = np.where(ours_is_a, b, a)[k]               # the opponent's battle with it
        Bk = B[k]
        out[name] = {"pairs": int(k.sum())}
        if k.any():
            out[name].update(trade_net=float(((d[us] - lo[us]) / Bk).mean()),
                             trade_opp=float(((lo[them] - d[them]) / Bk).mean()),
                             ratio_net=_ratio(d[us].sum(), lo[us].sum()),
                             ratio_opp=_ratio(lo[them].sum(), d[them].sum()))
    return out


def margin(winner, side, lost, start):
    """[n] the signed margin: the winner's gold left / its starting gold (lost, start: [n, 2] gold lost
    and starting cost of sides 1 and 2), + when `side` won, - when it lost, 0 without a winner."""
    winner, side = np.asarray(winner, dtype=int), np.asarray(side, dtype=int)
    lost, start = np.asarray(lost, dtype=float), np.asarray(start, dtype=float)
    n = len(winner)
    w = np.clip(winner, 1, 2) - 1
    left = np.clip(1 - lost[np.arange(n), w] / np.maximum(start[np.arange(n), w], 1e-9), 0, 1)
    sign = np.where(winner == side, 1.0, -1.0)
    return np.where(winner > 0, sign * left, 0.0)


def advantage(net_won, script_won, net_trade, script_trade, own, enemy):
    """The network minus the script on the same battles from the same side: [n] each (won: bool, trade:
    (gold destroyed - gold lost) / budget), own / enemy: our side's factions. -> {"all": cell,
    "matchups": {"EMP-SKV": cell}}, a cell {games, win_net, win_script, win_adv, trade_net, trade_script,
    gold_adv}."""
    nw, sw = np.asarray(net_won, dtype=float), np.asarray(script_won, dtype=float)
    nt, stt = np.asarray(net_trade, dtype=float), np.asarray(script_trade, dtype=float)
    keys = np.array([f"{matchups.short(a)}-{matchups.short(b)}" for a, b in zip(own, enemy)], dtype=object)

    def cell(sel):
        return {"games": int(sel.sum()), "win_net": float(nw[sel].mean()), "win_script": float(sw[sel].mean()),
                "win_adv": float((nw - sw)[sel].mean()), "trade_net": float(nt[sel].mean()),
                "trade_script": float(stt[sel].mean()), "gold_adv": float((nt - stt)[sel].mean())}
    out = {"all": cell(np.ones(len(nw), bool)) if len(nw) else None, "matchups": {}}
    for k in sorted(set(keys), key=_matchup_order):
        out["matchups"][k] = cell(keys == k)
    return out


def _matchup_order(m):
    return tuple(matchups._order(x) for x in m.split("-", 1))


def design(own, enemy, attacks, opponent):
    """(X [n, P], names): one column per opponent (the network's rating against it), one per unordered
    pair of different factions (+1 when ours is the first of the pair, -1 when the second, 0 in a
    mirror: the edge is antisymmetric) and the role (+1 attack, -1 defend)."""
    own = [matchups.short(f) for f in own]
    enemy = [matchups.short(f) for f in enemy]
    opps = list(dict.fromkeys(opponent))
    facs = sorted(set(own) | set(enemy), key=matchups._order)
    met = {frozenset((x, y)) for x, y in zip(own, enemy) if x != y}
    edges = [(a, b) for i, a in enumerate(facs) for b in facs[i + 1:] if frozenset((a, b)) in met]
    n = len(own)
    X = np.zeros((n, len(opps) + len(edges) + 1))
    for i in range(n):
        X[i, opps.index(opponent[i])] = 1.0
        for j, (a, b) in enumerate(edges):
            if (own[i], enemy[i]) == (a, b):
                X[i, len(opps) + j] = 1.0
            elif (own[i], enemy[i]) == (b, a):
                X[i, len(opps) + j] = -1.0
    X[:, -1] = np.where(np.asarray(attacks, dtype=bool), 1.0, -1.0)
    names = [f"rating/{o}" for o in opps] + [f"edge/{a}-{b}" for a, b in edges] + ["attack"]
    return X, names


def logistic(X, y, prior_sd=PRIOR_SD, iters=100):
    """Ridge logistic regression by Newton's method: (theta, covariance) (the inverse Hessian of the
    penalised log-likelihood; the Laplace approximation)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    P = X.shape[1]
    lam = np.eye(P) / prior_sd ** 2
    theta = np.zeros(P)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ theta)))
        g = X.T @ (y - p) - lam @ theta
        H = X.T @ (X * (p * (1 - p))[:, None]) + lam
        step = np.linalg.solve(H, g)
        theta += step
        if np.abs(step).max() < 1e-8:
            break
    p = 1 / (1 + np.exp(-(X @ theta)))
    cov = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]) + lam)
    return theta, cov


def _est(v, se):
    return {"value": float(v), "se": float(se), "lo": float(v - Z95 * se), "hi": float(v + Z95 * se)}


def fit(won, own, enemy, attacks, opponent, void=None, prior_sd=PRIOR_SD):
    """The rating of the network: P(win) = sigmoid(rating[opponent] + edge(ours, theirs) + attack x
    (+1 attack, -1 defend)), rating = skill_net - skill_opponent in logits (0: even with it in a mirror,
    roles balanced). -> {"overall": est, "by_opponent": {name: est}, "faction_edge": {"EMP-SKV": est},
    "attack": est, "battles": n}; est = {value, se, lo, hi} (95%). overall: the mean of the opponents'
    ratings. void [n] bool: battles without a result, left out."""
    won = np.asarray(won, dtype=bool)
    keep = np.ones(len(won), bool) if void is None else ~np.asarray(void, dtype=bool)
    if not keep.any():
        return {"battles": 0}
    sel = np.nonzero(keep)[0]
    X, names = design([own[i] for i in sel], [enemy[i] for i in sel], np.asarray(attacks)[sel],
                      [opponent[i] for i in sel])
    theta, cov = logistic(X, won[sel], prior_sd)
    se = np.sqrt(np.diag(cov))
    out = {"battles": int(len(sel)), "by_opponent": {}, "faction_edge": {}}
    ri = [i for i, n in enumerate(names) if n.startswith("rating/")]
    for i in ri:
        out["by_opponent"][names[i][7:]] = _est(theta[i], se[i])
    for i, n in enumerate(names):
        if n.startswith("edge/"):
            out["faction_edge"][n[5:]] = _est(theta[i], se[i])
    out["attack"] = _est(theta[-1], se[-1])
    w = np.zeros(len(theta))
    w[ri] = 1 / len(ri)
    out["overall"] = _est(w @ theta, np.sqrt(w @ cov @ w))
    return out


# --- report lines ---------------------------------------------------------------------------------

def _f(v, fmt):
    return "-" if v is None else fmt.format(v)


def summary(res):
    """The compact skill block of one evaluation (evaluate.play's result): {"rating": fit(), "pairs":
    {opp: pairs()}, "pair_gold": {opp: pair_gold()}, "advantage": {opp: advantage()}, "margin": {opp: {matchup: mean}}}."""
    out = {"rating": res.get("skill"), "pairs": {}, "pair_gold": {}, "advantage": {}, "margin": {}}
    for opp, o in res.get("by_opponent", {}).items():
        if o.get("pairs"):
            out["pairs"][opp] = o["pairs"]
        if (o.get("pair_gold") or {}).get("pairs"):
            out["pair_gold"][opp] = o["pair_gold"]
        if o.get("baseline"):
            out["advantage"][opp] = o["baseline"]
        if o.get("matchups"):
            out["margin"][opp] = {m: c.get("margin") for m, c in o["matchups"].items()}
            out["margin"][opp]["all"] = o.get("margin")
    return out


def rows(points):
    """[(title, cells)]: the skill block over several summary() results (before / after, minutes of a
    trend), a cell per result."""
    points = [p or {} for p in points]
    out = []

    def rating(p, opp=None):
        r = (p.get("rating") or {})
        e = r.get("overall") if opp is None else r.get("by_opponent", {}).get(opp)
        return "-" if not e else f"{e['value']:+.2f} ± {Z95 * e['se']:.2f}"
    out.append(("rating, overall (logit, ± 95%)", [rating(p) for p in points]))
    opps = list(dict.fromkeys(o for p in points for o in (p.get("rating") or {}).get("by_opponent", {})))
    for o in opps:
        out.append((f"rating, {o}", [rating(p, o) for p in points]))
    edges = list(dict.fromkeys(m for p in points for m in (p.get("rating") or {}).get("faction_edge", {})))
    for m in edges:
        out.append((f"faction edge {m} (fitted)", [_f((p.get("rating") or {}).get("faction_edge", {}).get(m, {})
                                                         .get("value"), "{:+.2f}") for p in points]))
    if any((p.get("rating") or {}).get("attack") for p in points):
        out.append(("attack edge (fitted)", [_f(((p.get("rating") or {}).get("attack") or {}).get("value"), "{:+.2f}")
                                             for p in points]))
    for o in dict.fromkeys(o for p in points for o in p.get("pairs", {})):
        last = next(p["pairs"][o] for p in reversed(points) if o in p.get("pairs", {}))

        def cell(x):
            if not x or x.get("pair_score") is None:
                return "-"
            return f"{x['pair_score']:+.3f} ({x['won_both']:.2f}/{x['split']:.2f}/{x['lost_both']:.2f})"
        out.append((f"pair score (both/split/neither), {o} (n {last['pairs']})",
                    [cell(p.get("pairs", {}).get(o)) for p in points]))
    out += gold_rows(points)
    for o in dict.fromkeys(o for p in points for o in p.get("advantage", {})):
        keys = ["all"] + list(dict.fromkeys(m for p in points for m in p.get("advantage", {}).get(o, {})
                                            .get("matchups", {})))
        for m in keys:
            def cell(p):
                a = p.get("advantage", {}).get(o) or {}
                c = a.get("all") if m == "all" else a.get("matchups", {}).get(m)
                return "-" if not c else f"{c['win_adv']:+.3f} / {c['gold_adv']:+.3f}"
            last = next((p["advantage"][o]["all"] if m == "all" else p["advantage"][o]["matchups"].get(m))
                        for p in reversed(points) if o in p.get("advantage", {}))
            out.append((f"vs script win / gold adv, {o} {m} (n {last['games'] if last else 0})",
                        [cell(p) for p in points]))
    for o in dict.fromkeys(o for p in points for o in p.get("margin", {})):
        ms = ["all"] + [m for m in dict.fromkeys(m for p in points for m in p.get("margin", {}).get(o, {}))
                        if m != "all"]
        out.append((f"margin, {o} " + " / ".join(ms),
                    [" / ".join(_f(p.get("margin", {}).get(o, {}).get(m), "{:+.3f}") for m in ms) for p in points]))
    return out


def gold_cells(g):
    """(pair gold, exchange, weak army) cells of one pair_gold() result ("-" when missing)."""
    if not g or not g.get("pairs"):
        return "-", "-", "-"
    pg, ex, w = g["pair_gold"], g["exchange"], g.get("weak") or {}
    weak = "-" if not w.get("pairs") else (f"{_f(w.get('ratio_net'), '{:.2f}')} vs {_f(w.get('ratio_opp'), '{:.2f}')} "
                                          f"(trade {w['trade_net']:+.3f} / {w['trade_opp']:+.3f})")
    return (f"{pg['value']:+.3f} ± {Z95 * pg['se']:.3f}",
            f"x{ex['value']:.2f} [{ex['lo']:.2f}, {ex['hi']:.2f}]", weak)


def gold_rows(points):
    """[(title, cells)]: the pair gold balance per opponent over points ({"pair_gold": {opp: ...}})."""
    out = []
    for o in dict.fromkeys(o for p in points for o in p.get("pair_gold", {})):
        last = next(p["pair_gold"][o] for p in reversed(points) if o in p.get("pair_gold", {}))
        cells = [gold_cells(p.get("pair_gold", {}).get(o)) for p in points]
        out.append((f"pair gold (ours - opp's, same armies, / budget, ± 95%), {o} (n {last['pairs']})",
                    [c[0] for c in cells]))
        out.append((f"pair exchange (our destroyed/lost / opp's, same army), {o}", [c[1] for c in cells]))
        out.append((f"weak army destroyed/lost net vs opp (trade), {o} (n {(last.get('weak') or {}).get('pairs', 0)})",
                    [c[2] for c in cells]))
    return out


def text(s):
    """A few lines of one summary() (the evaluation's printout)."""
    return [f"{title}: {cells[0]}" for title, cells in rows([s])]

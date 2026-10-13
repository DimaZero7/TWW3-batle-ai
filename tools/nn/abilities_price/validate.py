"""Layer 3: one generic validator of the ability price (docs/en/training/ability_price.md).

It knows nothing about any bonus: only the brick interface (bonuses.Brick: count, switch_off, its Description), the
trajectory's schema (bonuses.TRACE), the abilities as data (ability.py) and the measured branch prices.

(a) tests built from each brick's description, on random trajectories: condition switched off -> 0; bonus of size 0 ->
    0; bonus x2 -> counter x2 (proportional) or not smaller; a longer active time -> not smaller; >= 0; finite; not
    always 0 where its condition holds;
(b) distributions over real presses, per ability: share of presses priced above 0 (alarm > 95 % or < 5 %), tails
    beyond 3 standard deviations (alarm > 1 %), a brick whose counter is always 0, scale dependence (the price's rank
    correlation with the army's budget and with the battle time against the branch price's: alarm when they differ by
    more than 0.3);
(c) agreement with the open-loop branch prices of the same presses (cross-fitted rates: the rates of the other folds):
    sign agreement of the price net of the recharge cost (both sides less the same cost), Spearman rank correlation;
(d) the rates k: the branch prices (gold) regressed on the counters, one rate per bonus kind, k >= 0 (non-negative
    least squares; debuffs enter with a minus), 95 % intervals by bootstrap over the presses, plain least squares beside;
(e) a validity coefficient per ability V = (2 x sign agreement - 1) / 2 + Spearman / 2, and pass / fail by thresholds
    (sign agreement >= 0.7, Spearman >= 0.4); "no signal" when the branch price's real share (its reliability over the
    copies) is below REL_MIN - then it is noise and agreement cannot be judged.

The recharge cost (ability.py): dynamic by default - per press the best later press of the same ability within its
recharge window on the press's own trajectory (ability.best_later, hindsight, with the fitted rates), regressed on the
situation at the press (situation(): what is seen then, cross-fitted per ability) so it looks at no future; the card
shows the static start (the mean useful net x recharge_s / 60) and the hindsight cost beside it.

A file of a longer horizon (collect.py --horizon 120 --marks 30,60,90) is read at --horizon: the branch price at that
mark, the counters over that many seconds; the later presses use the whole trajectory.

    python -m tools.nn.abilities_price.validate --data build/abilities_price/natural120.npz \
        build/abilities_price/hold120.npz --horizon 60

Only numpy (scipy's nnls when it is there).
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tools.nn.abilities_price import ability as abl
from tools.nn.abilities_price import bonuses as bz

ROOT = Path(__file__).resolve().parents[3]
SIGN_MIN = 0.7
SPEARMAN_MIN = 0.4
SHARE_HI, SHARE_LO = 0.95, 0.05
TAIL_Z, TAIL_MAX = 3.0, 0.01
SCALE_GAP = 0.3
# the branch price's real share below this: it is mostly the battle's noise, and no price can agree with it (the rank
# correlation's ceiling is its square root, 0.5 here): the verdict is "no signal", not pass or fail
REL_MIN = 0.25


# --- (a) tests from the descriptions ---

def _blocks(rng, S1, U, p_on, mean_len):
    """[S1, U] bool: random runs (a two-state chain)."""
    out = np.zeros((S1, U), dtype=bool)
    state = rng.random(U) < p_on
    stay_on, stay_off = 1 - 1 / mean_len, 1 - p_on / (mean_len * max(1e-6, 1 - p_on))
    for s in range(S1):
        flip = rng.random(U) > np.where(state, stay_on, stay_off)
        state = state ^ flip
        out[s] = state
    return out


def random_trace(rng, S=60, U=8, active=None):
    """A random trajectory of the schema bonuses.TRACE (every field) and a mask: the owner (0) and a random half of the
    others get the bonus in seconds 1..active."""
    S1 = S + 1
    alive = np.ones((S1, U), dtype=bool)
    for h in range(U):
        if rng.random() < 0.2:
            alive[rng.integers(1, S1):, h] = False
    tr = {"alive": alive, "r": _blocks(rng, S1, U, 0.1, 5) & alive, "m": _blocks(rng, S1, U, 0.4, 8) & alive,
          "mv": _blocks(rng, S1, U, 0.5, 6) & alive}
    tr["sprint"] = tr["mv"] & ~tr["m"] & _blocks(rng, S1, U, 0.3, 3)
    charge = np.zeros((S1, U))
    for s in range(1, S1):
        charge[s] = np.where(rng.random(U) < 0.05, 1.0, np.clip(charge[s - 1] - 1 / 13, 0, 1))
    tr["charge"] = charge
    mmax = rng.uniform(40, 90, U)
    tr["mmax"] = mmax
    tr["morale"] = mmax[None, :] - np.abs(np.cumsum(rng.normal(0, 3, (S1, U)), 0)) + rng.normal(0, 2, (S1, U))
    tr["fatigue"] = np.clip(rng.uniform(0, 20000, U)[None, :] + np.cumsum(rng.normal(0, 600, (S1, U)), 0), 0, 30000)
    tr["x"] = rng.normal(0, 25, U)[None, :] + np.cumsum(rng.normal(0, 1, (S1, U)), 0)
    tr["z"] = rng.normal(0, 25, U)[None, :] + np.cumsum(rng.normal(0, 1, (S1, U)), 0)
    A = int(active if active is not None else rng.integers(5, S))
    tr["on"] = np.clip(A - np.arange(S1), 0, None).astype(float)
    tr["on"][0] = 0.0
    tr["owner"], tr["dt_s"] = 0, 1.0
    mask = np.zeros((S1, U), dtype=bool)
    who = rng.random(U) < 0.5
    who[0] = True
    mask[1:A + 1] = who[None, :]
    return tr, mask


@dataclass
class Check:
    brick: str
    test: str
    ok: bool
    detail: str = ""


def auto_tests(bricks, seed=0, n=60, tol=1e-6):
    """[Check] of every brick (module doc (a)), from its Description only."""
    rng = np.random.default_rng(seed)
    out = []
    for kind, b in bricks.items():
        d = b.desc
        v = d.example
        fails = {t: [] for t in ("off", "zero", "double", "seconds", "sign", "finite")}
        alive = 0
        for i in range(n):
            tr, mask = random_trace(rng)
            c = b.count(tr, mask, v)
            if not np.isfinite(c):
                fails["finite"].append(i)
                continue
            alive += c > 0
            if c < -tol:
                fails["sign"].append(i)
            if abs(b.count(b.switch_off(tr, mask, v), mask, v)) > tol:
                fails["off"].append(i)
            if abs(b.count(tr, mask, bz.scale(d.how, v, 0.0))) > tol:
                fails["zero"].append(i)
            c2 = b.count(tr, mask, bz.scale(d.how, v, 2.0))
            if d.proportional and abs(c2 - 2 * c) > tol * max(1.0, abs(c)) * 10:
                fails["double"].append(i)
            if not d.proportional and c2 < c - tol:
                fails["double"].append(i)
            last = np.nonzero(mask.any(1))[0]
            if len(last):
                prev = -1.0
                for cut in np.linspace(last[0], last[-1], 6).astype(int):
                    m2 = mask.copy()
                    m2[cut + 1:] = False
                    x = b.count(tr, m2, v)
                    if x < prev - tol:
                        fails["seconds"].append(i)
                        break
                    prev = x
        for t, f in fails.items():
            out.append(Check(kind, t, not f, f"{len(f)} of {n} trajectories" if f else ""))
        out.append(Check(kind, "not dead", alive > 0, f"{alive} of {n} trajectories count > 0"))
    return out


# --- data ---

def load(paths, horizon=None):
    """The presses of collect.py files, concatenated -> {field: array} (+ "source": file index per press). horizon (s):
    "diff" is the branch price at that mark of a longer file (its diff_at), "diff_end" the price at the file's own
    horizon."""
    parts, meta = [], []
    names = None
    keep = ("x", "z", "alive", "r", "m", "mv", "sprint", "charge", "morale", "fatigue", "mmax", "on", "diff", "budget",
            "key", "owner", "side", "pressed", "t", "opponent", "other", "row", "diff_at")
    for i, p in enumerate(paths):
        d = np.load(p)
        names = list(d["key_names"])
        m = json.loads(str(d["meta"]))
        meta.append(m)
        part = {k: d[k] for k in d.files if k in keep}
        part["diff_end"] = part["diff"]
        if horizon is not None and abs(float(m.get("horizon_s", horizon)) - horizon) > 1e-6:
            marks = [float(x) for x in m.get("marks_s", [])]
            if horizon not in marks:
                raise ValueError(f"{p}: horizon {m.get('horizon_s')} s, marks {marks}: no price at {horizon} s")
            part["diff"] = part["diff_at"][:, :, marks.index(horizon)]
        part["source"] = np.full(len(part["diff"]), i)
        parts.append(part)
    common = set.intersection(*(set(p) for p in parts))
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k in common}
    return data, names, meta


def trace(data, p, c, dt_s=1.0, upto=None):
    """The trajectory of press p, copy c (upto: its first upto + 1 samples)."""
    cut = slice(None) if upto is None else slice(0, int(upto) + 1)
    tr = {k: np.asarray(data[k][p, c] if k == "mmax" else data[k][p, c][cut],
                        dtype=bool if k in bz.FLAGS else np.float64)
          for k in ("x", "z", "alive", "r", "m", "mv", "sprint", "charge", "morale", "fatigue", "mmax", "on")}
    tr["owner"], tr["dt_s"] = int(data["owner"][p]), dt_s
    return tr


def counters(data, names, abilities, bricks, dt_s=1.0, upto=None):
    """(kinds, C [P, K] mean over the copies, Ccopy [P, copies, K]) of every press (0 for a kind its ability lacks);
    upto: over the first upto samples."""
    kinds = sorted({k for a in abilities.values() for k in a.kinds()})
    P, copies = data["diff"].shape
    Cc = np.zeros((P, copies, len(kinds)))
    for p in range(P):
        a = abilities.get(names[int(data["key"][p])])
        if a is None:
            continue
        for c in range(copies):
            for k, v in a.counters(trace(data, p, c, dt_s, upto), bricks).items():
                Cc[p, c, kinds.index(k)] = v
    return kinds, Cc.mean(1), Cc


def hindsight_recharge(data, names, abilities, bricks, rates, dt_s=1.0, horizon=60, copies=2, step=5):
    """[P] the dynamic recharge cost seen in hindsight (ability.best_later, mean over the first `copies` copies)."""
    k = {kd: r["k"] for kd, r in rates.items()}
    P, C = data["diff"].shape
    out = np.zeros(P)
    for p in range(P):
        a = abilities.get(names[int(data["key"][p])])
        if a is None:
            continue
        out[p] = np.mean([abl.best_later(a, trace(data, p, c, dt_s), bricks, k, horizon / dt_s, step)
                          for c in range(min(C, copies))])
    return out


SITUATION = ("1", "battle time / 600 s", "owner in melee", "owner moving", "friends within 35 m",
             "friends within 35 m in melee", "own units standing", "own units below the steady edge",
             "own units wavering or routing", "owner fatigue tier")


def situation(data, rules=bz.Rules()):
    """[P, len(SITUATION)] what is seen at the press (sample 0 of copy 0; the presser's side only)."""
    P = len(data["diff"])
    o = data["owner"].astype(int)
    j = np.arange(P)
    g = lambda k: np.asarray(data[k][:, 0, 0])
    alive, r, m, mv = g("alive").astype(bool), g("r").astype(bool), g("m").astype(bool), g("mv").astype(bool)
    x, z, mor = g("x").astype(float), g("z").astype(float), g("morale").astype(float)
    fat = g("fatigue").astype(float)
    d = np.hypot(x - x[j, o][:, None], z - z[j, o][:, None])
    near = alive & ~r & (d <= 35.0)
    near[j, o] = False
    stand = alive & ~r
    return np.stack([np.ones(P), data["t"] / 600.0, m[j, o], mv[j, o], near.sum(1), (near & m).sum(1), stand.sum(1),
                     (stand & (mor < rules.morale_steady)).sum(1), (alive & ((mor < rules.morale_waver) | r)).sum(1),
                     bz.tier(fat[j, o], rules.fatigue_thresholds)], 1).astype(float)


def situation_recharge(X, target, key, folds=5, ridge=1.0, seed=0):
    """[P] the recharge cost the situation at the press predicts: per ability a ridge regression of the hindsight cost
    (target) on X, cross-fitted (each press by the other folds), clipped at 0."""
    out = np.zeros(len(target))
    rng = np.random.default_rng(seed)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    Z[:, 0] = 1.0
    for a in np.unique(key):
        ix = np.nonzero(key == a)[0]
        f = rng.permutation(len(ix)) % folds
        for i in range(folds):
            tr, te = ix[f != i], ix[f == i]
            A = Z[tr]
            reg = ridge * np.eye(A.shape[1])
            reg[0, 0] = 0.0
            beta = np.linalg.solve(A.T @ A + reg, A.T @ target[tr])
            out[te] = np.clip(Z[te] @ beta, 0.0, None)
    return out


# --- statistics ---

def ranks(x):
    """Average ranks (ties share their mean rank)."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x), dtype=float)
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(a, b):
    if len(a) < 3:
        return float("nan")
    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def nnls(X, y):
    try:
        from scipy.optimize import nnls as _nnls
        return _nnls(X, y)[0]
    except ImportError:           # projected gradient (small problems)
        k = np.zeros(X.shape[1])
        L = np.linalg.norm(X, 2) ** 2 + 1e-12
        for _ in range(5000):
            k = np.clip(k - X.T @ (X @ k - y) / L, 0, None)
        return k


def design(C, kinds, bricks):
    """The regression's columns: a debuff's counter with a minus (y = sum buffs k c - sum debuffs k c, k >= 0)."""
    sign = np.array([-1.0 if bricks[k].desc.effect == "debuff" else 1.0 for k in kinds])
    return C * sign[None, :]


def fit_rates(C, y, kinds, bricks, boot=400, seed=0):
    """{kind: {"k", "lo", "hi", "ols", "ols_se", "n"}}: k >= 0 by non-negative least squares on the columns a press has,
    95 % bootstrap intervals; plain least squares and its standard error beside."""
    X = design(C, kinds, bricks)
    used = np.abs(X).sum(0) > 0
    k = np.zeros(len(kinds))
    k[used] = nnls(X[:, used], y)
    rng = np.random.default_rng(seed)
    B = np.zeros((boot, len(kinds)))
    for i in range(boot):
        ix = rng.integers(0, len(y), len(y))
        B[i, used] = nnls(X[ix][:, used], y[ix])
    ols = np.full(len(kinds), np.nan)
    se = np.full(len(kinds), np.nan)
    if used.sum():
        Xu = X[:, used]
        beta, *_ = np.linalg.lstsq(Xu, y, rcond=None)
        resid = y - Xu @ beta
        XtX = np.linalg.pinv(Xu.T @ Xu)
        meat = (Xu * resid[:, None] ** 2).T @ Xu                  # heteroscedasticity-robust (HC0)
        cov = XtX @ meat @ XtX
        ols[used], se[used] = beta, np.sqrt(np.clip(np.diag(cov), 0, None))
    return {kd: {"k": float(k[i]), "lo": float(np.quantile(B[:, i], 0.025)), "hi": float(np.quantile(B[:, i], 0.975)),
                 "ols": float(ols[i]), "ols_se": float(se[i]), "n": int((np.abs(X[:, i]) > 0).sum())}
            for i, kd in enumerate(kinds)}


def crossfit(C, y, kinds, bricks, folds=5, seed=0):
    """[P] each press's predicted net benefit (gold) by the rates fitted on the other folds."""
    X = design(C, kinds, bricks)
    used = np.abs(X).sum(0) > 0
    rng = np.random.default_rng(seed)
    f = rng.permutation(len(y)) % folds
    out = np.zeros(len(y))
    for i in range(folds):
        tr, te = f != i, f == i
        k = np.zeros(len(kinds))
        if used.sum():
            k[used] = nnls(X[tr][:, used], y[tr])
        out[te] = X[te] @ k
    return out


def reliability(d):
    """The branch price's real share of its spread (d [E, copies]): 1 - the copies' noise of the mean / the variance
    (ability_branch.reliability); its square root bounds any rank correlation with it."""
    E, K = d.shape
    if K < 2 or E < 8:
        return float("nan")
    v = float(d.mean(1).var())
    return max(0.0, 1.0 - float((d.var(1, ddof=1) / K).mean()) / v) if v > 0 else 0.0


# --- (b), (c), (e): the card ---

def game_ai_style(data, names, p, passports, sim_abilities):
    """Would the game's AI press this ability now (config/nn/sim.json abilities: its trigger and friends_min)?"""
    key = names[int(data["key"][p])]
    trig = sim_abilities.get("triggers", {}).get(key, sim_abilities.get("default_trigger"))
    o = int(data["owner"][p])
    tr0 = {k: data[k][p, 0, 0] for k in ("m", "alive", "r", "x", "z")}
    if trig != "melee" or not tr0["m"][o]:
        return False
    need = (sim_abilities.get("friends_min") or {}).get(key, 0)
    rng = float(passports[key]["range_m"])
    d = np.hypot(tr0["x"].astype(float) - float(tr0["x"][o]), tr0["z"].astype(float) - float(tr0["z"][o]))
    friends = tr0["alive"] & ~tr0["r"] & (d <= rng)
    friends[o] = False
    return int(friends.sum()) >= need


def card(data, names, abilities, bricks, kinds, C, rates, pred, passports, sim_abilities, recharge=None, others=None):
    """{ability key: row} (module doc (b), (c), (e)) and the alarms. recharge [P]: per-press recharge costs (gold; None:
    the static one); others {name: [P]}: other recharge costs whose share above 0 and sign agreement are shown too."""
    y = data["diff"].mean(1) * data["budget"].mean(1)                     # the branch price, gold
    budget = data["budget"].mean(1)
    key = np.array([names[int(k)] for k in data["key"]])
    X = design(C, kinds, bricks)
    k = np.array([rates[kd]["k"] for kd in kinds])
    net_in = X @ k                                                          # in-sample net benefit
    rows = {}
    for a in sorted(abilities):
        sel = key == a
        if not sel.any():
            continue
        ab = abilities[a]
        net = pred[sel]                                                     # cross-fitted
        static = abl.Recharge.fit({a: net}, abilities).cost(a)
        R = np.full(len(net), static) if recharge is None else recharge[sel]
        price = net - R
        yb = y[sel]
        alarms = []
        share = float((price > 0).mean())
        if share > SHARE_HI or share < SHARE_LO:
            alarms.append(f"share above 0 {share:.2f}")
        z = (price - price.mean()) / (price.std() + 1e-12)
        tail = float((np.abs(z) > TAIL_Z).mean())
        if tail > TAIL_MAX:
            alarms.append(f"tails {tail:.3f}")
        for kd in ab.kinds():
            if not (C[sel, kinds.index(kd)] > 0).any():
                alarms.append(f"{kd} always 0")
        sc = {}
        for name, v in (("budget", budget[sel]), ("time", data["t"][sel])):
            ro, rb = spearman(price, v), spearman(yb, v)
            sc[name] = (ro, rb)
            if np.isfinite(ro) and np.isfinite(rb) and abs(ro - rb) > SCALE_GAP:
                alarms.append(f"scale: rho(price, {name}) {ro:+.2f} vs branch {rb:+.2f}")
        sign = float(((price > 0) == (yb - R > 0)).mean())
        variants = {"static": np.full(len(net), static)}
        variants.update({k: v[sel] for k, v in (others or {}).items()})
        variants = {k: {"recharge_gold": float(v.mean()), "share_price_pos": float((net - v > 0).mean()),
                        "share_branch_pos": float((yb - v > 0).mean()), "sign": float(((net - v > 0) == (yb - v > 0)).mean())}
                    for k, v in variants.items()}
        rho = spearman(net, yb)
        thirds = np.array_split(np.argsort(net, kind="mergesort"), 3)
        by_third = [float(yb[t].mean()) for t in thirds]
        V = (2 * sign - 1) / 2 + (rho if np.isfinite(rho) else 0.0) / 2
        rel = reliability(data["diff"][sel] * data["budget"][sel])
        ok = bool(sign >= SIGN_MIN and np.isfinite(rho) and rho >= SPEARMAN_MIN)
        verdict = "pass" if ok else ("no signal" if not (rel >= REL_MIN) else "FAIL")
        ai = np.array([game_ai_style(data, names, p, passports, sim_abilities) for p in np.nonzero(sel)[0]])
        rows[a] = {"n": int(sel.sum()), "pressed": int(data["pressed"][sel].sum()), "recharge_gold": float(R.mean()),
                   "share_price_pos_branch": float((yb - R > 0).mean()), "variants": variants, "by_third": by_third,
                   "net_mean": float(net.mean()), "net_in_sample_mean": float(net_in[sel].mean()),
                   "branch_mean": float(yb.mean()), "branch_se": float(yb.std() / np.sqrt(len(yb))),
                   "share_price_pos": share, "share_branch_pos": float((yb > 0).mean()), "tail": tail,
                   "sign": sign, "spearman": rho, "rel": rel, "ceiling": float(np.sqrt(max(rel, 0.0))), "V": V,
                   "pass": ok, "verdict": verdict,
                   "scale": sc, "alarms": alarms,
                   "ai_n": int(ai.sum()), "ai_price_pos": float((price[ai] > 0).mean()) if ai.any() else float("nan"),
                   "ai_net_pos": float((net[ai] > 0).mean()) if ai.any() else float("nan"),
                   "ai_branch_less_recharge_pos": float((yb[ai] - R[ai] > 0).mean()) if ai.any() else float("nan"),
                   "ai_branch_pos": float((yb[ai] > 0).mean()) if ai.any() else float("nan"),
                   "nonzero": {kd: float((C[sel, kinds.index(kd)] > 0).mean()) for kd in ab.kinds()},
                   "other": float(data["other"][sel].mean()) if "other" in data else float("nan")}
    return rows


def short(key):
    return key.split("abilities_")[-1]


def markdown(checks, rates, rows, kinds, meta, P, recharge_kind="static", horizon=None):
    L = ["# Ability price: validity card", ""]
    L.append(f"Presses: {P}; price at {horizon:g} s; data: " if horizon else f"Presses: {P}; data: ")
    L[-1] += "; ".join(f"{m.get('ckpt')} hold={m.get('hold')} copies={m.get('copies')} horizon={m.get('horizon_s')} s"
                       for m in meta)
    L += ["", "## (a) Tests from the bricks' descriptions", "", "| brick | " + " | ".join(
        dict.fromkeys(c.test for c in checks)) + " |", "|---|" + "---|" * len(dict.fromkeys(c.test for c in checks))]
    tests = list(dict.fromkeys(c.test for c in checks))
    for b in dict.fromkeys(c.brick for c in checks):
        cells = []
        for t in tests:
            c = next(c for c in checks if c.brick == b and c.test == t)
            cells.append("ok" if c.ok else f"FAIL ({c.detail})")
        L.append(f"| {b} | " + " | ".join(cells) + " |")
    L += ["", "## (d) Rates k, gold per counter unit (non-negative least squares; 95 % bootstrap)", "",
          "| kind | k | 95 % | plain LS +- se | presses with it |", "|---|---|---|---|---|"]
    for kd in kinds:
        r = rates[kd]
        L.append(f"| {kd} | {r['k']:.4g} | {r['lo']:.3g} .. {r['hi']:.3g} | {r['ols']:.3g} +- {r['ols_se']:.2g} | "
                 f"{r['n']} |")
    L += ["", "## (b, c, e) Per ability", "",
          "Recharge: " + RECHARGE_WORDS.get(recharge_kind, recharge_kind) + ". Branch mean by price third: the branch "
          "price's mean (gold) over the presses in the lowest / middle / highest third of the price.", "",
          "| ability | presses (net's) | price > 0 | branch - recharge > 0 | sign agree | Spearman (ceiling) | branch rel "
          "| V | verdict | game-AI style: n, price > 0, branch - recharge > 0 | recharge, gold | mean net / branch, gold "
          "| branch mean by price third | alarms |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a, r in rows.items():
        L.append(f"| {short(a)} | {r['n']} ({r['pressed']}) | {r['share_price_pos']:.2f} | "
                 f"{r['share_price_pos_branch']:.2f} | {r['sign']:.2f} | {r['spearman']:.2f} ({r['ceiling']:.2f}) | "
                 f"{r['rel']:.2f} | {r['V']:.2f} | {r['verdict']} | {r['ai_n']}, {r['ai_price_pos']:.2f}, "
                 f"{r['ai_branch_less_recharge_pos']:.2f} | {r['recharge_gold']:.1f} | {r['net_mean']:.1f} / "
                 f"{r['branch_mean']:.1f} +- {r['branch_se']:.1f} | "
                 + " / ".join(f"{v:.1f}" for v in r["by_third"]) + f" | {'; '.join(r['alarms']) or '-'} |")
    L += ["", "Counter non-zero share per ability: " + "; ".join(
        f"{short(a)}: " + ", ".join(f"{k} {v:.2f}" for k, v in r["nonzero"].items()) for a, r in rows.items())]
    names = list(next(iter(rows.values()))["variants"]) if rows else []
    if names:
        L += ["", "## Recharge cost variants", "",
              "Per variant: mean cost (gold), share of price > 0, share of branch - cost > 0, sign agreement. " +
              " ".join(f"{k}: {RECHARGE_WORDS.get(k, k)}." for k in names), "",
              "| ability | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for a, r in rows.items():
            L.append(f"| {short(a)} | " + " | ".join(
                f"{v['recharge_gold']:.1f}; {v['share_price_pos']:.2f}; {v['share_branch_pos']:.2f}; {v['sign']:.2f}"
                for v in r["variants"].values()) + " |")
    return "\n".join(L) + "\n"


RECHARGE_WORDS = {
    "static": "the static start (the mean useful net benefit x recharge_s / 60)",
    "hindsight": "the best later press of the same ability within its recharge window on this trajectory (hindsight)",
    "situation": "the hindsight cost regressed on the situation at the press (cross-fitted per ability; no future)",
}


def run(paths, out=None, boot=400, seed=0, horizon=None, recharge="situation"):
    params_dir = ROOT / "config" / "nn"
    passports = json.loads((params_dir / "abilities.json").read_text(encoding="utf-8"))["abilities"]
    sim = json.loads((params_dir / "sim.json").read_text(encoding="utf-8"))
    rules = bz.rules(json.loads((params_dir / "game_rules.json").read_text(encoding="utf-8")), sim)
    bricks = bz.bricks(rules)
    abilities = abl.abilities(passports, bricks)
    checks = auto_tests(bricks, seed)
    data, names, meta = load(paths, horizon)
    dt_s = float(meta[0].get("dt_s", 1.0))
    H = float(horizon if horizon is not None else meta[0].get("horizon_s", 60.0))
    upto = int(round(H / dt_s))
    kinds, C, _ = counters(data, names, abilities, bricks, dt_s, upto)
    y = data["diff"].mean(1) * data["budget"].mean(1)
    rates = fit_rates(C, y, kinds, bricks, boot, seed)
    pred = crossfit(C, y, kinds, bricks, seed=seed)
    hind = hindsight_recharge(data, names, abilities, bricks, rates, dt_s, H)
    key = np.array([names[int(k)] for k in data["key"]])
    sit = situation_recharge(situation(data, rules), hind, key, seed=seed)
    costs = {"hindsight": hind, "situation": sit}
    rows = card(data, names, abilities, bricks, kinds, C, rates, pred, passports, sim["abilities"],
                recharge=costs.get(recharge), others=costs)
    md = markdown(checks, rates, rows, kinds, meta, len(y), recharge if recharge in costs else "static", H)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(md, encoding="utf-8")
        Path(out).with_suffix(".json").write_text(json.dumps(
            {"rates": rates, "rows": rows, "checks": [c.__dict__ for c in checks], "horizon_s": H,
             "recharge": recharge}, indent=1, default=float), encoding="utf-8")
    return md, rates, rows, checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["build/abilities_price/natural.npz", "build/abilities_price/hold.npz"])
    ap.add_argument("--out", default="build/abilities_price/card.md")
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizon", type=float, default=None, help="the price's horizon, s (a mark of a longer file)")
    ap.add_argument("--recharge", default="situation", choices=("static", "hindsight", "situation"))
    a = ap.parse_args()
    md, *_ = run(a.data, a.out, a.boot, a.seed, a.horizon, a.recharge)
    print(md)


if __name__ == "__main__":
    main()

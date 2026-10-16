"""The spell lab: the rates of the spell bricks (bonuses.SPELL_KINDS: gold per counter unit) measured in controlled
simulator pairs "cast / no cast", and the spell price checked on them (docs/en/training/ability_price.md, "Spells").

A pair is two copies of one small battle that differ only by one cast: in copy "on" the caster (the Gold Wizard with
Searing Doom, the Warlock Engineer with Warp Lightning: config/nn/pools.json "mages", their passports) is ordered to cast
his spell at the target (Orders.ability SPELL_AT, Orders.cast_target - the network's way) at the press second, in copy
"off" he is not; the same battle id (the simulator's draws: common random numbers), the same scripted orders (open
loop). y = (enemy gold lost - own gold lost) on - off, HORIZON_S after the press (the branch price's trade, in gold);
the counters of the spell's bricks on copy "on"'s trajectory of the caster (collect.fields: the simulator's own
bookkeeping bm_gold_e / bm_gold_f / bm_knock, the side's winds, the spell's timer), through layer 2 (ability.py
Ability.from_spell: its mask, the cast's window) - the very code that prices a cast in training.

The situations (FAMILIES; the target always an enemy unit - the network casts at the unit its target pointer chooses):
    dense    a standing enemy unit 80-190 m off (in range): the spell's best case
    moving   the unit walks or runs across, 80-190 m off: the strike point is fixed start_s after the wind-up
    runner   the unit runs straight away: the strike falls where it was (an empty place)
    melee    the unit fights a friend of the caster: the strike hits the friend too (friendly fire)
    lone     the enemy lord alone (a single entity): little to hit
    far      the unit 230-380 m off with an enemy shooter 30 m before it: the caster walks into range (the simulator's
             approach, the game's: the magic probe); the spells' 200 m are beyond the shooters' reach, so the walk costs
             time, not fire
Each side's lord stands 600 m behind (a battle does not end with its small fight). Nobody fires a lord ability (ai off).

The rates: k by non-negative least squares of y on the counters (validate.fit_rates: a debuff with a minus), 95 % by
bootstrap; the winds' rate (winds_cost) is not measurable in a 60 s pair (the next cast is 38 s away and the winds
bind only at the third): it is derived - a wind point is worth what the casts it pays for net, the mean useful net of
the casts / the spell's cost, times the share of casts made when the winds are short (WindsCost counts only those).
The card (validate's layer 3 on the pairs): per spell and family the casts' mean y and predicted price (cross-fitted),
the sign agreement and the rank correlation, the verdict as validate's (sign >= 0.7 and Spearman >= 0.4: valid).
The static recharge cost of a spell (ability.Recharge: the mean useful net x recharge_s / 60).

The direct price the training uses (tools/nn/train/spell_price.py; the user, 16.10.2026): the cast's enemy gold less the
own (its bombardment, the simulator's bookkeeping) less k_winds x its cost in winds; k_winds - what else the winds would
have bought - is the mean direct gold of a good cast (the "dense" pairs: a dense standing enemy in range) a wind point,
both spells pooled; the card's "direct" column is that price of each situation.

    DOCK_NAME=agent-magic5 bash tools/nn/dock.sh tools.nn.abilities_price.lab_spell --pairs 2048 \
        --out build/abilities_price/spell_lab [--write]       # --write: the rates into config/ability_price.json
"""
import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import numpy as np

from tools.nn.abilities_price import ability as abl
from tools.nn.abilities_price import bonuses as bz
from tools.nn.abilities_price import validate as va

ROOT = Path(__file__).resolve().parents[3]
FAMILIES = ("dense", "moving", "runner", "melee", "lone", "far")
HORIZON_S = 60
LORD_BACK_M = 600.0
SIGN_MIN, SPEARMAN_MIN = 0.7, 0.4          # validate's verdict
SPELL_KINDS = bz.SPELL_KINDS
EMP, SKV = "wh_main_emp_empire", "wh2_main_skv_skaven"


def config():
    """(pools.json, the casters {faction: (key, spell)}, the unit passports with the casters', abilities.json)."""
    from tools.nn.armies import pools as P
    pools = json.loads((ROOT / "config" / "nn" / "pools.json").read_text(encoding="utf-8"))
    units = {**P.mage_passports_of(pools.get("mages") or {}),
             **json.loads((ROOT / "config" / "nn" / "units.json").read_text(encoding="utf-8"))["units"]}
    casters = {f: (k, (units[k].get("spells") or [""])[0]) for f, k in (pools["mages"]["units"]).items()}
    doc = json.loads((ROOT / "config" / "nn" / "abilities.json").read_text(encoding="utf-8"))
    return pools, casters, units, doc


def pool_of(pools, units, faction, missile=None):
    """The faction's pool units (pools.json factions), only those with (True) / without (False) a missile weapon."""
    out = []
    for u in pools["factions"][faction]["units"]:
        p = units.get(u["key"])
        if p is None:
            continue
        if missile is None or bool(p.get("missile")) == missile:
            out.append((u["key"], float(u.get("width") or 30.0)))
    return out


def draw(n, rng, faction, pools, casters, units, families=FAMILIES, id0=0):
    """n scenarios of the caster of `faction` (module doc): plain dicts."""
    other = SKV if faction == EMP else EMP
    caster, spell = casters[faction]
    foes, friends = pool_of(pools, units, other), pool_of(pools, units, faction, missile=False)
    shooters = pool_of(pools, units, other, missile=True)
    out = []
    for i in range(n):
        fam = families[i % len(families)]
        d = float(rng.uniform(230, 380) if fam == "far" else rng.uniform(80, 190))
        y = float(rng.uniform(-40, 40))
        tkey, twidth = foes[rng.integers(len(foes))]
        s = {"id": id0 + i, "family": fam, "faction": faction, "other": other, "caster": caster, "spell": spell,
             "target": tkey, "t_width": twidth, "d": d, "y": y, "press": int(rng.integers(0, 4)), "friend": None,
             "shooter": None, "run": bool(rng.random() < 0.5)}
        if fam == "melee":
            s["friend"] = friends[rng.integers(len(friends))]
            s["press"] = int(rng.integers(6, 12))           # the melee under way
        elif fam == "lone":
            s["target"] = pools["factions"][other]["lord"]["key"]
            s["t_width"] = 5.0
        elif fam == "far":
            s["shooter"] = shooters[rng.integers(len(shooters))]
        out.append(s)
    return out


def army(s):
    """The battle description of a scenario (tools/nn/sim/scenario.py build): side 1 the caster's, side 2 the target's;
    slot order: the lord first, then the caster / the target, then the friend / the shooter."""
    pools = json.loads((ROOT / "config" / "nn" / "pools.json").read_text(encoding="utf-8"))
    l1, l2 = pools["factions"][s["faction"]]["lord"]["key"], pools["factions"][s["other"]]["lord"]["key"]
    d, y = s["d"], s["y"]
    u1 = [{"key": l1, "x": -LORD_BACK_M, "z": 0.0, "b": 90.0, "general": True},
          {"key": s["caster"], "x": 0.0, "z": 0.0, "b": 90.0, "general": False}]
    if s["family"] == "lone":
        u2 = [{"key": s["target"], "x": d, "z": y, "b": 270.0, "general": True}]
    else:
        u2 = [{"key": l2, "x": d + LORD_BACK_M, "z": y, "b": 270.0, "general": True},
              {"key": s["target"], "x": d, "z": y, "b": 270.0, "width": s["t_width"], "general": False}]
    if s["friend"]:
        u1.append({"key": s["friend"][0], "x": d - 12.0, "z": y, "b": 90.0, "width": s["friend"][1], "general": False})
    if s["shooter"]:
        u2.append({"key": s["shooter"][0], "x": d - 30.0, "z": y + 45.0, "b": 270.0, "width": s["shooter"][1],
                   "general": False})
    return {"attacker": 1, "sides": {1: {"faction": s["faction"], "ai": False, "units": u1},
                                     2: {"faction": s["other"], "ai": False, "units": u2}}}


def params_with_casters(params=None):
    """The simulator's parameters with the casters' passports (whatever the pools' share) and magic on."""
    from tools.nn.armies import pools as P
    from tools.nn.sim.params import load
    params = params or load()
    pools = json.loads(P.POOLS.read_text(encoding="utf-8"))
    return dataclasses.replace(params, units={**P.mage_passports_of(pools.get("mages") or {}), **params.units})


def play(scen, params, device, horizon_s=HORIZON_S):
    """The pairs of the scenarios (one faction's caster, slots as army()) -> {field: numpy}: per pair y (gold), y_own,
    y_foe, the trace of the caster (on, sp_gold_e, sp_gold_f, sp_knock, winds [n, S+1]), launched (the cast launched),
    gold_out (the spell's enemy gold over the horizon: a check of the trace)."""
    import torch
    from tools.nn.abilities_price import collect as co
    from tools.nn.sim import battle, orders as O, scenario as sim_scenario, state as S
    from tools.nn.train import reward
    n = len(scen)
    armies = [army(s) for s in scen]
    st = sim_scenario.build(armies + armies, params, device=device, per_side=3)
    H = st.N // 2
    ids = torch.tensor([s["id"] for s in scen], device=device)
    st.u[S.BATTLE_ID] = torch.cat([ids, ids])[:, None].expand(2 * n, st.N).clone()
    rows = torch.arange(n, device=device)
    caster = torch.full((n,), 1, dtype=torch.long, device=device)
    lone = torch.tensor([s["family"] == "lone" for s in scen], device=device)
    target = torch.where(lone, torch.full_like(caster, H), torch.full_like(caster, H + 1))
    friend = torch.tensor([s["friend"] is not None for s in scen], device=device)
    fam = [s["family"] for s in scen]
    press = torch.tensor([s["press"] for s in scen], device=device)
    run = torch.tensor([s["run"] for s in scen], device=device)
    k = max(1, int(round(1.0 / params.dt)))
    Sm = int(horizon_s)
    D = int(press.max()) + Sm
    rec = {f: torch.zeros((n, Sm + 1), device=device) for f in ("on",) + bz.SPELL_FIELDS}
    gold0 = torch.zeros((n, 2, 2), device=device)
    gold1 = torch.zeros((n, 2, 2), device=device)
    spacing = params.sim["formation"]["spacing_m"]
    own = torch.zeros((n, 1), dtype=torch.long, device=device) + 1
    slot = torch.full((n,), O.SPELL_AT, dtype=torch.long, device=device)

    def gold(part):
        g = reward.gold_sides({kk: v[part] for kk, v in st.u.items()})
        return g[:, 0], g[:, 1]                                 # side 1 (the caster's), side 2

    def orders():
        o = O.hold(2 * n, st.N, device)
        u = st.u
        tgt2 = torch.cat([target, target])
        r2 = torch.arange(2 * n, device=device)
        # the target's own orders: hold, walk / run across, run away, fight the friend
        tx, tz = u["x"][r2, tgt2], u["z"][r2, tgt2]
        mv = torch.tensor([f in ("moving", "runner") for f in fam] * 2, device=device)
        away = torch.tensor([f == "runner" for f in fam] * 2, device=device)
        o.kind[r2, tgt2] = torch.where(mv, torch.full_like(tgt2, O.MOVE), o.kind[r2, tgt2])
        o.x[r2, tgt2] = torch.where(away, tx + 400.0, tx)
        o.z[r2, tgt2] = torch.where(away, tz, tz + 300.0)
        o.run[r2, tgt2] = torch.where(away, torch.ones_like(mv), torch.cat([run, run]) & mv)
        fr = torch.cat([friend, friend])
        o.kind[r2, tgt2] = torch.where(fr, torch.full_like(tgt2, O.ATTACK), o.kind[r2, tgt2])
        o.target[r2, tgt2] = torch.where(fr, torch.full_like(tgt2, 2), o.target[r2, tgt2])
        o.kind[r2, torch.full_like(tgt2, 2)] = torch.where(fr, torch.full_like(tgt2, O.ATTACK),
                                                             o.kind[r2, torch.full_like(tgt2, 2)])
        o.target[r2, torch.full_like(tgt2, 2)] = torch.where(fr, tgt2, o.target[r2, torch.full_like(tgt2, 2)])
        # (the shooter holds: it shoots at will, the caster once he walks into its range)
        return o

    for d in range(D + 1):
        j = d - press
        sel = (j >= 0) & (j <= Sm)
        if bool(sel.any()):
            uu = {kk: v[:n] for kk, v in st.u.items()}
            fl = co.fields(uu, own, caster, slot, spacing)
            for f in rec:
                rec[f][rows[sel], j[sel]] = fl[f][sel].float()
            on_o, on_f = gold(slice(0, n))
            off_o, off_f = gold(slice(n, 2 * n))
            g = torch.stack([torch.stack([on_o, on_f], 1), torch.stack([off_o, off_f], 1)], 1)
            at0, atS = j == 0, j == Sm
            gold0[at0] = g[at0]
            gold1[atS] = g[atS]
        if d == D:
            break
        o = orders()
        cast = press == d
        o.ability[rows, caster] = torch.where(cast, slot, torch.full_like(slot, -1))
        o.cast_target[rows, caster] = torch.where(cast, target, torch.full_like(target, -1))
        for s_ in range(k):
            if s_:
                o = O.Orders(kind=torch.full_like(o.kind, O.KEEP), x=o.x, z=o.z, target=o.target, run=o.run)
            battle.step(st, o, params, params.dt)
            st.u["lost_worst"] = reward.track(st.u)
    out = {f: v.cpu().numpy() for f, v in rec.items()}
    g0, g1 = gold0.cpu().numpy(), gold1.cpu().numpy()
    net = lambda g: g[:, :, 1] - g[:, :, 0]                      # [n, copy]: the enemy's loss - the caster side's
    out["y"] = (net(g1)[:, 0] - net(g1)[:, 1]) - (net(g0)[:, 0] - net(g0)[:, 1])
    out["y_own"] = (g1[:, 0, 0] - g0[:, 0, 0]) - (g1[:, 1, 0] - g0[:, 1, 0])
    out["y_foe"] = (g1[:, 0, 1] - g0[:, 0, 1]) - (g1[:, 1, 1] - g0[:, 1, 1])
    out["launched"] = (out["on"][:, 1:] > 0).any(1)
    return out


def trace(data, i, dt_s=1.0):
    """The bonuses.TRACE of pair i's caster (U = 1, owner 0; the spell bricks read only the caster's fields)."""
    S1 = data["on"].shape[1]
    tr = {f: np.asarray(data[f][i], dtype=np.float64) for f in ("on",) + bz.SPELL_FIELDS}
    tr["alive"] = np.ones((S1, 1), dtype=bool)
    tr["owner"], tr["dt_s"] = 0, dt_s
    return tr


def counters(data, spell_key, passport, bricks):
    """[n, len(SPELL_KINDS)] the pairs' counters of the spell's bricks (Ability.from_spell, its masks)."""
    a = abl.Ability.from_spell(spell_key, passport, bricks)
    n = len(data["y"])
    C = np.zeros((n, len(SPELL_KINDS)))
    for i in range(n):
        c = a.counters(trace(data, i), bricks)
        for j, kd in enumerate(SPELL_KINDS):
            C[i, j] = c.get(kd, 0.0)
    return C, a


def card(fams, y, C, pred, launched):
    """{family: {n, launched, y, pred, dmg, friendly, knock, sign, spearman}} and the whole's."""
    out = {}
    fams = np.asarray(fams)
    for f in list(FAMILIES) + ["all"]:
        m = np.ones(len(y), bool) if f == "all" else fams == f
        if not m.any():
            continue
        s = va.spearman(pred[m], y[m]) if m.sum() > 2 else float("nan")
        out[f] = {"n": int(m.sum()), "launched": float(launched[m].mean()), "y": float(y[m].mean()),
                  "y_sd": float(y[m].std()), "pred": float(pred[m].mean()),
                  "dmg": float(C[m, 0].mean()), "friendly": float(C[m, 1].mean()), "knock": float(C[m, 2].mean()),
                  "sign": float((np.sign(pred[m]) == np.sign(y[m])).mean()), "spearman": s}
    return out


def winds_rate(net, mana):
    """The winds' rate (gold a wind point, module doc): the casts' mean useful net (> 0) / the cost (WindsCost counts
    it only for the casts made short of winds)."""
    useful = net[net > 0]
    if not len(useful) or mana <= 0:
        return 0.0
    return float(useful.mean() / mana)


def analyse(path_or_data, doc, boot=400, seed=0):
    """The lab's rates, card and the spells' static recharge cost from saved pairs -> dict."""
    d = dict(np.load(path_or_data, allow_pickle=True)) if isinstance(path_or_data, (str, Path)) else path_or_data
    bricks = bz.bricks()
    spells = doc.get("spells") or {}
    keys = [str(k) for k in d["spell_key"]]
    fams = [str(f) for f in d["family"]]
    out = {"spells": {}, "rates": {}}
    Cs, ys, ks = [], [], []
    by = {}
    for key in sorted(set(keys)):
        ix = np.array([i for i, k in enumerate(keys) if k == key])
        sub = {f: d[f][ix] for f in ("on", "y", "y_own", "y_foe", "launched") + bz.SPELL_FIELDS}
        C, a = counters(sub, key, spells[key], bricks)
        by[key] = (ix, C, a)
        Cs.append(C)
        ys.append(sub["y"])
        ks.append(np.full(len(ix), key))
    C = np.concatenate(Cs)
    y = np.concatenate(ys)
    order = np.concatenate([by[k][0] for k in sorted(by)])
    kinds_m = [k for k in SPELL_KINDS if k != "winds_cost"]
    Cm = C[:, :len(kinds_m)]
    rates = va.fit_rates(Cm, y, kinds_m, bricks, boot=boot, seed=seed)
    pred = va.crossfit(Cm, y, kinds_m, bricks, seed=seed)
    k_vec = np.array([rates[k]["k"] for k in kinds_m])
    net = va.design(Cm, kinds_m, bricks) @ k_vec
    # the winds' rate: derived (module doc) from the casts' net by the rates, per wind point
    mana = np.array([float(spells[k]["mana"]) for k in np.concatenate(ks)])
    scarce = C[:, SPELL_KINDS.index("winds_cost")] > 0
    k_w = winds_rate(net, float(np.median(mana)))
    rates["winds_cost"] = {"k": k_w, "lo": float("nan"), "hi": float("nan"), "ols": float("nan"), "ols_se": float("nan"),
                           "n": int(scarce.sum()), "derived": "mean useful net of the casts / the cost (gold a point)"}
    out["rates"] = rates
    fam_all = np.array(fams)[order]
    launched = np.asarray(d["launched"])[order]
    out["card"] = {"all spells": card(fam_all, y, C, pred, launched)}
    at = 0
    for key in sorted(by):
        ix, Ck, a = by[key]
        m = slice(at, at + len(ix))
        at += len(ix)
        pk = pred[m] - k_w * Ck[:, SPELL_KINDS.index("winds_cost")]
        c = card(fam_all[m], y[m], Ck, pk, launched[m])
        useful = net[m][net[m] > 0]
        rc = float(useful.mean() if len(useful) else 0.0) * a.recharge_s / 60.0
        allc = c["all"]
        valid = allc["sign"] >= SIGN_MIN and (allc["spearman"] or 0) >= SPEARMAN_MIN
        out["spells"][key] = {"card": c, "recharge_static": rc, "valid": bool(valid),
                              "verdict": "valid" if valid else "not valid"}
    out["auto_tests"] = [dataclasses.asdict(t) for t in va.auto_tests({k: bricks[k] for k in SPELL_KINDS})]
    # the direct price (module doc): k_winds from the good casts, the price of each situation
    direct = C[:, 0] - C[:, 1]
    launched_all = launched.astype(bool)
    good = (fam_all == "dense") & launched_all
    k_direct = float((direct[good] / mana[good]).mean()) if good.any() else 0.0
    out["spell_direct"] = {"k_winds": k_direct, "horizon_s": 30.0, "good_casts": int(good.sum()),
                           "why": "k_winds: the mean direct gold (enemy gold less own, the cast's bombardment) a wind "
                                  "point of the spell lab's casts at a dense standing enemy in range ('dense' pairs, "
                                  "both spells): what else the winds would have bought (tools/nn/train/spell_price.py)"}
    at = 0
    for key in sorted(by):
        ix = by[key][0]
        m = slice(at, at + len(ix))
        at += len(ix)
        price = direct[m] - k_direct * mana[m] * launched_all[m]
        out["spells"][key]["direct"] = {f: float(price[fam_all[m] == f].mean()) for f in FAMILIES
                                        if (fam_all[m] == f).any()}
    return out


def write_spec(res, path=None):
    """The spell rates, the valid spells and their static recharge cost into config/ability_price.json (the training's
    bonus price, train_price.py)."""
    from tools.nn.abilities_price import train_price as tp
    path = Path(path or tp.FILE)
    spec = json.loads(path.read_text(encoding="utf-8"))
    for k in SPELL_KINDS:
        spec["rates"][k] = float(res["rates"][k]["k"])
    valid = [k for k, v in res["spells"].items() if v["valid"]]
    spec["valid"] = sorted(set(spec.get("valid") or ()) - set(res["spells"]) | set(valid))
    spec["recharge_static"] = {**(spec.get("recharge_static") or {}),
                               **{k: v["recharge_static"] for k, v in res["spells"].items()}}
    spec["spell_direct"] = res["spell_direct"]
    spec.setdefault("_source", {})["spell_lab"] = res.get("_source", "tools/nn/abilities_price/lab_spell.py")
    path.write_text(json.dumps(spec, indent=1) + "\n", encoding="utf-8", newline="\n")
    return spec


def markdown(res):
    lines = ["# Spell lab", "", "| kind | k | 95 % | n |", "|---|---|---|---|"]
    for k, r in res["rates"].items():
        lines.append(f"| {k} | {r['k']:.3f} | {r['lo']:.3f}..{r['hi']:.3f} | {r['n']} |")
    for key, s in res["spells"].items():
        lines += ["", f"## {key}: {s['verdict']}, static recharge {s['recharge_static']:.1f} gold", "",
                  "| family | n | launched | y gold | sd | price | damage | friendly | knock | sign | Spearman |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for f, c in s["card"].items():
            lines.append(f"| {f} | {c['n']} | {c['launched']:.2f} | {c['y']:.1f} | {c['y_sd']:.1f} | {c['pred']:.1f} | "
                         f"{c['dmg']:.1f} | {c['friendly']:.1f} | {c['knock']:.1f} | {c['sign']:.2f} | "
                         f"{c['spearman']:.2f} |")
    d = res.get("spell_direct")
    if d:
        lines += ["", f"## The direct price: k_winds {d['k_winds']:.2f} gold a wind point ({d['good_casts']} good casts)",
                  "", "| situation | " + " | ".join(res["spells"]) + " |", "|---|" + "---|" * len(res["spells"])]
        for f in FAMILIES:
            lines.append(f"| {f} | " + " | ".join(f"{s['direct'].get(f, float('nan')):.1f}"
                                                   for s in res["spells"].values()) + " |")
    bad = [t for t in res["auto_tests"] if not t["ok"]]
    lines += ["", f"auto tests: {len(res['auto_tests']) - len(bad)} of {len(res['auto_tests'])} pass"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=1024, help="pairs per caster")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--out", default="build/abilities_price/spell_lab")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write", action="store_true", help="the rates into config/ability_price.json")
    ap.add_argument("--analyse", action="store_true", help="only analyse the saved pairs")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    pools, casters, units, doc = config()
    path = out / "pairs.npz"
    t0 = time.time()
    if not a.analyse:
        import torch
        device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
        params = params_with_casters()
        rng = np.random.default_rng(a.seed)
        parts = []
        for fac in sorted(casters):
            scen = draw(a.pairs, rng, fac, pools, casters, units, id0=len(parts) * 10 ** 6)
            for b0 in range(0, len(scen), a.batch):
                sc = scen[b0:b0 + a.batch]
                r = play(sc, params, device)
                r["spell_key"] = np.array([s["spell"] for s in sc])
                r["family"] = np.array([s["family"] for s in sc])
                r["target"] = np.array([s["target"] for s in sc])
                parts.append(r)
                print(f"{fac} {b0 + len(sc)}/{len(scen)} pairs, {time.time() - t0:.0f} s", flush=True)
        data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        np.savez_compressed(path, **data)
    res = analyse(path, doc)
    n_pairs = {k: v["card"]["all"]["n"] for k, v in res["spells"].items()}
    res["_source"] = {"pairs": path.relative_to(ROOT).as_posix(), "pairs_per_spell": n_pairs, "horizon_s": HORIZON_S,
                      "families": list(FAMILIES)}
    (out / "lab.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8", newline="\n")
    (out / "lab.md").write_text(markdown(res), encoding="utf-8", newline="\n")
    print(markdown(res))
    if a.write:
        write_spec(res)
        print("written: config/ability_price.json")


if __name__ == "__main__":
    main()

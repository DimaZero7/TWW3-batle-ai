"""The bonus price as the training uses it (docs/en/training/training.md "The ability head's own signal";
docs/en/training/ability_price.md): one press of a VALID ability (the validity card) priced by what its bonuses did on
the pressed branch's trajectory - layers 1 and 2 with the bonus lab's rates and curves, less the dynamic recharge cost.

The numbers live in config/ability_price.json (outside config/nn: not part of the simulator's version), made by `fit`
from the lab's card and the presses of itS25_12 (natural120 + hold120). The lab's card must be made with the bricks as
they are: the lab.json of commit 3f449b2 was counted with the earlier vigour and morale bricks (vigour 0.37 a second a
tier fresher, ~16x the rate of today's "% x s" counter: foe seeker's price came out ~15x its branch price);
`--relab` counts the same lab pairs (and layer 3) with today's bricks (lab_bricks_now.json):

    rates       the lab's whole-battle rate k of each bonus kind (gold per counter unit)
    curves      vigour: the rate by the presser's fatigue tier at the press (0.008 fresh .. 0.024 exhausted, narrow
                intervals; a bin below 0 counts 0, a bin without pairs the flat rate). Morale stays flat: its curve's
                bins are noise (95 % intervals 1-3 gold a point wide) and change no card (CURVE_KINDS)
    valid       the abilities priced this way (the card with the lab's rates: stand your ground, deadly onslaught,
                foe seeker); any other keeps its branch price
    recharge    per ability a ridge regression of the hindsight recharge cost on the situation at the press
                (validate.SITUATION: what is seen then, no future), clipped at 0. The hindsight cost (recharge_target):
                "next" (the default) - ability.next_useful: the net benefit of the same ability pressed at the first of
                5, 10, .. recharge_s seconds later on the same trajectory where its bonus would have work, less
                credit_gold_s x that second (pressing now brings it back that much earlier; credit_gold_s = the
                ability's mean press benefit in the data / (active_s + recharge_s)); "best" - ability.best_later, the
                best of those seconds (biased up: the maximum of ~18 noisy values, 6 % of stand your ground's presses
                came out > 0); the rates as above with the vigour rate of the presser's tier at the press

    price (gold) = benefit - debuffs - recharge cost; benefit and debuffs the mean over the branch's copies

    python -m tools.nn.abilities_price.train_price --relab build/abilities_price/lab \
        --out build/abilities_price/train_price/lab_bricks_now.json                                          # ~4 min
    python -m tools.nn.abilities_price.train_price --data build/abilities_price/natural120.npz \
        build/abilities_price/hold120.npz --lab build/abilities_price/train_price/lab_bricks_now.json   # ~3 min

Only numpy.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from tools.nn.abilities_price import ability as abl
from tools.nn.abilities_price import bonuses as bz

ROOT = Path(__file__).resolve().parents[3]
FILE = ROOT / "config" / "ability_price.json"
VALID = ("wh_main_character_abilities_deadly_onslaught", "wh_main_character_abilities_foe_seeker",
         "wh_main_character_abilities_stand_your_ground")
MORALE_EDGES = (-200.0, 16.0, 32.0, 48.0, 64.0, 1000.0)       # lab.CURVES["morale"] morale_at_press
TIER_EDGES = (-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5)              # lab.CURVES["vigour"] tier
CURVE_KINDS = ("vigour",)                                      # the kinds priced by their lab curve (module doc)
RIDGE = 1.0
SITUATION_N = 10                                               # len(validate.SITUATION)


def _bin(edges, x, n):
    return int(np.clip(np.searchsorted(np.asarray(edges, float), float(x), side="right") - 1, 0, n - 1))


def situation(tr, t, rules=bz.Rules()):
    """[SITUATION_N] what is seen at the press (sample 0 of a trajectory; t the battle time, s): validate.situation's
    columns for one press."""
    o = int(tr["owner"])
    alive, r, m, mv = (np.asarray(tr[k][0], bool) for k in ("alive", "r", "m", "mv"))
    x, z, mor, fat = (np.asarray(tr[k][0], float) for k in ("x", "z", "morale", "fatigue"))
    d = np.hypot(x - x[o], z - z[o])
    near = alive & ~r & (d <= 35.0)
    near[o] = False
    stand = alive & ~r
    return np.array([1.0, float(t) / 600.0, m[o], mv[o], near.sum(), (near & m).sum(), stand.sum(),
                     (stand & (mor < rules.morale_steady)).sum(), (alive & ((mor < rules.morale_waver) | r)).sum(),
                     bz.tier(fat[o], rules.fatigue_thresholds)], float)


class Pricer:
    """The bonus price of presses (module doc). spec: config/ability_price.json's dict; passports: abilities.json's
    `abilities`; spells: its `spells` (a spell is priced by its own bricks: bonuses.SPELL_KINDS, the spell lab's rates,
    lab_spell.py; its recharge cost the static one, spec "recharge_static")."""

    def __init__(self, spec, passports, rules=bz.Rules(), spells=None):
        self.spec = spec
        self.rules = rules
        self.bricks = bz.bricks(rules)
        self.abilities = abl.abilities(passports, self.bricks, spells=spells)
        self.rates = {k: float(v) for k, v in spec["rates"].items()}
        self.curves = spec.get("curves") or {}
        self.valid = set(spec["valid"]) & set(self.abilities)
        self.recharge = spec.get("recharge") or {}

    def ok(self, key):
        return key in self.valid

    def _vigour_rate(self, tr):
        c = self.curves.get("vigour")
        k = self.rates.get("vigour", 0.0)
        if not c:
            return k
        f = float(np.asarray(tr["fatigue"])[0, int(tr["owner"])])
        return c["rates"][_bin(c["edges"], bz.tier(f, self.rules.fatigue_thresholds), len(c["rates"]))]

    def _morale_gold(self, a, part, tr, mask):
        """The morale part in gold: each affected unit's counter x the rate of its morale bin at the press."""
        c = self.curves.get("morale")
        brick = self.bricks["morale"]
        if not c:
            return self.rates.get("morale", 0.0) * brick.count(tr, mask, part.value)
        tot = 0.0
        for u in np.nonzero(mask.any(0))[0]:
            mu = np.zeros_like(mask)
            mu[:, u] = mask[:, u]
            cu = brick.count(tr, mu, part.value)
            if cu > 0:
                tot += cu * c["rates"][_bin(c["edges"], float(np.asarray(tr["morale"])[0, u]), len(c["rates"]))]
        return tot

    def gold(self, key, tr):
        """(benefit, debuffs, {kind: counter}) in gold of one trajectory of a press of `key`."""
        a = self.abilities[key]
        masks = a.masks(tr)
        ben = deb = 0.0
        counts = {}
        for part in a.parts:
            mask = np.zeros_like(tr["alive"])
            for g in part.groups:
                mask = mask | masks[g]
            brick = self.bricks[part.kind]
            c = brick.count(tr, mask, part.value)
            counts[part.kind] = counts.get(part.kind, 0.0) + c
            if part.kind == "morale":
                g_ = self._morale_gold(a, part, tr, mask)
            elif part.kind == "vigour":
                g_ = self._vigour_rate(tr) * c
            else:
                g_ = self.rates.get(part.kind, 0.0) * c
            if brick.desc.effect == "debuff":
                deb += g_
            else:
                ben += g_
        return ben, deb, counts

    def recharge_cost(self, key, tr, t):
        """The dynamic recharge cost (gold) the situation at the press predicts (0 without a fit for the ability)."""
        r = self.recharge.get(key)
        if not r:
            return float((self.spec.get("recharge_static") or {}).get(key, 0.0))
        x = situation(tr, t, self.rules)
        z = (x - np.asarray(r["mu"])) / np.asarray(r["sd"])
        z[0] = 1.0
        return max(0.0, float(z @ np.asarray(r["beta"])))

    def price(self, key, traces, t):
        """{"benefit", "debuffs", "recharge", "price", "work"} in gold of a press of `key` (a valid one), traces: its
        copies' trajectories (bonuses.TRACE), t: the battle time at the press. work: the copies' mean counters summed
        (0: the bonus did no useful work), "copies": each copy's benefit - debuffs."""
        per = [self.gold(key, tr) for tr in traces]
        net = np.array([b - d for b, d, _ in per])
        rc = self.recharge_cost(key, traces[0], t)
        ben = float(np.mean([b for b, _, _ in per]))
        deb = float(np.mean([d for _, d, _ in per]))
        work = float(np.mean([sum(c.values()) for _, _, c in per]))
        return {"benefit": ben, "debuffs": deb, "recharge": rc, "price": ben - deb - rc, "work": work,
                "copies": net - rc}


def load(path=None, passports=None, rules=None):
    """The Pricer of config/ability_price.json (path), the passports and rules of config/nn."""
    spec = json.loads(Path(path or FILE).read_text(encoding="utf-8"))
    cfg = ROOT / "config" / "nn"
    doc = json.loads((cfg / "abilities.json").read_text(encoding="utf-8"))
    if passports is None:
        passports = doc["abilities"]
    if rules is None:
        sim = json.loads((cfg / "sim.json").read_text(encoding="utf-8"))
        rules = bz.rules(json.loads((cfg / "game_rules.json").read_text(encoding="utf-8")), sim)
    return Pricer(spec, passports, rules, spells=doc.get("spells") or {})


# --- fit: the spec from the lab's card and the presses ---

def lab_spec(lab, kinds=CURVE_KINDS):
    """rates and curves of the lab's card (lab.json): whole-battle rates, the whole-battle curves of the `kinds` among
    morale (by the unit's morale at the press) and vigour (by the presser's tier), a bin below 0 at 0, a bin without
    pairs flat."""
    rates = {k: float(v["k"]) for k, v in lab["rates"].items()}
    curves = {}
    for kind, name, edges in (("morale", "morale_at_press", MORALE_EDGES), ("vigour", "tier", TIER_EDGES)):
        if kind not in kinds:
            continue
        rows = ((lab.get("curves") or {}).get("whole battles") or {}).get(kind, {}).get(name)
        if not rows:
            continue
        by = {float(r["lo_x"]): max(0.0, float(r["k"])) for r in rows if np.isfinite(r["k"])}
        curves[kind] = {"edges": list(edges), "rates": [by.get(float(e), max(0.0, rates[kind])) for e in edges[:-1]],
                        "by": name}
    return rates, curves


def fit(data_paths, lab_path, out=None, horizon=60.0, copies=4, step=5, seed=0, target="next"):
    from tools.nn.abilities_price import validate as va
    lab = json.loads(Path(lab_path).read_text(encoding="utf-8"))
    rates, curves = lab_spec(lab)
    spec = {"rates": rates, "curves": curves, "valid": list(VALID), "recharge_target": target, "recharge": {}}
    cfg = ROOT / "config" / "nn"
    passports = json.loads((cfg / "abilities.json").read_text(encoding="utf-8"))["abilities"]
    sim = json.loads((cfg / "sim.json").read_text(encoding="utf-8"))
    rules = bz.rules(json.loads((cfg / "game_rules.json").read_text(encoding="utf-8")), sim)
    pr = Pricer(spec, passports, rules)
    data, names, meta = va.load(data_paths, horizon)
    dt_s = float(meta[0].get("dt_s", 1.0))
    P, C = data["diff"].shape
    key = np.array([names[int(k)] for k in data["key"]])
    X = va.situation(data, rules)
    own = np.zeros(P)                     # each press's own net benefit over the horizon (mean over the copies)
    for p in range(P):
        if key[p] in pr.valid:
            own[p] = np.mean([np.subtract(*pr.gold(key[p], va.trace(data, p, c, dt_s, upto=horizon / dt_s))[:2])
                              for c in range(C)])
    credit = {a: (float(own[key == a].mean()) / (pr.abilities[a].active_s + pr.abilities[a].recharge_s)
                  if target == "next" and (key == a).any() else 0.0) for a in pr.valid}
    later = abl.TARGETS[target]
    hind = np.zeros(P)
    for p in range(P):
        a = pr.abilities.get(key[p])
        if a is None or key[p] not in pr.valid:
            continue
        vals = []
        for c in range(min(C, copies)):
            tr = va.trace(data, p, c, dt_s)
            k = dict(rates, vigour=pr._vigour_rate(tr))
            vals.append(later(a, tr, pr.bricks, k, horizon / dt_s, step, credit_s=credit[key[p]]))
        hind[p] = np.mean(vals)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    Z[:, 0] = 1.0
    fitted = {}
    for a in sorted(pr.valid):
        ix = np.nonzero(key == a)[0]
        if len(ix) < 20:
            continue
        A = Z[ix]
        reg = RIDGE * np.eye(A.shape[1])
        reg[0, 0] = 0.0
        beta = np.linalg.solve(A.T @ A + reg, A.T @ hind[ix])
        pred = np.clip(A @ beta, 0.0, None)
        spec["recharge"][a] = {"mu": mu.tolist(), "sd": sd.tolist(), "beta": beta.tolist(), "n": int(len(ix)),
                               "hindsight_mean": float(hind[ix].mean()), "predicted_mean": float(pred.mean()),
                               "own_mean": float(own[ix].mean()), "credit_gold_s": credit[a]}
        fitted[a] = (float(hind[ix].mean()), float(pred.mean()), len(ix), float(own[ix].mean()),
                     float((own[ix] - pred > 0).mean()))
    spec["_source"] = {"lab": str(lab_path), "data": [str(p) for p in data_paths], "horizon_s": horizon,
                       "hindsight_copies": min(C, copies), "step_s": step, "ridge": RIDGE, "recharge_target": target,
                       "situation": list(va.SITUATION)}
    if out:
        Path(out).write_text(json.dumps(spec, indent=1) + "\n", encoding="utf-8")
    return spec, fitted


def relab(lab_dir, paths, out, boot=400, seed=0):
    """lab.report's rates, curves and layer-3 card of the lab's pairs in lab_dir counted with today's bricks, written to
    `out` (json) only (lab.report writes into lab_dir) -> the dict."""
    from tools.nn.abilities_price import lab as lb
    pools, units, passports, sim, rules_ = lb.load_config()
    bricks = bz.bricks(bz.rules(rules_, sim))
    rates, fam_rates = {}, {}
    curves = {"whole battles": {}, "small fights": {}}
    for kd in lb.KINDS:
        fams = {fam: lb.load_family(Path(lab_dir) / pat.format(kd), units, bricks) for fam, pat in lb.FAMILY_FILES
                if (Path(lab_dir) / pat.format(kd)).exists()}
        if not fams:
            continue
        fam_rates[kd] = {fam: lb.summarise([fd], boot, seed)[0] for fam, fd in fams.items()}
        main = [fams[f] for f in lb.MAIN if f in fams] or list(fams.values())
        rates[kd], pooled = lb.summarise(main, boot, seed)
        curves["whole battles"][kd] = lb.curves_of(kd, pooled, boot, seed)
        print(kd, round(rates[kd]["k"], 5), round(rates[kd]["lo"], 5), round(rates[kd]["hi"], 5), flush=True)
    l3 = {}
    if paths and all(Path(p).exists() for p in paths):
        sets = {"lab": (rates, None), "lab + curves": (rates, curves["whole battles"])}
        l3, _, _, _ = lb.layer3(paths, sets, bricks, passports, sim, boot=200, seed=seed)
    d = {"rates": rates, "families": fam_rates, "curves": curves, "layer3": l3}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(d, indent=1, default=float), encoding="utf-8")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=None,
                    help="the presses (collect.py): default natural120 + hold120 (fit), natural + hold (--relab)")
    ap.add_argument("--lab", default="build/abilities_price/train_price/lab_bricks_now.json")
    ap.add_argument("--out", default=None, help="default config/ability_price.json (fit)")
    ap.add_argument("--horizon", type=float, default=60.0)
    ap.add_argument("--relab", default=None, metavar="LAB_DIR", help="count the lab's pairs with today's bricks")
    ap.add_argument("--recharge-target", default="next", choices=tuple(abl.TARGETS),
                    help="the hindsight recharge cost: next (the first later second with work, less the credit) or best")
    a = ap.parse_args()
    if a.relab:
        d = relab(a.relab, a.data or ["build/abilities_price/natural.npz", "build/abilities_price/hold.npz"],
                  a.out or "build/abilities_price/train_price/lab_bricks_now.json")
        for way, rows in d["layer3"].items():
            for k, r in rows.items():
                print(way, k.split("abilities_")[-1], r.get("verdict"), round(r["sign"], 2), round(r["spearman"], 2))
        return
    spec, fitted = fit(a.data or ["build/abilities_price/natural120.npz", "build/abilities_price/hold120.npz"], a.lab,
                       a.out or str(FILE), a.horizon, target=a.recharge_target)
    for k, (h, p, n, b, pos) in fitted.items():
        print(f"{k}: {n} presses, benefit {b:.1f} gold, hindsight recharge {h:.1f}, predicted {p:.1f}, "
              f"price > 0 {pos:.2f}")
    print("rates", {k: round(v, 4) for k, v in spec["rates"].items()})
    print("curves", spec["curves"])


if __name__ == "__main__":
    main()

"""The bonus lab's simulator runs (lab.py has the design; docs/en/training/ability_price.md, "Bonus lab"). torch.

A batch of n pairs is one simulator batch of 2n battles: copy "on" of pair i in row i, copy "off" in row n + i, both
with battle id (the state's `row`, the seed of every hash the simulator draws from) id_i. The lab ability is in the
lord's slot 0 in both copies (lab passports added to the parameters' abilities, modelled; his real abilities taken out,
nobody's fired by the game-AI rule); at its press second copy "on" orders it, copy "off" does not. The orders are the
script's on copy "on", replayed in copy "off" (open loop). Each second (a decision: 2 steps of 0.5 s) the trajectory of
the lord's side (collect.fields: bonuses.TRACE) is taken in copy "on" from the press (sample 0, before it acts) to 60 s
after; the gold at 60 s in both copies.

Families "line" and "runup" (lab.draw: a small fight, the lab script) and "battle": a whole battle of the v2 armies
(tools/nn/armies, as training's), both sides played by the ai_like script, the lord of a side that carries the bonus
pressing it at a battle time drawn from the presses' own times; the battle runs once to the press, then is doubled
into the pair.

    DOCK_NAME=agent-bonuslab DOCK_BUILD=<main>/build bash tools/nn/dock.sh tools.nn.abilities_price.lab_sim \
        --pairs 8192 --battle-pairs 3072 --out build/abilities_price/lab
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from tools.nn.abilities_price import bonuses as bz
from tools.nn.abilities_price import collect as co
from tools.nn.abilities_price import lab
from tools.nn.sim import abilities as sab
from tools.nn.sim import battle
from tools.nn.sim import orders as O
from tools.nn.sim import scenario as sim_scenario
from tools.nn.sim import state as S
from tools.nn.sim.params import Params, load
from tools.nn.train import reward

FLOATS = ("x", "z", "charge", "morale", "fatigue")
FLAGS = ("alive", "r", "m", "mv", "sprint")
BATTLE_PER_SIDE = 20          # the lord and 19 units (tools/nn/armies)


def params_with(passports_lab, params=None):
    """The simulator's parameters with the lab abilities added (modelled, never fired by a rule)."""
    p = params or load()
    ab = dict(p.abilities)
    ab.update(passports_lab)
    sim = json.loads(json.dumps(p.sim))
    sim["abilities"]["model"] = list(sim["abilities"]["model"]) + sorted(passports_lab)
    for k in passports_lab:
        sim["abilities"]["triggers"][k] = "never"
    return Params(p.units, p.rules, sim, ab, p.effects)


class Plan:
    """The batch's script and press: per battle row (2n), slot: mode, the target's slot; press second per pair."""

    def __init__(self, scen, st, per_side, device):
        n = len(scen)
        B, N = st.B, st.N
        self.n = n
        mode = torch.full((B, N), lab.RESERVE_HOLD, dtype=torch.long)
        tgt = torch.full((B, N), -1, dtype=torch.long)
        lab_unit = torch.zeros((B, N), dtype=torch.bool)
        for i, s in enumerate(scen):
            slot = sim_scenario.slots(s["army"], per_side)
            for name, m in s["mode"].items():
                for row in (i, n + i):
                    mode[row, slot[name]] = m
                    lab_unit[row, slot[name]] = m != lab.RESERVE_HOLD
                    if name in s["target"]:
                        tgt[row, slot[name]] = slot[s["target"][name]]
        self.mode, self.tgt, self.lab = mode.to(device), tgt.to(device), lab_unit.to(device)
        self.press = torch.tensor([s["press"] for s in scen], device=device)


def script(st, plan):
    """The lab's orders [B, N]: NEAREST attacks the nearest standing lab enemy (running; missile units close and
    shoot), HOLD and the reserve hold, TARGET attacks its target while it stands, then the nearest."""
    u = st.u
    alive = (u["side"] > 0) & (u["men"] > 0) & ~u["gone"]
    standing = alive & ~u["r"]
    dx = u["x"][:, None, :] - u["x"][:, :, None]
    dz = u["z"][:, None, :] - u["z"][:, :, None]
    d = torch.sqrt(dx * dx + dz * dz)
    enemy = (u["side"][:, :, None] != u["side"][:, None, :]) & standing[:, None, :] & plan.lab[:, None, :]
    d = torch.where(enemy, d, torch.full_like(d, 1e9))
    near = d.argmin(2)
    has = d.min(2).values < 1e9
    t_ok = (plan.tgt >= 0) & standing.gather(1, plan.tgt.clamp(min=0))
    want = torch.where((plan.mode == lab.TARGET) & t_ok, plan.tgt, near)
    attack = standing & (((plan.mode == lab.NEAREST) | (plan.mode == lab.TARGET)) & (has | t_ok))
    o = O.hold(st.B, st.N, st.device)
    o.kind = torch.where(attack, torch.full_like(o.kind, O.ATTACK), o.kind)
    o.target = torch.where(attack, want, torch.full_like(want, -1))
    o.run = attack.clone()
    return o


def _lab_slot(u, rows, lord, ab):
    """The lord's slots: the lab ability in slot 0, ready; his real ones out."""
    for k in range(sab.SLOTS):
        u[f"ab{k}"][rows, lord] = -1
    u["ab0"][rows, lord] = ab
    u["ab0_on"][rows, lord] = 0.0
    u["ab0_cd"][rows, lord] = 0.0


def build(scen, params, device, id0=0):
    """The batch state (copies on: rows 0..n-1, off: n..2n-1), the plan, the lab slots set, the start fatigue."""
    armies = [s["army"] for s in scen]
    st = sim_scenario.build(armies + armies, params, device=device, per_side=lab.PER_SIDE)
    n = len(scen)
    names = sab.keys(params)
    u = st.u
    ids = torch.tensor([id0 + s["id"] for s in scen], device=device)
    u[S.BATTLE_ID] = torch.cat([ids, ids])[:, None].expand(2 * n, st.N).clone()
    lord = torch.zeros(2 * n, dtype=torch.long, device=device)              # side 1's lord is slot 0
    ab = torch.tensor([names.index(s["key"]) for s in scen] * 2, device=device)
    _lab_slot(u, torch.arange(2 * n, device=device), lord, ab)
    for i, s in enumerate(scen):
        slot = sim_scenario.slots(s["army"], lab.PER_SIDE)
        for name, f in s["fatigue"].items():
            u["fatigue"][i, slot[name]] = f
            u["fatigue"][n + i, slot[name]] = f
    u["lost_worst"] = torch.zeros_like(u["men"])
    return st, Plan(scen, st, lab.PER_SIDE, device)


def _decision(st, o, params, k):
    """One decision: the orders on its first step, KEEP on the rest; the units' worst loss tracked (reward.track).
    The step is torch.compile'd on the GPU (battle.stepper: ~25x faster; one compile per batch shape)."""
    step = battle.stepper(st.device)
    for s in range(k):
        if s:
            o = O.Orders(kind=torch.full_like(o.kind, O.KEEP), x=o.x, z=o.z, target=o.target, run=o.run,
                         ability=torch.full_like(o.ability, -1))
        step(st, o, params, params.dt)
        st.u["lost_worst"] = reward.track(st.u)


def pairs(st, n, press, own, owner, side, orders, params, horizon_s=lab.HORIZON_S, keep_off=False, trigger=None,
          wait_s=0):
    """The pairs' run (module doc) on st (2n battles: on, then off) -> {field: numpy}: the trace of copy on ([n, S+1,
    H] of the slots own [n, H], the lord owner [n] an absolute slot; on [n, S+1]; mmax [n, H]), y (gold on - off at the
    horizon from side [n]'s view: enemy lost - own lost), y_own / y_foe (own / enemy gold lost, on - off), fired (the
    press fired), y0 (gold on - off at the press: 0 by construction), tbusy, press (the press's decision);
    keep_off: copy off's trace (off_<field>). orders(st) -> Orders of every battle (only copy on's are used).
    trigger(st, n, d) -> [n] bool: instead of the fixed press, each pair presses at the first decision its bonus's
    condition holds (the copies are the same until then), at wait_s at the latest."""
    dev = st.device
    if trigger is not None:
        press = torch.full((n,), 10 ** 6, dtype=torch.long, device=dev)
    H = own.shape[1]
    Sm = int(horizon_s)
    k = max(1, int(round(1.0 / params.dt)))
    spacing = params.sim["formation"]["spacing_m"]
    slot0 = torch.zeros(n, dtype=torch.long, device=dev)
    rec = {}
    copies = (("", slice(0, n)),) + ((("off_", slice(n, 2 * n)),) if keep_off else ())
    for pre, _ in copies:
        for f in FLAGS:
            rec[pre + f] = torch.zeros((n, Sm + 1, H), dtype=torch.bool, device=dev)
        for f in FLOATS:
            rec[pre + f] = torch.zeros((n, Sm + 1, H), device=dev)
        rec[pre + "on"] = torch.zeros((n, Sm + 1), device=dev)
    # (a diagnostic, not a trace field) the lord's attack target is in melee: the simulator then gives no charge
    # sprint and no charge (charge.free_target_only), while the trace's `sprint` (collect.fields) counts the run-in
    rec["tbusy"] = torch.zeros((n, Sm + 1), dtype=torch.bool, device=dev)
    mmax = torch.zeros((n, H), device=dev)
    gold0 = torch.zeros((n, 2, 2), device=dev)           # [pair, copy, own / foe]
    gold1 = torch.zeros((n, 2, 2), device=dev)
    D = int(wait_s) + Sm if trigger is not None else int(press.max()) + Sm
    rows = torch.arange(n, device=dev)
    s_ix = (side - 1).long()[:, None]

    def gold(part):
        g = reward.gold_sides({kk: v[part] for kk, v in st.u.items()})
        return g.gather(1, s_ix)[:, 0], g.gather(1, 1 - s_ix)[:, 0]

    for d in range(D + 1):
        if trigger is not None and d <= wait_s:
            und = press > d
            now = und & (trigger(st, n, d) if d < wait_s else torch.ones_like(und))
            press = torch.where(now, torch.full_like(press, d), press)
        j = d - press                                           # the sample index of each pair
        sel = (j >= 0) & (j <= Sm)
        if bool(sel.any()):
            for pre, part in copies:
                uu = {kk: v[part] for kk, v in st.u.items()}
                fl = co.fields(uu, own, owner, slot0, spacing)
                for f in FLAGS + FLOATS:
                    rec[pre + f][rows[sel], j[sel]] = fl[f][sel].to(rec[pre + f].dtype)
                rec[pre + "on"][rows[sel], j[sel]] = fl["on"][sel]
                if not pre:
                    tg = uu["order_target"].gather(1, owner[:, None])[:, 0]
                    busy = (tg >= 0) & uu["m"].gather(1, tg.clamp(min=0)[:, None])[:, 0]
                    rec["tbusy"][rows[sel], j[sel]] = busy[sel]
            on_o, on_f = gold(slice(0, n))
            off_o, off_f = gold(slice(n, 2 * n))
            at0 = j == 0
            if bool(at0.any()):
                mmax[at0] = (st.u["leadership"] + st.u["morale_bonus"])[:n].gather(1, own)[at0]
                gold0[at0] = torch.stack([torch.stack([on_o, on_f], 1), torch.stack([off_o, off_f], 1)], 1)[at0]
            atS = j == Sm
            if bool(atS.any()):
                gold1[atS] = torch.stack([torch.stack([on_o, on_f], 1), torch.stack([off_o, off_f], 1)], 1)[atS]
        if d == D:
            break
        o = orders(st)
        for f in O.FIELDS:                                       # open loop: copy off plays copy on's orders
            v = getattr(o, f)
            v[n:] = v[:n]
        o.ability.fill_(-1)
        o.ability[rows, owner] = torch.where(press == d, torch.zeros_like(press), torch.full_like(press, -1))
        _decision(st, o, params, k)
    out = {f: v.cpu().numpy() for f, v in rec.items()}
    out["mmax"] = mmax.cpu().numpy()
    g0, g1 = gold0.cpu().numpy(), gold1.cpu().numpy()
    net = lambda g: g[:, :, 1] - g[:, :, 0]                      # [n, copy]: enemy lost - own lost
    out["y"] = (net(g1)[:, 0] - net(g1)[:, 1]) - (net(g0)[:, 0] - net(g0)[:, 1])
    out["y0"] = net(g0)[:, 0] - net(g0)[:, 1]
    out["y_own"] = (g1[:, 0, 0] - g0[:, 0, 0]) - (g1[:, 1, 0] - g0[:, 1, 0])
    out["y_foe"] = (g1[:, 0, 1] - g0[:, 0, 1]) - (g1[:, 1, 1] - g0[:, 1, 1])
    out["fired"] = out["on"][:, 1] > 0
    out["press"] = press.cpu().numpy()
    return out


def trigger_of(kind, owner, params):
    """trigger(st, n, d) -> [n]: the kind's bonus has work for the lord (owner [n]) now (copy on, rows :n): damage -
    he is in melee; defence - a unit of his within the aura's 35 m (or he) is in melee; charge, charge speed - he runs
    at an attack target not yet in melee, its centre within 70 m; speed - he moves out of melee; morale - a standing
    unit of his within 35 m (or he) is 8 points or more below its normal top; vigour - he fights or moves."""
    def f(st, n, d):
        u = {k: v[:n] for k, v in st.u.items()}
        j = torch.arange(n, device=st.device)
        g = lambda name: u[name][j, owner]
        alive = (u["side"] > 0) & (u["men"] > 0) & ~u["gone"]
        standing = alive & ~u["r"]
        me = standing[j, owner]
        mine = (u["side"] == u["side"][j, owner][:, None]) & standing
        near = mine & (torch.hypot(u["x"] - g("x")[:, None], u["z"] - g("z")[:, None]) <= 35.0)
        if kind == "melee_damage":
            return me & g("m")
        if kind == "melee_defence":
            return me & (near & u["m"]).any(1)
        if kind in ("charge", "charge_speed"):
            t = g("order_target")
            dist = torch.hypot(u["x"].gather(1, t.clamp(min=0)[:, None])[:, 0] - g("x"),
                               u["z"].gather(1, t.clamp(min=0)[:, None])[:, 0] - g("z"))
            return me & (g("order_kind") == O.ATTACK) & (t >= 0) & ~g("m") & g("mv") & (dist <= 70.0)
        if kind == "speed":
            return me & g("mv") & ~g("m")
        if kind == "morale":
            return me & (near & (u["morale"] <= u["leadership"] + u["morale_bonus"] - 8.0)).any(1)
        return me & (g("m") | g("mv"))
    return f


def play(scen, params, device, horizon_s=lab.HORIZON_S, id0=0, keep_off=False):
    """A batch of the small families' pairs (lab.draw) -> pairs()'s fields."""
    st, plan = build(scen, params, device, id0)
    n = len(scen)
    own = torch.arange(lab.TRACE_H, device=device)[None, :].expand(n, lab.TRACE_H)
    zero = torch.zeros(n, dtype=torch.long, device=device)
    return pairs(st, n, plan.press, own, zero, torch.ones(n, dtype=torch.long, device=device),
                 lambda s: script(s, plan), params, horizon_s, keep_off)


# --- the battle family ---

def draw_battles(kind, n, rng, vlist, times, seed_base=0, per_batch=512):
    """[(press second, [scenario])]: whole v2 battles (tools/nn/armies, as training's, attacker by role), the lord of a
    side that carries one of the kind's variants (vlist) pressing it; one press time per batch (from `times`, the
    presses' own battle times)."""
    from tools.nn.armies import generate
    out, cur = [], []
    i = 0
    while sum(len(b) for _, b in out) + len(cur) < n:
        seed = int(rng.integers(generate.TRAIN_SEEDS.start, generate.TRAIN_SEEDS.stop))
        arena = generate.battle(seed, armies="v2")
        role = "attack" if rng.random() < 0.5 else "defend"
        army = sim_scenario.from_arena(arena, role)
        lords = {sd: next((u["key"] for u in army["sides"][sd]["units"] if u.get("general")), None) for sd in (1, 2)}
        cand = [(sd, v) for sd in (1, 2) for v in vlist if lords[sd] in v.lords]
        if not cand:
            continue
        sd, v = cand[rng.integers(len(cand))]
        dose = float(rng.choice(lab.DOSES, p=lab.DOSE_P))
        for s_ in (1, 2):
            # nobody fires a real ability; the subject's side's shooters as the network's (skirmish off), the other's
            # as the game's AI keeps them (on)
            army["sides"][s_]["ai"] = False
            army["sides"][s_]["skirmish"] = s_ != sd
        foe = [u["key"] for u in army["sides"][3 - sd]["units"]]
        cur.append({"army": army, "side": sd, "key": lab.lab_key(v, dose), "kind": kind, "variant": v.source,
                    "dose": dose, "lord": lords[sd], "id": seed_base + i, "family": "battle", "seed": seed,
                    "role": role, "foe_keys": foe, "friends": len(army["sides"][sd]["units"]) - 1,
                    "enemies": len(foe)})
        i += 1
        if len(cur) == per_batch:
            out.append((int(round(float(times[rng.integers(len(times))]))), cur))
            cur = []
    if cur:
        out.append((int(round(float(times[rng.integers(len(times))]))), cur))
    for t, b in out:
        for s in b:
            s["press"] = 0
            s["t_press"] = t
    return out


def play_battle(scen, t_press, params, device, horizon_s=lab.HORIZON_S, keep_off=False, kind=None, wait_s=0):
    """A batch of battle-family pairs: the battles run once to t_press (ai_like both sides), then doubled; kind: each
    pair presses when the kind's condition holds for its lord (trigger_of), at t_press + wait_s at the latest."""
    from tools.nn.train import opponents
    n = len(scen)
    H = BATTLE_PER_SIDE
    st = sim_scenario.build([s["army"] for s in scen], params, device=device, per_side=H)
    u = st.u
    u[S.BATTLE_ID] = torch.tensor([s["id"] for s in scen], device=device)[:, None].expand(n, st.N).clone()
    side = torch.tensor([s["side"] for s in scen], device=device)
    owner = (side - 1) * H                                    # a side's general is its first slot
    names = sab.keys(params)
    _lab_slot(u, torch.arange(n, device=device), owner, torch.tensor([names.index(s["key"]) for s in scen],
                                                                      device=device))
    u["lost_worst"] = torch.zeros_like(u["men"])
    k = max(1, int(round(1.0 / params.dt)))
    for _ in range(int(t_press)):
        o = opponents.ai_like(st)
        o.ability.fill_(-1)
        _decision(st, o, params, k)
    big = S.State({kk: torch.cat([v, v]) for kk, v in st.u.items()}, torch.cat([st.t, st.t]),
                  torch.cat([st.attacker, st.attacker]), torch.cat([st.done, st.done]),
                  torch.cat([st.winner, st.winner]), torch.cat([st.lord_dead_s, st.lord_dead_s]), st.bounds,
                  list(st.keys) + list(st.keys))
    own = owner[:, None] + torch.arange(H, device=device)[None, :]
    return pairs(big, n, torch.zeros(n, dtype=torch.long, device=device), own, owner, side, opponents.ai_like,
                 params, horizon_s, keep_off, trigger_of(kind, owner, params) if kind else None, wait_s)


def _save(path, scen, data, plab, meta):
    used = sorted({s["key"] for s in scen})
    meta = dict(meta, scenarios=scen, passports={k: plab[k] for k in used})
    small = {}
    for k, v in data.items():
        small[k] = v.astype(np.float16) if k in ("x", "z", "charge", "on") else v
    np.savez_compressed(path, key=np.array([s["key"] for s in scen]), meta=np.array(json.dumps(meta)), **small)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=8192, help="pairs per kind, the small families (0: none)")
    ap.add_argument("--batch", type=int, default=2048, help="pairs per simulator batch")
    ap.add_argument("--battle-pairs", type=int, default=3072, help="pairs per kind, whole battles (0: none)")
    ap.add_argument("--battle-batch", type=int, default=512)
    ap.add_argument("--battle-trigger", type=int, default=0,
                    help="> 0: whole battles press when the kind's condition holds (trigger_of), at most this many "
                         "seconds after the drawn battle time (file battle_cond_<kind>.npz)")
    ap.add_argument("--kinds", nargs="*", default=list(lab.KINDS))
    ap.add_argument("--out", default="build/abilities_price/lab")
    ap.add_argument("--presses", nargs="*",
                    default=["build/abilities_price/natural.npz", "build/abilities_price/hold.npz"],
                    help="collect.py files: the presser's fatigue and the battle times of the presses")
    ap.add_argument("--max-t", type=float, default=600.0, help="battle family: press times above this are left out")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    pools, units, passports, sim, rules_ = lab.load_config()
    bricks = bz.bricks(bz.rules(rules_, sim))
    var = lab.variants(passports, bricks, units)
    plab = {lab.lab_key(v, dose): lab.lab_passport(v, dose, passports, bricks)
            for vs in var.values() for v in vs for dose in lab.DOSES}
    params_lab = params_with(plab)
    pool = lab.pool_of(pools)
    fat = lab.empirical_fatigue(a.presses)
    times = np.concatenate([np.load(p)["t"] for p in a.presses if Path(p).exists()]) if a.presses else np.array([180.])
    times = times[times <= a.max_t]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    for ki, kind in enumerate(a.kinds):
        ki_ = lab.KINDS.index(kind)
        if a.pairs:
            t0 = time.time()
            rng = np.random.default_rng(a.seed * 100 + ki_)
            scen = lab.draw(kind, a.pairs, rng, pool, units, var[kind], fat, seed_base=(ki_ + 1) * 1_000_000)
            parts = []
            with torch.no_grad():
                for b0 in range(0, a.pairs, a.batch):
                    parts.append(play(scen[b0:b0 + a.batch], params_lab, a.device))
                    print(f"{kind}: pairs {b0 + len(scen[b0:b0 + a.batch])}/{a.pairs}, {time.time() - t0:.0f} s",
                          flush=True)
            data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
            _save(out / f"pairs_{kind}.npz", scen, data, plab, {"kind": kind, "pairs": a.pairs, "family": "small",
                                                                  "seconds": round(time.time() - t0),
                                                                  "device": a.device, "seed": a.seed})
            print(f"{kind}: {a.pairs} pairs, fired {data['fired'].mean():.2f}, mean gold {data['y'].mean():+.2f}, "
                  f"{time.time() - t0:.0f} s", flush=True)
        if a.battle_pairs:
            t0 = time.time()
            cond = a.battle_trigger > 0
            fam = "battle_cond" if cond else "battle"
            rng = np.random.default_rng(a.seed * 100 + (70 if cond else 50) + ki_)
            batches = draw_battles(kind, a.battle_pairs, rng, var[kind], times,
                                   seed_base=(ki_ + 1) * 1_000_000 + (700_000 if cond else 500_000),
                                   per_batch=a.battle_batch)
            scen, parts = [], []
            with torch.no_grad():
                for t_press, b in batches:
                    r = play_battle(b, t_press, params_lab, a.device, kind=kind if cond else None,
                                    wait_s=a.battle_trigger)
                    for s, p in zip(b, r["press"]):
                        s["press"] = int(p)
                    parts.append(r)
                    scen += b
                    print(f"{kind} {fam}: pairs {len(scen)}/{a.battle_pairs} (from {t_press} s), "
                          f"{time.time() - t0:.0f} s", flush=True)
            data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
            _save(out / f"{fam}_{kind}.npz", scen, data, plab, {"kind": kind, "pairs": len(scen), "family": fam,
                                                                "seconds": round(time.time() - t0),
                                                                "device": a.device, "seed": a.seed})
            print(f"{kind} {fam}: {len(scen)} pairs, fired {data['fired'].mean():.2f}, mean gold "
                  f"{data['y'].mean():+.2f}, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()

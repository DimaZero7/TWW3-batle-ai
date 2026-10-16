"""The data of the ability price (docs/en/training/ability_price.md): simulator battles of a network, its ability presses
(and moments it left a ready ability: the price of pressing then) replayed open loop - the measured price - with the
pressed branch's trajectory of the presser's side recorded a second at a time (bonuses.TRACE) for the bricks.

    DOCK_NAME=agent-abprice DOCK_BUILD=<main>/build bash tools/nn/dock.sh tools.nn.abilities_price.collect \
        --ckpt build/v2/itS25_12.pt --out build/abilities_price/presses.npz [--hold]

The training's machinery does the capture and the branches (ability_branch.Brancher.capture, branch_env, the recorded
orders, trade); play_pair() here is ability_branch.play_pair's open loop with two additions:
  * `each` gets the pressed branch after every decision (the recorder);
  * the price at marks (decisions) inside the horizon too (`diff_at`: 30, 60, 90 s of a 120 s horizon in one play);
  * one_slot (default): "not" drops only the pressed SLOT's orders of the unit - as ability_branch.play_pair does (a
    test: the same prices to the bit). one_slot=False is the training's old way (all of the unit's ability orders
    dropped in "not": another ability of the same lord the network presses within the horizon - the Warlord has three -
    then lands in this press's price); `other` counts the decisions the pressed branch orders another slot of the
    unit.
--hold: the learner's network never presses in the main battles (its ability orders dropped), so the lords' abilities
stay ready at every stage of a battle, in melee too, and the replayed moments (--missed) cover them all; the slot of
such a moment is a random ready one (--random-slot), not the head's favourite. In the branches the network plays as it
is (presses included).

The file (numpy, savez_compressed): per press p, copy c, sample s (0 the press, s after second s), unit h of the
presser's side (H slots): the TRACE fields [P, C, S+1, H] (mmax [P, C, H], on [P, C, S+1]); per press: key (index into
key_names), owner (h of the presser), side, pressed (a press of the network, else a moment it left), t (battle time),
opponent (league code), other (decisions the pressed branch ordered another slot of the unit, mean over copies),
row (the battle's row in the main battles: with t it tells the presses of one battle), diff [P, C] (trade pressed -
trade not at the horizon, budget share), diff_at [P, C, M] the same at the marks (meta "marks_s"), budget [P, C]
(gold); --compare-slots: diff_alt [P, C] the same played the other way (all slots / one slot).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from tools.nn.armies import generate
from tools.nn.sim import abilities as sab
from tools.nn.sim import geometry
from tools.nn.sim import orders as O
from tools.nn.sim import state as S
from tools.nn.train import ability_branch as ab
from tools.nn.train import bankproc, checkpoint, league, reward, rollout, scenes

FLOATS = ("x", "z", "charge", "morale", "fatigue")
FLAGS = ("alive", "r", "m", "mv", "sprint")


def play_pair(template, actor, snap, unit, side, slot, copies, decisions, seed, each=None, one_slot=True, marks=()):
    """ability_branch.play_pair(openloop=True) with each(i, pressed env, not env) after every decision, the price at
    the marks (numbers of decisions played) and, one_slot=False, "not" dropping all of the unit's ability orders (module
    doc). -> (diff [b, copies], budget [b, copies], other [b, copies]: decisions "pressed" ordered another slot of the
    unit, at [b, copies, len(marks)]: trade pressed - trade not after each mark's decisions)."""
    b = len(snap["keys"])
    dev = template.device
    rep = torch.arange(b, device=dev).repeat_interleave(copies)
    big = ab.select(snap, rep)
    copy = torch.arange(copies, device=dev).repeat(b)
    big["u/" + S.BATTLE_ID] = big["u/" + S.BATTLE_ID] + ab.ID_STEP * (copy + 1)[:, None]
    envs = {}
    for name, press in (("pressed", True), ("not", False)):
        env, parts, scripted = ab.branch_env(template, big)
        o = parts[0][1]
        rows = env.rows_learn
        j = torch.arange(len(rep), device=dev)
        r_of = torch.full((2 * len(rep),), -1, dtype=torch.long, device=dev)
        r_of[rows] = torch.arange(len(rows), device=dev)
        r = r_of[(side[rep] - 1) * len(rep) + j]
        n = unit[rep]
        o.ability[r, n] = slot[rep] if press else -1
        envs[name] = (env, parts, scripted)
    said = []
    envs["pressed"][0]._play = ab._recording_play(envs["pressed"][0], said)
    k_rep = slot[rep]
    other = torch.zeros(len(rep), device=dev)
    s_rep = side[rep]
    at = []
    state = (torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    try:
        for i in range(decisions):
            for name in ("pressed", "not"):
                env, parts, scripted = envs[name]
                torch.manual_seed(seed * 7919 + i)
                if i == 0:
                    env.replay(parts, scripted)
                elif name == "not":
                    p, sc, delay = said.pop()
                    p = [(rows_, O.Orders(*(getattr(o_, f).clone() for f in O.FIELDS))) for rows_, o_ in p]
                    a = p[0][1].ability[r, n]
                    other += ((a >= 0) & (a != k_rep)).float()
                    p[0][1].ability[r, n] = torch.where(a == k_rep, torch.full_like(a, -1), a) if one_slot else -1
                    ab._replay_orders(env, p, sc, delay)
                else:
                    env.step(actor)
            if each is not None:
                each(i, envs["pressed"][0], envs["not"][0])
            if i + 1 in marks:
                at.append(ab.trade(envs["pressed"][0].st.u, s_rep, template.weights.rout_share)
                          - ab.trade(envs["not"][0].st.u, s_rep, template.weights.rout_share))
    finally:
        torch.set_rng_state(state[0])
        if state[1] is not None:
            torch.cuda.set_rng_state_all(state[1])
        said.clear()
        for env, _, _ in envs.values():
            ab._unhook(env)
    ends = {name: ab.trade(envs[name][0].st.u, s_rep, template.weights.rout_share) for name in envs}
    budget = reward.budget(envs["not"][0].st.u)
    at = torch.stack(at, 1).reshape(b, copies, -1) if at else torch.zeros(b, copies, 0, device=dev)
    return ((ends["pressed"] - ends["not"]).reshape(b, copies), budget.reshape(b, copies),
            other.reshape(b, copies), at)


def fields(u, own, owner, slot, spacing):
    """{field: [b, H]} of the battles' units `own` [b, H] (the presser's side), on [b] the owner's ability timer (a
    spell's slot, O.SPELL_AT + k: its spell k's), the spells' bookkeeping [b] of the owner (bonuses.SPELL_FIELDS: 0
    without magic)."""
    g = lambda v: v.gather(1, own)
    alive = (u["side"] > 0) & (u["men"] > 0) & ~u["gone"]
    gap = geometry.pairwise(u, spacing)["gap"]
    tgt = u["order_target"]
    gap_t = gap.gather(2, tgt.clamp(min=0)[:, :, None])[:, :, 0]
    sprint = (alive & ~u["r"] & (u["order_kind"] == O.ATTACK) & (tgt >= 0) & ~u["m"] & u["mv"]
              & (gap_t <= u["charge_dist"]))
    out = {"alive": g(alive), "sprint": g(sprint)}
    for k in ("r", "m", "mv"):
        out[k] = g(u[k])
    for k in FLOATS:
        out[k] = g(u[k].float())
    on = torch.stack([u[f"ab{k}_on"] for k in range(sab.SLOTS)], 2)                  # [b, N, SLOTS]
    j = torch.arange(len(owner), device=owner.device)
    ab_on = on[j, owner, slot.clamp(min=0, max=sab.SLOTS - 1)]
    zero = torch.zeros(len(owner), device=owner.device)
    if "sp0_on" in u:
        sp = torch.stack([u[f"sp{k}_on"] for k in range(O.SPELLS)], 2)               # [b, N, SPELLS]
        sp_on = sp[j, owner, (slot - O.SPELL_AT).clamp(min=0, max=O.SPELLS - 1)]
        out["on"] = torch.where(slot >= O.SPELL_AT, sp_on, ab_on)
    else:
        out["on"] = torch.where(slot >= O.SPELL_AT, zero, ab_on)
    for f, name in (("sp_gold_e", "bm_gold_e"), ("sp_gold_f", "bm_gold_f"), ("sp_knock", "bm_knock"),
                    ("winds", "wm_cur")):
        out[f] = u[name].gather(1, owner[:, None])[:, 0].float() if name in u else zero
    return out


class Recorder:
    """The pressed branch's trajectory: sample 0 from the snapshot, then one after each decision (play_pair's each)."""

    def __init__(self, snap, unit, side, slot, copies, params):
        dev = unit.device
        b = len(unit)
        self.b, self.K = b, copies
        rep = torch.arange(b, device=dev).repeat_interleave(copies)
        N = snap["u/side"].shape[1]
        H = N // 2
        self.H = H
        self.own = ((side[rep] - 1) * H)[:, None] + torch.arange(H, device=dev)[None, :]          # [bK, H]
        self.owner = unit[rep]
        self.slot = slot[rep]
        self.spacing = params.sim["formation"]["spacing_m"]
        u0 = {k[2:]: v[rep] for k, v in snap.items() if k.startswith("u/")}
        self.mmax = (u0["leadership"] + u0["morale_bonus"]).gather(1, self.own)
        j = torch.arange(len(rep), device=dev)
        key = torch.stack([u0[f"ab{k}"] for k in range(sab.SLOTS)], 2)[
            j, self.owner, self.slot.clamp(min=0, max=sab.SLOTS - 1)]
        if "sp0" in u0:
            # a spell's slot: its key's index after the abilities' (key_names: sab.keys + magic.keys)
            sp = torch.stack([u0[f"sp{k}"] for k in range(O.SPELLS)], 2)[
                j, self.owner, (self.slot - O.SPELL_AT).clamp(min=0, max=O.SPELLS - 1)]
            n_ab = len(sab.keys(params))
            key = torch.where(self.slot >= O.SPELL_AT, torch.where(sp >= 0, sp + n_ab, torch.full_like(sp, -1)), key)
        self.key = key
        self.samples = [self._take(u0)]

    def _take(self, u):
        return fields(u, self.own, self.owner, self.slot, self.spacing)

    def each(self, i, pressed, not_):
        self.samples.append(self._take(pressed.st.u))

    def result(self, n):
        """{field: numpy [n, K, S+1, H] (on and the spells' fields: [n, K, S+1])} of the first n snapshots (the rest:
        padding)."""
        b, K = self.b, self.K
        out = {}
        for k in self.samples[0]:
            x = torch.stack([s[k] for s in self.samples], 1)                              # [bK, S+1, ...]
            x = x.reshape(b, K, *x.shape[1:])[:n]
            out[k] = x.cpu().numpy()
        out["mmax"] = self.mmax.reshape(b, K, -1)[:n].cpu().numpy()
        out["key"] = self.key.reshape(b, K)[:n, 0].cpu().numpy()
        return out


def ready_slots(u, b, n, table):
    """The slots of unit n of snapshot battle b a player can fire now (an active self-cast ability, ready, not on)."""
    out = []
    standing = bool(u["men"][b, n] > 0) and not bool(u["r"][b, n]) and not bool(u["gone"][b, n])
    for k in range(sab.SLOTS):
        a = int(u[f"ab{k}"][b, n])
        if a < 0 or not standing:
            continue
        row = table[a]
        if row[sab.COL["passive"]] > 0 or row[sab.COL["self_cast"]] <= 0:
            continue
        if float(u[f"ab{k}_on"][b, n]) <= 0 and float(u[f"ab{k}_cd"][b, n]) <= 0:
            out.append(k)
    return out


def play_chunk(env, actor, br, seed, one_slot=True, random_slot=False, gen=None, compare=False, marks=()):
    """The chunk's captured moments (br.chunk) played open loop with the recorder: -> {field: numpy}, per press.
    compare: each batch played again the other way (one slot / all slots, the same seeds): diff_alt."""
    c, cfg = br.chunk, br.cfg
    dev = env.device
    snap = ab.concat(c.snaps)
    entries = list(c.entries)
    if random_slot:
        # a moment left (not a press): a random ready slot instead of the head's favourite
        table = sab.table(env.params, "cpu").numpy()
        u = {k[2:]: v.cpu() for k, v in snap.items() if k.startswith("u/")}
        for i, e in enumerate(entries):
            if not e[5]:
                ok = ready_slots(u, e[6], e[2], table)
                if ok:
                    entries[i] = e[:4] + (int(ok[gen.integers(len(ok))]),) + e[5:]
    E = len(entries)
    at = torch.tensor([e[6] for e in entries], device=dev)
    unit = torch.tensor([e[2] for e in entries], device=dev)
    side = torch.tensor([e[3] for e in entries], device=dev)
    slot = torch.tensor([e[4] for e in entries], device=dev)
    kinds = ab.kind_of(snap)[at].tolist()
    decisions = max(1, int(round(cfg.horizon_s / env.decision_s)))
    K = max(1, int(cfg.copies))
    comp = br.comp
    todo = {k: [i for i, kk in enumerate(kinds) if kk == k] for k in comp}
    assert sum(map(len, todo.values())) == E
    parts = []
    ab._release(dev)
    calls = 0
    while any(todo.values()):
        part, fill = [], []
        for k in sorted(comp):
            part += todo[k][:comp[k]]
            fill += [k] * (comp[k] - len(todo[k][:comp[k]]))
            todo[k] = todo[k][comp[k]:]
        ix = torch.tensor(part, device=dev)
        sub = ab.select(snap, at[ix])
        u_, s_, k_ = unit[ix], side[ix], slot[ix]
        if fill:
            ps, pu, pside, pslot = ab.pad_snap(env, sub, 0, fill)
            sub = ab.concat([sub, ps])
            u_, s_, k_ = torch.cat([u_, pu]), torch.cat([s_, pside]), torch.cat([k_, pslot])
        rec = Recorder(sub, u_, s_, k_, K, env.params)
        calls += 1
        d, bg, oth, at_ = play_pair(env, actor, sub, u_, s_, k_, K, decisions, seed * 1000 + calls, each=rec.each,
                                    one_slot=one_slot, marks=marks)
        n = len(part)
        r = rec.result(n)
        r["diff"] = d[:n].cpu().numpy()
        r["budget"] = bg[:n].cpu().numpy()
        r["other"] = oth[:n].mean(1).cpu().numpy()
        r["diff_at"] = at_[:n].cpu().numpy()
        if compare:
            d2, _, _, _ = play_pair(env, actor, sub, u_, s_, k_, K, decisions, seed * 1000 + calls, one_slot=not one_slot)
            r["diff_alt"] = d2[:n].cpu().numpy()
        r["owner"] = (u_[:n] - (s_[:n] - 1) * rec.H).cpu().numpy()
        r["side"] = s_[:n].cpu().numpy()
        r["pressed"] = np.array([entries[i][5] for i in part], dtype=bool)
        r["t"] = snap["st/t"][at[ix]].cpu().numpy()
        r["row"] = np.array([int(env.rows_learn[entries[i][1]]) % env.B for i in part])
        r["opponent"] = np.asarray(snap["lay/opponent"])[at[ix].cpu().numpy()]
        parts.append(r)
    ab._release(dev)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


class Held(rollout.Battles):
    """Battles whose learner never presses an ability (its ability orders dropped; module doc --hold)."""

    def _act(self, actor, a, frame, rows, h, greedy):
        obs_r, h0, logits, action, orders, h_new = super()._act(actor, a, frame, rows, h, greedy)
        if action.ability is not None:
            action.ability = torch.full_like(action.ability, -1)
        orders.ability = torch.full_like(orders.ability, -1)
        return obs_r, h0, logits, action, orders, h_new


def battles(device, B, mix, seed, compile=None, hold=False):
    params = rollout.params_with_limit(3600)
    lay = league.layout(B, len(scenes.SCENES), mix, scene_attacker=scenes.attackers())
    rng = np.random.default_rng(seed)
    seeds = rng.integers(generate.TRAIN_SEEDS.start, generate.TRAIN_SEEDS.stop, max(B, 256))
    src = bankproc.build(bankproc.Spec(seeds, seed, 19, params, lay, armies="v2"), device)
    cls = Held if hold else rollout.Battles
    return cls(lay, scenes.SCENES, device, params, seed=seed, source=src, compile=compile)


def collect(env, actor, cfg, target, minutes, chunk, seed, one_slot=True, random_slot=False, compare=False,
            log=print, marks=()):
    """Plays env's battles with the actor and prices the captured moments until `target` presses or `minutes`."""
    br = ab.Brancher(cfg, env.device)
    env.brancher = br
    gen = np.random.default_rng(seed + 17)
    out, n, t0 = [], 0, time.time()
    rounds = 0
    while n < target and time.time() - t0 < minutes * 60:
        br.begin(env, chunk)
        with torch.no_grad():
            for _ in range(chunk):
                env.step(actor)
        if br.chunk.entries:
            rounds += 1
            r = play_chunk(env, actor, br, seed + rounds, one_slot, random_slot, gen, compare, marks)
            out.append(r)
            n += len(r["diff"])
        br.chunk = ab._Chunk()
        log(f"round {rounds}: {n} presses, battle time {float(env.st.t.mean()):.0f} s, {time.time() - t0:.0f} s",
            flush=True)
    env.brancher = None
    return {k: np.concatenate([o[k] for o in out]) for k in out[0]} if out else {}


def save(path, data, key_names, meta):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    small = {}
    for k, v in data.items():
        if k in ("x", "z", "charge", "morale", "on"):
            v = v.astype(np.float16)
        small[k] = v
    np.savez_compressed(path, key_names=np.array(key_names), meta=np.array(json.dumps(meta)), **small)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="build/v2/itS25_12.pt")
    ap.add_argument("--out", default="build/abilities_price/presses.npz")
    ap.add_argument("--battles", type=int, default=128)
    ap.add_argument("--mix", default='{"self": 0.3, "ai_like": 0.7}')
    ap.add_argument("--presses", type=int, default=3000, help="stop after this many priced moments")
    ap.add_argument("--minutes", type=float, default=40)
    ap.add_argument("--chunk", type=int, default=40, help="decisions between two branch rounds")
    ap.add_argument("--share", type=float, default=1.0, help="share of the network's presses replayed")
    ap.add_argument("--missed", type=float, default=0.0, help="chance a moment of a ready ability left is replayed")
    ap.add_argument("--hold", action="store_true", help="the learner never presses in the main battles (module doc)")
    ap.add_argument("--random-slot", action="store_true", help="a moment left: a random ready slot")
    ap.add_argument("--all-slots", action="store_true",
                    help="'not' drops all of the unit's ability orders (the training's old way)")
    ap.add_argument("--compare-slots", action="store_true",
                    help="each batch also played the other way (one slot / all slots): diff_alt")
    ap.add_argument("--copies", type=int, default=4)
    ap.add_argument("--horizon", type=float, default=60.0)
    ap.add_argument("--marks", default="30,60,90", help="seconds inside the horizon the price is also taken at")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--no-compile", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    env = battles(a.device, a.battles, json.loads(a.mix), a.seed, compile=False if a.no_compile else None, hold=a.hold)
    actor = checkpoint.load_policy(a.ckpt, a.device)
    cfg = ab.Config(share=a.share, missed=a.missed, copies=a.copies, horizon_s=a.horizon, batch=a.batch,
                    openloop=True, seed=a.seed)
    marks_s = [float(x) for x in a.marks.split(",") if x.strip() and float(x) < a.horizon]
    marks = tuple(max(1, int(round(m / env.decision_s))) for m in marks_s)
    t0 = time.time()
    data = collect(env, actor, cfg, a.presses, a.minutes, a.chunk, a.seed, one_slot=not a.all_slots,
                   random_slot=a.random_slot, compare=a.compare_slots, marks=marks)
    meta = {"ckpt": a.ckpt, "mix": a.mix, "copies": a.copies, "horizon_s": a.horizon, "share": a.share,
            "missed": a.missed, "hold": a.hold, "random_slot": a.random_slot, "one_slot": not a.all_slots,
            "seconds": round(time.time() - t0), "dt_s": env.decision_s, "marks_s": marks_s,
            "opponents": {str(v): k for k, v in league.CODE.items()}}
    from tools.nn.sim import magic
    save(a.out, data, sab.keys(env.params) + magic.keys(env.params), meta)
    print(f"saved {len(data.get('diff', []))} presses to {a.out} in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()

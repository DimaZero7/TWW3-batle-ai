"""The ability branches: an ability press priced by its real effect, replayed in the simulator (docs/en/training/training.md
"Abilities by their effect"; run.py --ability-branch, off by default).

At a decision of the training batch (rollout.Battles.step) a share of the learner's presses (the ability head chose a
slot) and of the moments an ability was ready but not pressed are snapshotted: the battle's state before the decision's
simulator steps, everything the networks and the scripts need to go on (the observation's memory, the networks'
memories after the decision, the commitments, the decision's orders). At the chunk's end (rollout.collect, before the
update) every snapshot branches in two:

    pressed   - the decision as it was, the unit's ability pressed (the slot it chose; at a missed moment the ready slot
                the head likes best);
    not       - the same decision with the unit's ability not pressed (no effect, no recharge: the simulator fires nothing);

K copies of each (copies=4), horizon_s (60 s) on, both sides played as in the batch (the learner's current weights, the
past version of the battle's slot, the scripts). In `not` the network goes on as it would and may press the ability a
decision later: the price is then "now against the policy's own later", near 0 while the head presses whatever is
ready (itS23: 0.9996 a decision). hold_s > 0 keeps the slot unused in `not` that long (Battles.ability_hold): "now
against not within hold_s". Common random numbers: copy c of `pressed` and copy c of `not` sit at
the same place of two batches of the same shape and every decision of both starts from the same seed of torch's
generator (the networks' sampling, the orders' latency), and the copy's battle id (state `row`, the simulator's hash
draws) is the same in both - the two differ only by the press and what follows from it. The copies differ by their
seeds' positions and ids.

The price of the press = mean over the copies of (trade of `pressed` - trade of `not`) at the horizon, the trade as
the v2 reward's gold term from the unit's side: (enemy gold destroyed - own gold lost) / budget (reward.gold_sides, the
worst share lost so far). It is the extra advantage of the ability head's log-probability of the choice made: +price for
a press, -price for a missed moment (its choice was "none"; the price is that of pressing then). ppo.update adds
cfg.ability x the clipped surrogate of the ability head alone (not the reward, not the critic; the gradient reaches the
shared layers through the head as PPO's does), its advantages divided by the chunk's root mean square of the prices
(at least ADV_FLOOR) and clipped to ADV_CLIP. The weight decays (run.py --ability-branch-weight, -decay).

Cost: each snapshot plays 2 x copies battles for horizon_s. All the chunk's snapshots go in ONE batch of a fixed
shape a branch: the GPU step costs about the same at 64 battles and at 512 (~0.06 s a decision: launches, not work),
so it is the number of sequential decisions that costs, not the battles. The batch has `batch` slots (min(batch, cap)
with a cap) split between the battle kinds (who plays each side: the learner, a past version, a script;
kinds_of_ctrl) by their share of the main batch's learner rows (composition()); the capture keeps within each kind's
slots (Brancher.capture); a free slot is padding (a snapshot battle relabelled to the kind and ended: frozen, its
result dropped). Same counts of each kind give the same learner and past rows every time: one shape, compiled once a
process (~50 s), then 2 branches x horizon decisions (5.6-7.8 s an update at cap 128, copies 4: 1024 battles of 60 s;
GPU, S25's options). The allocator's cache is released around the branches (_release).
Without a cap, snapshots beyond the slots play in further batches of the same shape.

The ability critic (Config.critic; run.py --ability-critic, off by default; ability_critic.py): a few percent of the
moments are replayed and their price is noisy, so the head learns little from them. With the critic, every decision
records each moment (a press, or a ready ability left: Brancher._moments) with the critic's input; at the chunk's end
the replayed moments' prices go into the critic's buffer (Brancher.target: the regression target, the measured price),
it takes a few steps, and once it is ready (warm-up: critic_warmup samples and its held-out error below the mean's)
EVERY moment of the chunk gets a price as the head's advantage (COMA's way: the critic's value of the unit's option at
each decision, one pass): +price at a press, -price at a missed moment (the slot the head likes best).
    not replayed  the critic's Q x critic_weight;
    replayed      the critic as a control variate (Q-Prop): Q + (measured - Q) clipped to +-critic_resid x the buffer's
                  root mean square, or the measured price as it is - whichever of the two has the smaller variance over
                  the chunk's replayed moments (both standard deviations logged: price_sd, cv_sd; cv_used).
Divided by the root mean square of the buffer's measured prices (at least ADV_FLOOR: the critic's Q is smaller where it
is unsure) and clipped as above. Before it is ready: the replayed moments only, as without it.

How much of a price is real: the chunk's prices per copy are kept (Brancher.copies [E, copies]); the log's price_rel =
reliability(): 1 - the copies' noise variance of the mean / the prices' variance - the share that differs between
moments, the ceiling of what any critic can explain; the critic keeps each price's two halves of the copies and logs
the buffer's split-half value (critic_rel).

Extension points (not used yet): a bootstrapped (TD) target of the critic (Brancher.target), a cost of an ability left
unused through a battle (Config.unused_penalty: must be 0).
"""
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from tools.nn.model import heads as hd
from tools.nn.model import observation as ob
from tools.nn.model.frame import Frame
from tools.nn.sim import orders as O
from tools.nn.sim import state as S
from tools.nn.train import league, reward, scenes

# A press whose price is within this share of the budget either way is "empty" (the log's zero share): 0.2 % of an
# army, ~20 gold of a 10 000-gold army over the horizon
ZERO = 0.002
ADV_FLOOR = 0.005          # the prices' root mean square at least this (budget share): tiny noise is not blown up
ADV_CLIP = 5.0             # the normalised advantage within +-this
ID_STEP = 1_000_003        # a copy's battle id: the snapshot's + ID_STEP x (copy + 1)
FRAME = ("cx", "cz", "ux", "uz")
MEMORY = tuple(f for f in ob.Memory.__dataclass_fields__ if f != "frame")


@dataclass(frozen=True)
class Config:
    share: float = 0.2          # the share of the learner's presses replayed
    missed: float = 0.005       # the chance a moment of a ready ability not pressed (per unit and decision) is replayed
    copies: int = 4             # copies of each branch (common random numbers in pairs)
    horizon_s: float = 60.0     # seconds of battle each branch plays
    cap: int = 0                # snapshots a chunk at most (0: no cap); spread over the decisions
    batch: int = 128            # snapshots per branch batch (x copies battles in each of the two branches)
    seed: int = 0
    hold_s: float = 0.0         # "not": the unit's slot also stays unused this many seconds (0: the network goes on as
    #                             it would - and may press it a decision later; module doc)
    unused_penalty: float = 0.0  # reserved (a cost of an ability unused through a battle): not implemented, must be 0
    # the ability critic (ability_critic.py; run.py --ability-critic): off by default
    critic: bool = False
    critic_weight: float = 1.0  # its predicted price x this at the moments not replayed
    critic_resid: float = 2.0   # a replayed moment: Q + (measured - Q) within +-this x the buffer's root mean square
    critic_warmup: int = 2000   # replayed moments in its buffer before it is used (and it must beat the mean)
    critic_buffer: int = 25000  # replayed moments it keeps (the last)
    critic_steps: int = 32      # its optimiser's steps an update

    def __post_init__(self):
        if self.unused_penalty:
            raise ValueError("ability_branch.Config.unused_penalty is reserved: not implemented yet")


# --- snapshots: flat dicts of per-battle data ([b, ...] tensors, lists, numpy arrays) ---

def _sides_at(x, rows, B, idx):
    """Rows' data x [R, ...] (row = (side - 1) * B + battle, R = len(rows)) of the battles idx [b] -> [b, 2, ...]
    (zeros where a side has no row)."""
    pos = torch.full((2 * B,), -1, dtype=torch.long, device=rows.device)
    pos[rows] = torch.arange(len(rows), device=rows.device)
    out = []
    for s in range(2):
        p = pos[s * B + idx]
        v = x[p.clamp(min=0)]
        out.append(torch.where((p >= 0).reshape(-1, *[1] * (v.dim() - 1)), v, torch.zeros_like(v)))
    return torch.stack(out, 1)


def _from_sides(X, rows, B):
    """[B, 2, ...] -> the rows' [R, ...]."""
    return X[rows % B, rows // B]


def _two(v, B):
    """Per row of both sides [2B, ...] -> [B, 2, ...]."""
    return v.reshape(2, B, *v.shape[1:]).transpose(0, 1)


def _untwo(v):
    """[B, 2, ...] -> [2B, ...]."""
    return v.transpose(0, 1).reshape(-1, *v.shape[2:])


def snapshot(env, idx, info):
    """The battles idx (a long tensor on the device) of env (rollout.Battles) at its decision now, before its simulator
    steps (Battles.step's capture), as a flat dict. info: the decision's {"parts", "scripted", "h_new", "h_past"}."""
    B = env.B
    il = idx.tolist()
    st, d = env.st, {}
    for k, v in st.u.items():
        d["u/" + k] = v[idx]
    for k in ("t", "attacker", "done", "winner", "lord_dead_s"):
        d["st/" + k] = getattr(st, k)[idx]
    d["keys"] = [list(st.keys[i]) for i in il]
    for k in scenes.LiveSetup.FIELDS:
        d["setup/" + k] = getattr(env.setup.arrays, k)[idx]
    for s in (1, 2):
        d[f"char/{s}"] = env.setup.char[s][idx]
        m = env.mem[s]
        for k in FRAME:
            d[f"mem{s}/frame/{k}"] = getattr(m.frame, k)[idx]
        for k in MEMORY:
            v = getattr(m, k)
            if v is not None:
                d[f"mem{s}/{k}"] = v[idx]
    d["factions"] = [env.setup.factions[i] for i in il]
    for k in ("health", "last_hit", "hit_rate", "bank_row", "want", "ctrl", "by_rule", "is_past", "battle_slot"):
        d["b/" + k] = getattr(env, k)[idx]
    d["lay/scene"], d["lay/learner"], d["lay/opponent"] = (np.asarray(getattr(env.layout, k))[il]
                                                           for k in ("scene", "learner", "opponent"))
    for k, v in env.cstate.items():
        d["commit/" + k] = _two(v, B)[idx]
    # the networks' memories after the decision, per battle and side
    d["h"] = _sides_at(info["h_new"], env.rows_learn, B, idx)
    rp = env.rows_past
    hp = info.get("h_past") or {}
    if len(rp) and hp:
        slot = env.battle_slot[rp % B]
        rows_h = None
        for k, h in hp.items():
            rows_h = h if rows_h is None else torch.where((slot == k).reshape(-1, 1, 1), h, rows_h)
        d["hp"] = _sides_at(rows_h, rp, B, idx)
    # the decision's orders: the networks' per side, the scripts'
    for name, rows in (("learn", env.rows_learn), ("past", env.rows_past)):
        o = next((o for r, o in info["parts"] if r is rows), None)
        if o is not None:
            for f in O.FIELDS:
                d[f"o_{name}/{f}"] = _sides_at(getattr(o, f), rows, B, idx)
    if info.get("scripted") is not None:
        for f in O.FIELDS:
            d[f"o_script/{f}"] = getattr(info["scripted"], f)[idx]
    return d


def concat(snaps):
    """Snapshots' dicts -> one (their battles in order). A key some lack (a past memory) is zero-filled for them."""
    keys = []
    for s in snaps:
        keys += [k for k in s if k not in keys]
    out = {}
    for k in keys:
        have = [s for s in snaps if k in s]
        x = have[0][k]
        if isinstance(x, list):
            out[k] = [v for s in snaps for v in s[k]]
        elif isinstance(x, np.ndarray):
            out[k] = np.concatenate([s[k] for s in snaps])
        else:
            parts = [s[k] if k in s else x.new_zeros((len(s["keys"]),) + tuple(x.shape[1:])) for s in snaps]
            out[k] = torch.cat(parts)
    return out


def select(snap, idx):
    """The battles idx (a long tensor, repeats allowed) of a snapshot dict."""
    il = idx.tolist()
    out = {}
    for k, v in snap.items():
        if isinstance(v, list):
            out[k] = [v[i] for i in il]
        elif isinstance(v, np.ndarray):
            out[k] = v[il]
        else:
            out[k] = v[idx.to(v.device)]
    return out


def branch_env(template, snap):
    """A rollout.Battles of the snapshot's battles (no restarts, no teachers, fresh counters) that shares the template's
    simulator, networks' slots, scripts and compiled steps. -> (env, its first decision's parts, the scripts' Orders)."""
    from tools.nn.train import rollout
    env = rollout.Battles.__new__(rollout.Battles)
    env.__dict__.update(template.__dict__)
    dev = template.device
    B = len(snap["keys"])
    env.B = B
    env.auto_reset = False
    env.brancher = None
    env.ability_hold = None
    env.layout =league.Layout(snap["lay/scene"], snap["lay/learner"], snap["lay/opponent"])
    for k in ("health", "last_hit", "hit_rate", "bank_row", "want", "ctrl", "by_rule", "is_past", "battle_slot"):
        setattr(env, k, snap["b/" + k].clone())
    u = {k[2:]: v.clone() for k, v in snap.items() if k.startswith("u/")}
    env.st = S.State(u, *(snap["st/" + k].clone() for k in ("t", "attacker", "done", "winner", "lord_dead_s")),
                     template.st.bounds, [list(r) for r in snap["keys"]])
    arrays = ob._Arrays(**{k: snap["setup/" + k].clone() for k in scenes.LiveSetup.FIELDS})
    env.setup = scenes.LiveSetup(arrays, {s: snap[f"char/{s}"].clone() for s in (1, 2)}, list(snap["factions"]))
    env.bounds2 = torch.cat([env.setup.bounds, env.setup.bounds])

    def memory(s):
        return ob.Memory(Frame(*(snap[f"mem{s}/frame/{k}"].clone() for k in FRAME)),
                         *(snap[f"mem{s}/{k}"].clone() if f"mem{s}/{k}" in snap else None for k in MEMORY))
    env.mem = {s: memory(s) for s in (1, 2)}
    env.cmem = env.mem                                   # (the critic's view is not taken in a branch)
    env.cstate = {k[len("commit/"):]: _untwo(v.clone()) for k, v in snap.items() if k.startswith("commit/")}
    flat = torch.cat([env.ctrl[:, 0], env.ctrl[:, 1]])
    env.rows_learn = (flat == league.LEARNER).nonzero().squeeze(1)
    env.rows_past = (flat == league.CODE["past"]).nonzero().squeeze(1)
    from tools.nn.train.rollout import placement
    env.place_learn = placement(env.rows_learn, B)
    env.place_past = placement(env.rows_past, B)
    opp = torch.as_tensor(env.layout.opponent, device=dev)
    env.row_opp = torch.cat([opp, opp])[env.rows_learn]
    env.row_normal = torch.ones_like(env.row_opp, dtype=torch.bool)
    env.h_learn = _from_sides(snap["h"], env.rows_learn, B).clone()
    env.past = list(template.past)
    env.slot_stats = torch.zeros_like(template.slot_stats)
    hp = _from_sides(snap["hp"], env.rows_past, B).clone() if len(env.rows_past) and "hp" in snap else None
    used = set(env.battle_slot[env.is_past].unique().tolist()) if len(env.rows_past) else set()
    env.slot_used = used or {template.active}
    env.active = min(env.slot_used)
    env.past_h = [hp if (k in used and hp is not None) else None for k in range(template.past_slots)]
    # no teachers, fresh counters
    env.teach, env.teach_normal, env.teach_names, env.teach_heavy = {}, (), (), ()
    env.teach_share = torch.ones(0, device=dev)
    env.share_host = []
    env.teach_draw = torch.zeros(B, device=dev)
    env.decisions = 0
    for k in ("stats", "orders_stats", "kind_stats", "lord_stats", "switch_stats", "ability_stats", "nan_stats",
              "part_stats", "part_steps"):
        setattr(env, k, torch.zeros_like(getattr(template, k)))
    env.kind_battle = torch.zeros((B,) + tuple(template.kind_battle.shape[1:]), device=dev)
    env.order_battle = torch.zeros((B,) + tuple(template.order_battle.shape[1:]), device=dev)
    env.row_attacks = None
    env.cur = None
    env.battles = 0
    env.timeouts = 0

    def orders(name, rows):
        if f"o_{name}/kind" not in snap:
            return None
        return O.Orders(*(_from_sides(snap[f"o_{name}/{f}"], rows, B).clone() for f in O.FIELDS))
    parts = [(env.rows_learn, orders("learn", env.rows_learn))]
    if len(env.rows_past):
        o = orders("past", env.rows_past)
        if o is not None:
            parts.append((env.rows_past, o))
    scripted = O.Orders(*(snap[f"o_script/{f}"].clone() for f in O.FIELDS)) if "o_script/kind" in snap else None
    return env, parts, scripted


def trade(u, side, rout_share=reward.Weights.rout_share):
    """[B] the side's (side [B]: 1 or 2) trade now: (the enemy's gold lost - its own) / budget (the v2 reward's gold term,
    each unit's worst share lost so far)."""
    g = reward.gold_sides(u, rout_share)
    s = (side - 1).long()[:, None]
    own, foe = g.gather(1, s)[:, 0], g.gather(1, 1 - s)[:, 0]
    return (foe - own) / reward.budget(u)


def play_pair(template, actor, snap, unit, side, slot, copies, decisions, seed, hold_s=0.0):
    """The two branches of each snapshot battle (snap: b battles; unit, side, slot [b]: the unit whose ability differs,
    its side 1 / 2 and the slot pressed), copies copies each, `decisions` decisions (the snapshot's own first); hold_s:
    in "not" the slot also stays unused that many seconds. -> (diff [b, copies]: trade pressed - trade not at the end,
    budget [b, copies]: the battles' budgets in gold)."""
    b = len(snap["keys"])
    dev = template.device
    rep = torch.arange(b, device=dev).repeat_interleave(copies)
    big = select(snap, rep)
    copy = torch.arange(copies, device=dev).repeat(b)
    big["u/" + S.BATTLE_ID] = big["u/" + S.BATTLE_ID] + ID_STEP * (copy + 1)[:, None]
    ends = {}
    envs = {}
    for name, press in (("pressed", True), ("not", False)):
        env, parts, scripted = branch_env(template, big)
        o = parts[0][1]
        rows = env.rows_learn
        # the unit's row: (side - 1) * B + battle
        j = torch.arange(len(rep), device=dev)
        r_of = torch.full((2 * len(rep),), -1, dtype=torch.long, device=dev)
        r_of[rows] = torch.arange(len(rows), device=dev)
        r = r_of[(side[rep] - 1) * len(rep) + j]
        n = unit[rep]
        o.ability[r, n] = slot[rep] if press else -1
        if not press and hold_s > 0:
            until = torch.full((len(rep), env.N), -1.0, device=dev)
            held = torch.full((len(rep), env.N), -1, dtype=torch.long, device=dev)
            until[j, n] = env.st.t + float(hold_s)
            held[j, n] = slot[rep]
            env.ability_hold = (until, held)
        envs[name] = (env, parts, scripted)
    state = (torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    try:
        for i in range(decisions):
            for name in ("pressed", "not"):
                env, parts, scripted = envs[name]
                torch.manual_seed(seed * 7919 + i)
                if i == 0:
                    env.replay(parts, scripted)
                else:
                    env.step(actor)
    finally:
        torch.set_rng_state(state[0])
        if state[1] is not None:
            torch.cuda.set_rng_state_all(state[1])
    s_rep = side[rep]
    for name in ("pressed", "not"):
        ends[name] = trade(envs[name][0].st.u, s_rep, template.weights.rout_share)
    budget = reward.budget(envs["not"][0].st.u)
    return (ends["pressed"] - ends["not"]).reshape(b, copies), budget.reshape(b, copies)


def kinds_of_ctrl(ctrl):
    """[b, 2] controllers -> [b] each battle's kind: who plays each side, the learner's network (L), a past version (P)
    or a script, as L1 x 8 + P1 x 4 + L2 x 2 + P2 (10: self-play; 9 / 6: L against P / P against L; 8 / 2: L against a
    script / a script against L). The same counts of each kind give the same learner and past rows on each side: the
    compiled steps' shapes (rollout.placement splits the rows by side)."""
    L, P = league.LEARNER, league.CODE["past"]
    c = ctrl.long()
    return (c[:, 0] == L) * 8 + (c[:, 0] == P) * 4 + (c[:, 1] == L) * 2 + (c[:, 1] == P)


def kind_of(snap):
    """[b] each snapshot battle's kind (kinds_of_ctrl)."""
    return kinds_of_ctrl(snap["b/ctrl"])


def learner_rows(kind):
    """The learner's rows of a battle of this kind (kinds_of_ctrl): 1 or 2."""
    return (kind >> 3) + ((kind >> 1) & 1)


def composition(ctrl, n):
    """{kind: slots} of a branch batch of n slots: the kinds of the battles ctrl [B, 2] (the main batch) by their share
    of its learner rows (where the presses come from), each kind present at least 1 slot (largest remainders)."""
    rows = {}
    for k in kinds_of_ctrl(ctrl).tolist():
        rows[k] = rows.get(k, 0) + learner_rows(k)
    ks = sorted(rows)
    n = max(int(n), len(ks))
    total = sum(rows.values())
    want = {k: n * rows[k] / total for k in ks}
    out = {k: max(1, int(math.floor(want[k]))) for k in ks}
    for k in sorted(ks, key=lambda k: out[k] - want[k]):          # the largest remainders first
        if sum(out.values()) >= n:
            break
        out[k] += 1
    while sum(out.values()) > n:                                   # (the minimum of 1 overfilled)
        out[max(ks, key=lambda k: out[k])] -= 1
    return out


def pad_snap(template, snap, src, kinds):
    """Padding of a branch batch: battle src (an index into snap) once per entry of `kinds`, relabelled to a battle of
    that kind of the template (its controllers, scripts, layout; the active past slot) and ended (frozen: the steps
    change nothing; its result is dropped). -> (snapshot dict, unit, side, slot): side a side the learner's network
    plays, slot -1 (nothing pressed)."""
    dev = template.device
    sel = select(snap, torch.full((len(kinds),), int(src), dtype=torch.long, device=dev))
    first = {}
    for b, k in enumerate(kinds_of_ctrl(template.ctrl).tolist()):
        first.setdefault(k, b)
    il = [first[k] for k in kinds]
    rep = torch.tensor(il, dtype=torch.long, device=dev)
    sel["b/ctrl"] = template.ctrl[rep].clone()
    sel["b/by_rule"] = template.by_rule[rep].clone()
    sel["b/is_past"] = template.is_past[rep].clone()
    sel["b/battle_slot"] = torch.full_like(sel["b/battle_slot"], int(template.active))
    for k in ("scene", "learner", "opponent"):
        sel["lay/" + k] = np.asarray(getattr(template.layout, k))[il]
    sel["st/done"] = torch.ones_like(sel["st/done"])
    side = (sel["b/ctrl"][:, 0] != league.LEARNER).long() + 1
    unit = torch.zeros(len(kinds), dtype=torch.long, device=dev)
    return sel, unit, side, torch.full_like(unit, -1)


def choice(info, r, n):
    """The moments' (rows r, units n [k] of the decision's learner rows) slot, whether it was pressed, and the
    log-probability of the head's choice: a press's slot, else the ready slot the head likes best (its choice "none")."""
    obs, action = info["obs"], info["action"]
    ab = torch.nan_to_num(info["logits"]["ability"].float(), nan=hd.NEG)
    was = action.ability[r, n]
    pressed = was >= 0
    usable = obs["abil_ok"][r, n]                                       # [k, SLOTS]
    best = torch.where(usable, ab[r, n, 1:], torch.full_like(ab[r, n, 1:], hd.NEG)).argmax(-1)
    lp = torch.log_softmax(ab[r, n], -1).gather(1, (was.clamp(min=-1) + 1)[:, None])[:, 0]
    return torch.where(pressed, was, best), pressed, lp


@dataclass
class _Chunk:
    T: int = 0
    t: int = 0
    snaps: list = field(default_factory=list)
    entries: list = field(default_factory=list)      # (t, r, n, side, slot, press, battle index in the snapshots, lp,
    #                                                    the moment's index: the critic's, else -1)
    moments: list = field(default_factory=list)      # with the critic: per decision (t, r, n, slot, press, lp, budget,
    #                                                    features) of every press and missed moment ([M] tensors)
    n_moments: int = 0
    n_battles: int = 0
    presses: object = None                           # [] on the device: the learner's presses this chunk
    missed: object = None                            # ... the moments a ready ability was not pressed
    taken: int = 0
    room: dict = None                                # {kind: slots left} with a cap (the batch's composition)


class Brancher:
    """Captures snapshots during a chunk (rollout.Battles.step -> capture) and prices them at its end (finish)."""

    def __init__(self, cfg=Config(), device="cpu"):
        self.cfg = cfg
        self.critic = None
        if cfg.critic:
            from tools.nn.train import ability_critic
            self.critic = ability_critic.AbilityCritic(device, buffer=cfg.critic_buffer, warmup=cfg.critic_warmup,
                                                       steps=cfg.critic_steps, seed=cfg.seed)
        self.gen = torch.Generator(device=device).manual_seed(int(cfg.seed) + 104729)
        self.chunk = _Chunk()
        self.stats = {}
        self.calls = 0
        self.copies = None                       # the last chunk's prices per copy [E, copies] (_play)
        self.comp = None                         # {kind: slots} of a branch batch (composition), set at begin
        self.kind_b = None                       # [B] each main battle's kind, on the host

    def slots(self):
        """Snapshots a branch batch holds."""
        cfg = self.cfg
        return max(1, min(cfg.batch, cfg.cap) if cfg.cap else cfg.batch)

    def begin(self, env, T):
        z = torch.zeros((), device=env.device)
        if self.kind_b is None or len(self.kind_b) != env.B:
            self.comp = composition(env.ctrl, self.slots())
            self.kind_b = kinds_of_ctrl(env.ctrl).tolist()
        room = None
        if self.cfg.cap:
            rounds = max(1, math.ceil(self.cfg.cap / sum(self.comp.values())))
            room = {k: v * rounds for k, v in self.comp.items()}
        self.chunk = _Chunk(T=T, presses=z.clone(), missed=z.clone(), room=room)

    def capture(self, env, info):
        c = self.chunk
        t = c.t
        c.t += 1
        if c.presses is None:                      # (a step outside collect: not counted)
            return
        obs, action, logits = info["obs"], info["action"], info["logits"]
        if action.ability is None or "ability" not in logits or obs.get("abil_ok") is None:
            return
        rows = env.rows_learn
        live = obs["ctrl"] & ~env.st.done[rows % env.B][:, None]
        press = live & (action.ability >= 0)
        missed = live & (action.ability < 0) & obs["abil_ok"].any(-1)
        c.presses += press.sum()
        c.missed += missed.sum()
        draw = torch.rand(press.shape, generator=self.gen, device=press.device)
        take = (press & (draw < self.cfg.share)) | (missed & (draw < self.cfg.missed))
        where = None
        if self.critic is not None:
            where = self._moments(env, info, press | missed, t)
        nz = take.nonzero()                                                 # [k, 2] (r, n): reads the GPU
        if not len(nz):
            return
        if self.cfg.cap:
            left = max(0, self.cfg.cap - c.taken)
            per = max(1, math.ceil(self.cfg.cap / max(1, c.T)))
            m = min(left, per)
            # at random, within each battle kind's slots of the branch batch (composition)
            order = torch.randperm(len(nz), generator=self.gen, device=self.gen.device).tolist()
            kb = [self.kind_b[i] for i in (rows[nz[:, 0]] % env.B).tolist()]
            keep = []
            for i in order:
                if len(keep) >= m:
                    break
                if c.room.get(kb[i], 0) > 0:
                    c.room[kb[i]] -= 1
                    keep.append(i)
            if not keep:
                return
            nz = nz[torch.tensor(keep, dtype=torch.long, device=nz.device)]
        r, n = nz[:, 0], nz[:, 1]
        slot, pressed, lp = choice(info, r, n)
        moment = where[r, n] if where is not None else torch.full_like(r, -1)
        b = rows[r] % env.B
        side = rows[r] // env.B + 1
        ub, inv = torch.unique(b, return_inverse=True)
        c.snaps.append(snapshot(env, ub, info))
        at = c.n_battles + inv
        c.n_battles += len(ub)
        c.taken += len(nz)
        for x in zip(r.tolist(), n.tolist(), side.tolist(), slot.tolist(), pressed.tolist(), at.tolist(), lp.tolist(),
                     moment.tolist()):
            c.entries.append((t,) + x)

    def _moments(self, env, info, mask, t):
        """The critic's record of every moment of the decision (mask [R, N]: a press or a ready ability left) ->
        [R, N] each unit's moment index (-1: none)."""
        from tools.nn.train import ability_critic
        c = self.chunk
        where = torch.full(mask.shape, -1, dtype=torch.long, device=mask.device)
        mom = mask.nonzero()                                                # reads the GPU
        if not len(mom):
            return where
        r, n = mom[:, 0], mom[:, 1]
        slot, pressed, lp = choice(info, r, n)
        b = env.rows_learn[r] % env.B
        X = ability_critic.features(info["obs"], r, n, slot)
        c.moments.append((torch.full_like(r, t), r, n, slot, pressed, lp, reward.budget(env.st.u)[b], X))
        where[r, n] = torch.arange(len(r), device=r.device) + c.n_moments
        c.n_moments += len(r)
        return where

    @torch.no_grad()
    def finish(self, env, actor):
        """Plays the chunk's branches -> {"adv", "mask", "lp"} [T, R, N] (ppo.update's ability term), or None when no
        moment is priced; the stats for the log in self.stats. With the critic ready, every moment of the chunk is
        priced (module doc "The ability critic")."""
        c = self.chunk
        t0 = time.time()
        presses, missed = (int(x) for x in torch.stack([c.presses, c.missed]).tolist())
        R, N = len(env.rows_learn), env.N
        self.stats = {"presses": presses, "missed_moments": missed, "replayed_presses": 0, "replayed_missed": 0}
        if not c.entries and not (self.critic is not None and c.n_moments):
            self.chunk = _Chunk()
            return None
        dev = env.device
        out = {"adv": torch.zeros(c.T, R, N, device=dev), "mask": torch.zeros(c.T, R, N, dtype=torch.bool, device=dev),
               "lp": torch.zeros(c.T, R, N, device=dev)}
        price = torch.zeros(0, device=dev)
        if c.entries:
            price, se, budget = self._play(env, actor)
            pressed = torch.tensor([e[5] for e in c.entries], device=dev)
            t = torch.tensor([e[0] for e in c.entries], device=dev)
            r = torch.tensor([e[1] for e in c.entries], device=dev)
            unit = torch.tensor([e[2] for e in c.entries], device=dev)
            lp = torch.tensor([e[7] for e in c.entries], device=dev)
            rms = float(price.pow(2).mean().sqrt())
            gold = price * budget.mean(1)
            p, m = pressed, ~pressed
            st = self.stats
            st.update(replayed_presses=int(p.sum()), replayed_missed=int(m.sum()), price_rms=round(rms, 5),
                      price_se=round(float(se.mean()), 5))
            if bool(p.any()):
                st.update(price_press=round(float(price[p].mean()), 5), gold_press=round(float(gold[p].mean()), 1),
                          zero_press=round(float((price[p].abs() < ZERO).float().mean()), 3),
                          good_press=round(float((price[p] > ZERO).float().mean()), 3))
            if bool(m.any()):
                st.update(price_missed=round(float(price[m].mean()), 5), gold_missed=round(float(gold[m].mean()), 1),
                          good_missed=round(float((price[m] > ZERO).float().mean()), 3))
            self.last = {"price": price, "se": se, "pressed": pressed, "entries": list(c.entries)}
            # the replayed moments by their measured price (all there is without the critic or before it is ready)
            adv = (torch.where(pressed, price, -price) / max(rms, ADV_FLOOR)).clamp(-ADV_CLIP, ADV_CLIP)
            out["adv"][t, r, unit] = adv
            out["mask"][t, r, unit] = True
            out["lp"][t, r, unit] = lp
        if self.critic is not None and c.n_moments:
            self._criticise(out, price)
        self.stats["seconds"] = round(time.time() - t0, 2)
        self.chunk = _Chunk()
        return out if bool(out["mask"].any()) else None

    def _play(self, env, actor):
        """The branches of the chunk's entries -> (price [E] budget share, its standard error over the copies [E],
        budget [E, copies] gold)."""
        c, cfg = self.chunk, self.cfg
        dev = env.device
        snap = concat(c.snaps)
        E = len(c.entries)
        at = torch.tensor([e[6] for e in c.entries], device=dev)
        unit = torch.tensor([e[2] for e in c.entries], device=dev)
        side = torch.tensor([e[3] for e in c.entries], device=dev)
        slot = torch.tensor([e[4] for e in c.entries], device=dev)
        kinds = kind_of(snap)[at].tolist()
        decisions = max(1, int(round(cfg.horizon_s / env.decision_s)))
        K = max(1, int(cfg.copies))
        diff = torch.zeros(E, K, device=dev)
        budget = torch.ones(E, K, device=dev)
        played = 0
        comp = self.comp or composition(env.ctrl, self.slots())
        todo = {k: [i for i, kk in enumerate(kinds) if kk == k] for k in comp}
        assert sum(map(len, todo.values())) == E, "a snapshot of a battle kind the main batch does not have"
        _release(dev)          # the update's cached blocks back to the driver: the branches' batches fit (module doc)
        while any(todo.values()):
            # one batch of the fixed composition: each kind's snapshots, then its free slots as padding
            part, fill = [], []
            for k in sorted(comp):
                part += todo[k][:comp[k]]
                fill += [k] * (comp[k] - len(todo[k][:comp[k]]))
                todo[k] = todo[k][comp[k]:]
            ix = torch.tensor(part, device=dev)
            sub = select(snap, at[ix])
            u_, s_, k_ = unit[ix], side[ix], slot[ix]
            if fill:
                ps, pu, pside, pslot = pad_snap(env, sub, 0, fill)
                sub = concat([sub, ps])
                u_, s_, k_ = torch.cat([u_, pu]), torch.cat([s_, pside]), torch.cat([k_, pslot])
            self.calls += 1
            d, bg = play_pair(env, actor, sub, u_, s_, k_, K, decisions, cfg.seed + self.calls, cfg.hold_s)
            diff[ix] = d[:len(part)]
            budget[ix] = bg[:len(part)]
            played += 2 * K * (len(part) + len(fill))
        _release(dev)          # ... and the branches' for the update
        self.stats.update(branch_battles=played, branch_battle_s=round(played * decisions * env.decision_s))
        self.copies = diff
        rel = reliability(diff)
        if rel is not None:
            self.stats["price_rel"] = round(rel, 3)
        price = diff.mean(1)                                                # [E] budget share
        se = diff.std(1) / math.sqrt(K) if K > 1 else torch.zeros_like(price)
        return price, se, budget

    def target(self, price):
        """The critic's regression target of the replayed moments: their measured price. Extension point: a bootstrapped
        (TD) target, e.g. the price plus the critic's value of the unit's next ready moment."""
        return price

    def _criticise(self, out, price):
        """The ability critic (module doc). Its prediction of every moment of the chunk comes first, by the critic as
        the last update left it (the replayed moments' measured prices are not in it yet: the control variate stays
        independent of them); once ready, every moment gets its price as the head's advantage (out, in place). Then
        the replayed prices go into its buffer and it fits."""
        from tools.nn.train import ability_critic
        c, cfg, crit = self.chunk, self.cfg, self.critic
        t, r, n, slot, pressed, lp, budget, X = (torch.cat(x) for x in zip(*c.moments))
        e_m = torch.tensor([e[8] for e in c.entries], dtype=torch.long, device=X.device)
        t1 = time.time()
        st = self.stats
        on = crit.ready()
        q = crit.predict(X) if "train" in crit.last else torch.zeros(len(X), device=X.device)
        st["critic_on"] = int(on)
        st["moments"] = len(X)
        gold = q * budget                                                   # the predicted price in gold
        if bool(pressed.any()):
            st["q_press"] = round(float(gold[pressed].mean()), 2)
        if bool((~pressed).any()):
            st["q_missed"] = round(float(gold[~pressed].mean()), 2)
        st["q_std"] = round(float(gold.std()), 2) if len(gold) > 1 else 0.0
        if on:
            scale = max(crit.rms(), ADV_FLOOR)
            used = q * cfg.critic_weight
            if len(e_m):
                lim = cfg.critic_resid * crit.rms()
                cv = q[e_m] + (price - q[e_m]).clamp(-lim, lim)
                sd_p, sd_cv = (float(x.std()) if len(x) > 1 else 0.0 for x in (price, cv))
                use_cv = sd_cv <= sd_p
                used[e_m] = cv if use_cv else price
                st.update(price_sd=round(sd_p, 5), cv_sd=round(sd_cv, 5), cv_used=int(use_cv))
            adv = (torch.where(pressed, used, -used) / scale).clamp(-ADV_CLIP, ADV_CLIP)
            out["adv"][t, r, n] = adv
            out["mask"][t, r, n] = True
            out["lp"][t, r, n] = lp
        st["covered"] = round(float(out["mask"].sum()) / len(X), 4)
        if len(e_m):
            d = self.copies
            K = d.shape[1] if d is not None and len(d) == len(e_m) else 0
            halves = torch.stack([d[:, :K // 2].mean(1), d[:, K // 2:].mean(1)], 1) if K >= 2 else None
            crit.add(X[e_m], self.target(price), halves)
        st.update({"critic_" + k: v for k, v in ability_critic.rounded(crit.fit()).items()})
        rel = crit.reliability()
        if rel is not None:
            st["critic_rel"] = round(rel, 3)
        st["critic_s"] = round(time.time() - t1, 2)


def reliability(d):
    """The real share of the prices' spread (it differs between moments) against the copies' noise: d [E, K] the
    prices per copy -> 1 - mean(variance over the copies / K) / variance of the price, at least 0 (None with fewer than 2
    copies or 8 moments). It bounds what any critic can explain of the measured price (module doc)."""
    E, K = d.shape
    if K < 2 or E < 8:
        return None
    v = float(d.mean(1).var())
    if v <= 0:
        return 0.0
    return max(0.0, 1.0 - float((d.var(1) / K).mean()) / v)


def _release(dev):
    """The caching allocator's free blocks back to the driver (CUDA). The branches' two batches come on top of the main
    batch and its chunk: with the update's cache still held, video memory overflowed into shared system memory (Windows
    lets it rather than failing) and a step went from 0.06 s to ~1 s."""
    if torch.device(dev).type == "cuda":
        torch.cuda.empty_cache()


def weight(w0, decay, updates):
    """The ability term's weight after `updates` updates: w0 down to 0 linearly over `decay` updates (0: no decay)."""
    if not decay:
        return float(w0)
    return float(w0) * max(0.0, 1.0 - updates / float(decay))

"""Many simulated battles at once, played by the learner and its opponents (docs/en/training/training.md).

Battles(layout) holds B battles on one device. A row is one side of one battle: row = (side - 1) * B
+ battle, so both sides' observations stack into one batch of 2B rows. Each row has a controller
(tools/nn/train/league.py): the learner, a past version, or a script. Every step() is one decision
of the networks, as often as in the game (tools/nn/train/cadence.py: by default every 1 s of battle =
2 simulator steps of 0.5 s, the orders landing ~0.36 s late):

    observe both sides -> the learner acts for its rows (and the past version for its rows) ->
    the simulator steps until the next decision (the networks' orders land after the latency, KEEP
    otherwise; the scripts give theirs every simulator step) -> reward summed over them ->
    finished battles start again (auto_reset) with fresh randomised numbers.

The observation (and its memory: last sightings, speeds) is taken only at decisions, as the companion
takes the game's state once a second.

step() returns the learner's transition: its observation, action, log-probabilities, the critic's
value, reward and done, per learner row. collect() stacks T of them with the memory the chunk
began with, so the update can run the memory (GRU) through the chunk again.

Abilities: the networks (learner and past version) choose their lords' abilities (heads.sample with
abilities=True; Orders.ability); only the scripted opponents' lords fire by the game-AI rule
(sim abilities.set_rule, again after every restart: the bank's rows bring scenario.build's `ai`).
A transition keeps only the abilities' state (abil [.., SLOTS, DYNAMIC]) and the battle's bank row
(abil_row); full_obs() puts the passports back from the bank (batch["abil_static"]) for the update:
the whole input would be ~3 GB for 1024 battles x 64 decisions.

The drills' teacher (teach= {drill name: its skilled script}; tools/nn/train/drills/teach.py): at every
decision the script labels the learner's units in that drill's battles on the same state (it does
not act); the transition carries the labels ("teach") for PPO's imitation term. Only a share of a drill's
battles may count (set_teach_shares: the adaptive teacher, tools/nn/train/teach_auto.py; default all):
every battle draws a number when it starts, and it is labelled while that number is below its drill's
share (a battle is labelled whole; a larger share keeps the battles a smaller one had). A teach= value may be
a Drill (drills.Drill): then its `teacher` script labels (default `skilled`), and in an EMBEDDED battle of the
drill (drills.embedded_rows) only at its `moments` (the situation; the clean and broad frames: every unit).
The teacher in normal battles (teach_normal= {drill name: Drill}; run.py --teach-normal): in the rows of no
drill (the ordinary training battles) the drill's teacher script labels our units at the drill's `moments`
only (e.g. kiting: a slower melee enemy closing in, and the run-back it keeps going), under the name
"<drill>@normal" with a share of its own (the same per-battle draw). The labels' drill index and the share's
pick are per unit ([R, N]). A teacher whose share is 0 labels nothing: its script runs only every TEACH_PROBE-th
decision (~20 % of a decision's time otherwise), its labels there only measure the agreement (PPO's log).
"""
import warnings

import numpy as np
import torch

from tools.nn.model import abilities as mab
from tools.nn.model import heads as hd
from tools.nn.model import observation as ob
from tools.nn.model import policy
from tools.nn.model.decide import to_orders
from tools.nn.model.frame import Frame
from tools.nn.sim import abilities as sim_abilities
from tools.nn.sim import battle
from tools.nn.sim import orders as O
from tools.nn.sim import state as S
from tools.nn.sim.params import load
from tools.nn.train import drills, league, opponents, randomise, reward, scenes
from tools.nn.train import cadence as cad
from tools.nn.train.drills import teach as drill_teach

CRITIC_KEYS = ("tokens", "ctx", "own", "attend", "pos")
FRAME = ("cx", "cz", "ux", "uz")
MEMORY = ("last_x", "last_z", "last_t", "seen", "dead", "prev_x", "prev_z", "prev_t", "prev_vis", "last_melee_t",
          "last_rout_t", "lord_dead_t", "prev_hp", "hit_t", "prev_gold", "rate", "rate_t",
          "prev_ammo", "volley_t")
ROLES = ("attack", "defend")
# Outcome counters: the opponents, and the untrained network as "past" on its own.
STAT_NAMES = league.OPPONENTS + ("untrained",)
UNTRAINED = len(league.OPPONENTS) + 1
# A teacher with a share of 0 runs its script every this many decisions only (the agreement is still measured)
TEACH_PROBE = 8


_OBSERVE = {}
_FAST = {}


def fast(fn, use):
    """fn compiled by torch.compile (one graph per batch shape) when use, else fn itself. The
    bookkeeping of a decision (the scripts, rewards, counters) is hundreds of small operations on
    [B, N] tensors: on the GPU their launches, not the work, took most of the time."""
    if not use:
        return fn
    if fn not in _FAST:
        torch._dynamo.config.recompile_limit = max(64, torch._dynamo.config.recompile_limit)
        _FAST[fn] = torch.compile(fn, dynamic=False)
    return _FAST[fn]


def observer(device, compile=None):
    """observation.observe, compiled by torch.compile on CUDA (~14x faster: it is hundreds of small
    operations on [B, N] tensors)."""
    use = compile if compile is not None else torch.device(device).type == "cuda"
    if not use:
        return ob.observe
    if "f" not in _OBSERVE:
        # One graph per batch shape and view (actor, critic): evaluation builds batches of new sizes.
        torch._dynamo.config.recompile_limit = max(64, torch._dynamo.config.recompile_limit)
        _OBSERVE["f"] = torch.compile(ob.observe, dynamic=False)
    return _OBSERVE["f"]


def params_with_limit(limit_s=None):
    """The simulator's numbers, with the battle's time limit changed (None: as in config/nn/sim.json)."""
    p = load()
    return p if limit_s is None else p.with_cal("battle_limit_s", value=float(limit_s))


def stack(obs1, obs2, device):
    """Two sides' observations -> (dict of [2B, ...] tensors, Frame of [2B])."""
    d1, d2 = policy.to_torch(obs1, device), policy.to_torch(obs2, device)
    out = {k: torch.cat([d1[k], d2[k]]) for k in d1}
    frame = Frame(*(torch.cat([torch.as_tensor(getattr(obs1.frame, k), device=device).float(),
                               torch.as_tensor(getattr(obs2.frame, k), device=device).float()]) for k in FRAME))
    return out, frame


def rows_of(d, rows):
    return {k: v[rows] for k, v in d.items()}


def frame_rows(frame, rows):
    return Frame(*(getattr(frame, k)[rows] for k in FRAME))


def merge_memory(old, new, rows):
    """Memory rows [B] (bool) from `new`, the rest from `old`."""
    def w(a, b):
        return torch.where(rows.reshape(-1, *([1] * (a.dim() - 1))), b, a)
    frame = Frame(*(w(getattr(old.frame, k), getattr(new.frame, k)) for k in FRAME))
    return ob.Memory(frame, *(w(getattr(old, k), getattr(new, k)) for k in MEMORY))


def full_obs(obs, table):
    """A stored observation (rows of collect's batch["obs"], [..]) with the abilities' passports put back:
    abil [.., N, SLOTS, DYNAMIC] + table[abil_row] -> abil [.., N, SLOTS, SIZE]; abil_row dropped.
    An observation without abilities is returned as it is."""
    if "abil_row" not in obs:
        return obs
    out = {k: v for k, v in obs.items() if k != "abil_row"}
    out["abil"] = torch.cat([obs["abil"], table[obs["abil_row"]]], -1)
    return out


def placement(rows, B):
    """Where the rows [R] (row = (side - 1) * B + battle) go in the batch: ((side, battles, rows' positions)
    for side 1 and 2), as index tensors made once (boolean indexing reads its size back from the GPU)."""
    b, two = rows % B, rows >= B
    return tuple((s, b[sel], torch.nonzero(sel).squeeze(1)) for s, sel in ((1, ~two), (2, two)))


def scripts_of(layout):
    """{opponent name: script} the layout's rows may need: the scripted opponents, and the enemy scripts
    of the drills it plays (tools/nn/train/drills)."""
    out = dict(opponents.SCRIPTS)
    used = set(np.unique(layout.opponent).tolist())
    names = [n for n in drills.NAMES if league.CODE[drills.opponent(n)] in used]
    if names:
        out.update(drills.enemy_scripts(drills.load(names)))
    return out


def assemble_orders(st, ctrl, scripts, parts):
    """The batch's Orders [B, N]: scripts ((code, script), ...) give the orders of the sides whose
    controller (ctrl [B, 2]) is their code; parts ((placement, Orders [rows, N]), ...) those of the
    networks' rows."""
    B, N = st.B, st.N
    side = {s: O.hold(B, N, st.device) for s in (1, 2)}
    for code, script in scripts:
        o = script(st)
        for s in (1, 2):
            use = (ctrl[:, s - 1] == code)[:, None].expand(B, N)
            side[s] = O.merge(side[s], o, use)
    for place, o in parts:
        for s, b, i in place:
            for k in O.FIELDS:
                getattr(side[s], k)[b] = getattr(o, k)[i]
    return O.merge(side[1], side[2], st.u["side"] == 2)


def _teacher_of(f):
    """(script, moments or None) of a teach= value: a script, or a drills.Drill (its teacher, else skilled)."""
    if isinstance(f, drills.Drill):
        return (f.teacher or f.skilled), f.moments
    return f, None


def learner_units(u, ctrl):
    """[B, N] the units of the sides the learner plays (ctrl [B, 2])."""
    side = u["side"]
    learner = torch.zeros_like(side, dtype=torch.bool)
    for s in (1, 2):
        learner = learner | ((side == s) & (ctrl[:, s - 1] == league.LEARNER)[:, None])
    return learner


def _orders_cost(st, orders, was_done, ctrl, weights, orders_stats, order_battle):
    """Before the step: what each side pays for its order changes [B, 2] and the target switches [B, N];
    counts the learner's order changes and standing unit-steps (in place: orders_stats, order_battle)."""
    u = st.u
    changes = reward.order_changes(u, orders, weights.order_move_m) & ~was_done[:, None]
    switched = reward.retargets(u, orders) & ~was_done[:, None]
    cost = reward.order_cost(changes, u["side"], weights, switched)
    learner = learner_units(u, ctrl)
    standing = (u["side"] > 0) & (u["men"] > 0) & ~u["gone"] & ~u["r"] & learner & ~was_done[:, None]
    c, n = (changes & learner).float().sum(1), standing.float().sum(1)
    orders_stats += torch.stack([c.sum(), n.sum()])
    order_battle[:, 0] += c
    order_battle[:, 2] += n
    return cost, switched


def _rewards(st, health, was_done, hit_rate, last_hit, cost, weights, dt, rb, rs, attacks, part_stats, part_steps,
             timeouts):
    """After the step: (measure after, finished [B], the terms {PARTS: [B, 2]}, the reward [B, 2], the
    attacker's damage rate and last hit [B]); counts the learner's reward by term and role, and the
    timeouts (in place)."""
    after = reward.measure(st, weights.rout_share, weights.lord_rout)
    finished = st.done & ~was_done
    terms = reward.parts(health, after, finished, st.winner, weights)
    if weights.idle_rate > 0:
        # only a steady damage rate counts as attacking (reward.idle_cost: idle_rate)
        hit_rate = reward.hit_rate(hit_rate, health, after, st.attacker, dt, weights.idle_window_s)
        hit = hit_rate >= weights.idle_rate
    else:
        hit = reward.struck(health, after, st.attacker)
    last_hit = torch.where(hit, st.t, last_hit)
    terms["idle"] = -reward.idle_cost(st, weights, last_hit) * (~finished).float()[:, None]
    terms["orders"] = -cost
    r = sum(terms[k] for k in reward.PARTS)
    timeouts += (finished & (after[:, 0, 1] > 0) & (after[:, 1, 1] > 0)).sum()
    live = (~was_done[rb]).float()
    role = 1 - attacks.long()                                    # 0 the learner attacks, 1 defends
    part_stats.index_add_(0, role, torch.stack([terms[k][rb, rs] for k in reward.PARTS], 1) * live[:, None])
    part_steps.index_add_(0, role, live)
    return after, finished, terms, r, hit_rate, last_hit


def _decide(actor, obs, h, frame, bounds, greedy):
    """The actor's decision: (logits, sampled Action, Orders, new memory). Compiled with fast(), the
    network's weights are the graph's inputs (the learner, the past version and their updates share
    it); the sampling then draws from the compiled code's own random stream (the same distribution)."""
    logits, h_new = actor(obs, h)
    action = hd.sample(logits, greedy, abilities=True)
    return logits, action, to_orders(actor.cfg, action, obs, frame, bounds), h_new


def _log_prob(logits, action, ctrl):
    return hd.log_prob(logits, action, ctrl)[0]


def _values(critic, obs):
    """The critic's value [R]."""
    return critic(obs)


def open_rows(source, want=None):
    """(State, LiveSetup, bank rows) of a batch that starts with the battles the source picks."""
    idx = source.pick(want)
    b, bs = source.bank.state, source.bank.setup
    st = S.State({k: v[idx].clone() for k, v in b.u.items()}, b.t[idx].clone(), b.attacker[idx].clone(),
                 b.done[idx].clone(), b.winner[idx].clone(), b.lord_dead_s[idx].clone(), b.bounds,
                 [list(b.keys[int(i)]) for i in idx.tolist()])
    arrays = ob._Arrays(**{k: getattr(bs.arrays, k)[idx].clone() for k in scenes.LiveSetup.FIELDS})
    setup = scenes.LiveSetup(arrays, {s: bs.char[s][idx].clone() for s in (1, 2)},
                             [bs.factions[int(i)] for i in idx.tolist()])
    return st, setup, idx


def restart_rows(st, setup, source, rows, want=None):
    """Battles where rows [B] is true start again as battles the source picks (in place). -> bank rows."""
    idx = source.pick(want)
    scenes.take_rows(st, source.bank.state, rows, idx)
    setup.take(source.bank.setup, rows, idx)
    return idx


class Battles:
    def __init__(self, layout, scene_list=scenes.SCENES, device="cpu", params=None, spread=randomise.Spread(),
                 weights=reward.Weights(), seed=0, auto_reset=True, compile=None, source=None, cadence=None,
                 teach=None, teach_normal=None, attack_only=league.ATTACK_ONLY):
        self.device = torch.device(device)
        self.params = params or load()
        # how often the networks decide and how late their orders land (cadence.py; default: the game's)
        self.cadence = cadence or cad.GAME
        self.k = self.cadence.steps(self.params.dt)
        self.cadence.delay(self.params.dt)                     # (checks that the latency fits a decision)
        self.layout = layout
        self.spread = spread
        self.weights = weights
        if weights.idle_rate > 0 and (weights.idle_rate, weights.idle_window_s, weights.rout_share) != (
                ob.RATE_MIN, ob.RATE_WINDOW, ob.ROUT_SHARE):
            warnings.warn(f"the idle clock (idle_rate {weights.idle_rate}, idle_window_s {weights.idle_window_s}, "
                          f"rout_share {weights.rout_share}) is not the one the observation shows (PROGRESS: "
                          f"{ob.RATE_MIN}, {ob.RATE_WINDOW}, {ob.ROUT_SHARE})")
        self.auto_reset = auto_reset
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        # Where battles come from: the fixed scenes (battle b plays layout.scene[b]) or generated armies.
        self.source = source or scenes.Fixed(layout.scene, scene_list, self.params, self.device)
        self.bank = self.source.bank
        self.B, self.N = layout.B, self.bank.N
        self.advance = battle.stepper(self.device, compile)
        self.look = observer(self.device, compile)
        # the bookkeeping around the simulator's step, compiled like it on CUDA (fast())
        self.compiled = use = compile if compile is not None else self.device.type == "cuda"
        self._assemble = fast(assemble_orders, use)
        self._orders_cost = fast(_orders_cost, use)
        self._rewards = fast(_rewards, use)
        self._measure = fast(reward.measure, use)
        self._track = fast(reward.track, use)
        self._decide = fast(_decide, use)
        self._log_prob = fast(_log_prob, use)
        self._values = fast(_values, use)
        # the attack_only opponents (`hold`; training: run.py --defend-only) are met only as the defender:
        # their battles must have the learner attacking.
        only = np.isin(layout.opponent, [league.CODE[n] for n in attack_only])
        self.want = torch.as_tensor(np.where(only, layout.learner, 0), device=self.device)
        self.ctrl = torch.as_tensor(layout.controllers(), device=self.device)           # [B, 2]
        flat = torch.cat([self.ctrl[:, 0], self.ctrl[:, 1]])
        self.rows_learn = (flat == league.LEARNER).nonzero().squeeze(1)
        self.rows_past = (flat == league.CODE["past"]).nonzero().squeeze(1)
        self.place_learn = placement(self.rows_learn, self.B)
        self.place_past = placement(self.rows_past, self.B)
        self.scripts = {league.CODE[n]: f for n, f in scripts_of(layout).items() if bool((self.ctrl == league.CODE[n]).any())}
        opp = torch.as_tensor(layout.opponent, device=self.device)
        self.row_opp = torch.cat([opp, opp])[self.rows_learn]                          # [R]
        drill_codes = torch.tensor([league.CODE[drills.opponent(n)] for n in drills.NAMES], device=self.device)
        self.row_normal = ~torch.isin(self.row_opp, drill_codes)                         # [R] a row of no drill
        self.past_actor = None
        self.past_untrained = False
        # the drills' teacher: {league code: (index, script, moments or None)} of the taught drills the layout
        # plays; the teacher in normal battles: ((index, script, moments), ...) on the rows of no drill
        teach, teach_normal = dict(teach or {}), dict(teach_normal or {})
        self.teach_names = tuple(teach) + tuple(drills.normal_name(n) for n in teach_normal)
        used = set(np.unique(layout.opponent).tolist())
        self.teach = {league.CODE[drills.opponent(n)]: (i, *_teacher_of(f)) for i, (n, f) in enumerate(teach.items())
                      if league.CODE[drills.opponent(n)] in used}
        self.teach_normal = tuple((len(teach) + i, *_teacher_of(d)) for i, d in enumerate(teach_normal.values()))
        # a heavy drill's teacher script (drills.Drill.heavy: direct_fire's position evaluator) runs only on the battles
        # whose units it may label now (a normal row drawn below its share), not on the whole batch
        self.teach_heavy = tuple(bool(getattr(d, "heavy", False)) for d in teach_normal.values())
        assert all(m is not None for _, _, m in self.teach_normal), "a teacher in normal battles needs the drill's moments"
        # the share of each taught drill's battles labelled ([drills], default all) and each battle's draw [B]
        self.teach_share = torch.ones(len(self.teach_names), device=self.device)
        self.share_host = [1.0] * len(self.teach_names)              # the same on the host (no read-back)
        self.decisions = 0                                            # step() calls (TEACH_PROBE)
        self.teach_gen = torch.Generator(device=self.device).manual_seed(seed + 7919)
        self.teach_draw = torch.zeros(self.B, device=self.device)
        self.stats = torch.zeros(2 * (len(STAT_NAMES) + 1), 3, device=self.device)    # games, wins, seconds
        # by (opponent, the learner's role)
        self.row_attacks = None
        self.orders_stats = torch.zeros(2, device=self.device)                         # changes, unit-steps
        self.kind_stats = torch.zeros(len(O.KINDS), dtype=torch.long, device=self.device)   # decisions by kind
        self.lord_stats = torch.zeros(3, device=self.device)        # learner battles ended, own / enemy lord dead
        self.switch_stats = torch.zeros(1, device=self.device)      # the learner's attack target switches
        # The same per battle (evaluation reads them per opponent): decisions by kind [B, kinds], and
        # order changes, target switches and standing unit-steps [B, 3].
        self.kind_battle = torch.zeros(self.B, len(O.KINDS), device=self.device)
        self.order_battle = torch.zeros(self.B, 3, device=self.device)
        # Only the scripted opponents fire their lords' abilities by the game-AI rule; a side a
        # network plays (the learner, self-play, a past version) fires them by its own order.
        self.by_rule = self.ctrl >= league.CODE["nearest"]                              # [B, 2]
        self.ability_stats = torch.zeros(2, device=self.device)   # the learner's ability uses, its ended battles
        # The learner's reward by term (reward.PARTS) and its decisions, by role (attack, defend)
        self.part_stats = torch.zeros(2, len(reward.PARTS), device=self.device)
        self.part_steps = torch.zeros(2, device=self.device)
        self.start()

    @property
    def R(self):
        return len(self.rows_learn)

    def start(self):
        self.st, self.setup, idx = open_rows(self.source, self.want)
        sim_abilities.set_rule(self.st.u, self.by_rule)
        self.bank_row = idx.clone()
        self.bounds2 = torch.cat([self.setup.bounds, self.setup.bounds])
        every = torch.ones(self.B, dtype=torch.bool, device=self.device)
        randomise.apply(self.st, every, self.spread, self.gen)
        state = self.st.observation()
        self.mem = {s: ob.start(state, self.setup, s) for s in (1, 2)}
        self.cmem = {s: ob.start(state, self.setup, s) for s in (1, 2)}
        self.h_learn = None
        self.h_past = None
        self.health = reward.measure(self.st, self.weights.rout_share, self.weights.lord_rout)
        # [B] the battle time of the attacker's last damage (reward.idle_cost), -1 before its first
        self.last_hit = torch.full((self.B,), -1.0, device=self.device)
        self.hit_rate = torch.zeros(self.B, device=self.device)     # [B] reward.hit_rate (with idle_rate)
        self.cur = None
        self.battles = 0
        self.timeouts = 0
        if self.teach_names:
            self.teach_draw = torch.rand(self.B, generator=self.teach_gen, device=self.device)

    def set_teach_shares(self, shares):
        """{drill: the share of its battles the teacher labels} (drills not given keep theirs)."""
        for i, n in enumerate(self.teach_names):
            if n in shares:
                self.share_host[i] = float(min(1.0, max(0.0, shares[n])))
                self.teach_share[i] = self.share_host[i]

    def teach_shares(self):
        return {n: float(self.teach_share[i]) for i, n in enumerate(self.teach_names)}

    def set_past(self, actor, untrained=False):
        self.past_actor = actor
        self.past_untrained = untrained
        self.h_past = None

    # --- observation ---
    def observe(self, critic=True):
        state = self.st.observation()
        obs, cobs = {}, {}
        for s in (1, 2):
            obs[s], self.mem[s] = self.look(state, self.setup, s, self.mem[s])
        a, frame = stack(obs[1], obs[2], self.device)
        c = None
        if critic:
            for s in (1, 2):
                cobs[s], self.cmem[s] = self.look(state, self.setup, s, self.cmem[s], full=True)
            full, _ = stack(cobs[1], cobs[2], self.device)
            c = {k: full[k][self.rows_learn] for k in CRITIC_KEYS}
        return a, frame, c

    # --- orders ---
    def _act(self, actor, a, frame, rows, h, greedy):
        obs_r = rows_of(a, rows)
        if h is None:
            h = actor.initial(obs_r)
        logits, action, orders, h_new = self._decide(actor, obs_r, h, frame_rows(frame, rows), self.bounds2[rows],
                                                     greedy)
        return obs_r, h, logits, action, orders, h_new

    def assemble(self, parts):
        """[(rows, Orders [len(rows), N])] and the scripts -> the batch's Orders [B, N]."""
        places = [(self.place_learn if rows is self.rows_learn else self.place_past if rows is self.rows_past
                   else placement(rows, self.B), o) for rows, o in parts]
        return self._assemble(self.st, self.ctrl, tuple(self.scripts.items()), tuple(places))

    @property
    def decision_s(self):
        """Seconds of battle a step() plays (one decision: self.k simulator steps)."""
        return self.k * self.params.dt

    def _landing(self, orders, rows, land):
        """The networks' rows' Orders of one simulator step of a decision: the decision's own where it
        lands now (land [B] or a bool for all), KEEP elsewhere (the orders in force go on)."""
        if land is True:
            return orders
        keep = O.Orders(torch.full_like(orders.kind, O.KEEP), orders.x, orders.z, orders.target, orders.run,
                        torch.full_like(orders.ability, -1))
        if land is False:
            return keep
        return O.merge(keep, orders, land[rows % self.B][:, None].expand_as(orders.kind))

    # --- one decision and its simulator steps ---
    @torch.no_grad()
    def step(self, actor, critic=None, greedy=False, each=None):
        """One decision of the networks and the self.k simulator steps until the next (cadence.py): their
        orders land after the cadence's latency (KEEP before and after), the scripts give theirs every step.
        The reward and done of the learner rows are summed / any over the steps; finished battles restart
        (auto_reset) after the last. each(live [B]): called after every simulator step (live: the battles
        running before it), e.g. the evaluation's per-step counts."""
        if self.cur is None:
            self.cur = self.observe(critic is not None)
        a, frame, c = self.cur
        abil_row = self.bank_row[self.rows_learn % self.B]           # the battles the observation was made of
        obs_r, h_prev, logits, action, orders, h_new = self._act(actor, a, frame, self.rows_learn, self.h_learn, greedy)
        parts = [(self.rows_learn, orders)]
        h_past_new = None
        if len(self.rows_past) and self.past_actor is not None:
            *_, o_past, h_past_new = self._act(self.past_actor, a, frame, self.rows_past, self.h_past, False)
            parts.append((self.rows_past, o_past))
        lp = self._log_prob(logits, action, obs_r["ctrl"])
        taught = self._teach_labels(actor.cfg, obs_r, frame) if self.teach_names and self.R else None
        self.decisions += 1
        value = None
        if critic is not None:
            value = self._values(critic, c)
        acting = obs_r["ctrl"] & ~self.st.done[self.rows_learn % self.B][:, None]
        # (a comparison, not bincount / one_hot: they read the GPU's answer back to size their output)
        kinds = action.kind[..., None] == torch.arange(len(O.KINDS), device=self.device)
        per_row = (kinds & acting[..., None]).sum(1)                                    # [R, kinds]
        self.kind_stats += per_row.sum(0)
        self.kind_battle.index_add_(0, self.rows_learn % self.B, per_row.float())

        attacks = self.st.attacker[self.rows_learn % self.B] == self.rows_learn // self.B + 1   # [R], this battle
        rb, rs = self.rows_learn % self.B, self.rows_learn // self.B
        # When the decision's orders land: a fixed step, or (a latency between steps) at random per battle.
        whole, frac = self.cadence.delay(self.params.dt)
        delay = self.cadence.delays(self.params.dt, self.B, self.device) if frac else None
        r_rows = None
        finished = torch.zeros_like(self.st.done)
        for j in range(self.k):
            land = (delay == j) if delay is not None and j in (whole, whole + 1) else (j == whole)
            now = [(rows, self._landing(o, rows, land)) for rows, o in parts]
            was_done = self.st.done.clone()
            r, fin = self._sim_step(now, attacks, rb, rs, was_done)
            r_rows = r if r_rows is None else r_rows + r
            finished = finished | fin
            if each is not None:
                each(~was_done)
        d_rows = finished[rb]

        if self.auto_reset:
            self._reset(finished)
        self.health = self._measure(self.st, self.weights.rout_share, self.weights.lord_rout)
        keep = (~finished).float()
        self.h_learn = h_new * keep[rb][:, None, None]
        if h_past_new is not None:
            self.h_past = h_past_new * keep[self.rows_past % self.B][:, None, None]
        self.cur = self.observe(critic is not None)
        if "abil" in obs_r:                          # keep the state, not the passports (full_obs)
            # a copy: a slice (view) would keep the whole input of every step alive (~60 MB each)
            obs_r = dict(obs_r, abil=obs_r["abil"][..., :mab.DYNAMIC].clone(), abil_row=abil_row)
        return {"obs": obs_r, "action": action, "lp": lp, "value": value, "reward": r_rows,
                "done": d_rows, "critic_obs": c, "attacks": attacks, "teach": taught}

    def _teach_labels(self, cfg, obs_r, frame):
        """The teachers' labels of the learner rows at this decision (drills/teach.py): (Action [R, N],
        valid [R, N], drill [R, N]: index into teach_names, -1 for a unit no teacher labels (a taught drill's
        row: all its units), picked [R, N]: the unit's battle is among its teacher's labelled share)."""
        R, N = obs_r["ctrl"].shape
        dev = self.device
        z = torch.zeros((R, N), dtype=torch.long, device=dev)
        a = hd.Action(z.clone(), z.clone(), z - 1, torch.zeros((R, N), dtype=torch.bool, device=dev))
        valid = torch.zeros((R, N), dtype=torch.bool, device=dev)
        drill = torch.full((R, N), -1, dtype=torch.long, device=dev)
        if not self.teach and not self.teach_normal:
            return a, valid, drill, torch.zeros((R, N), dtype=torch.bool, device=dev)
        b = self.rows_learn % self.B
        live = ~self.st.done[b]
        frame_r = frame_rows(frame, self.rows_learn)
        emb = drills.embedded_rows(self.st)[b]                                           # [R]
        # a share of 0 labels nothing: the script runs on the probe decisions only (the agreement)
        idle = (lambda i: self.share_host[i] <= 0 and self.decisions % TEACH_PROBE != 0)
        for code, (i, script, moments) in self.teach.items():
            if idle(i):
                continue
            mine = self.row_opp == code                                                  # [R]
            o = script(self.st)
            lab, ok = drill_teach.label(cfg, o, obs_r, frame_r, self.rows_learn, self.B)
            if moments is not None:                                                      # embedded: the moments only
                ok = ok & (moments(self.st, o)[b] | ~emb[:, None])
            sel = mine[:, None] & ok & live[:, None]
            for f in ("kind", "point", "target", "run"):
                setattr(a, f, torch.where(mine[:, None], getattr(lab, f), getattr(a, f)))
            valid = valid | sel
            drill = torch.where(mine[:, None], torch.full_like(drill, i), drill)
        for k, (i, script, moments) in enumerate(self.teach_normal):
            if idle(i):
                continue
            if self.teach_heavy[k]:
                o, mom = self._heavy_teacher(i, script, moments, b, live)
                if o is None:
                    continue
            else:
                o = script(self.st)
                mom = moments(self.st, o)
            lab, ok = drill_teach.label(cfg, o, obs_r, frame_r, self.rows_learn, self.B)
            sel = self.row_normal[:, None] & ok & mom[b] & live[:, None] & (drill < 0)
            for f in ("kind", "point", "target", "run"):
                setattr(a, f, torch.where(sel, getattr(lab, f), getattr(a, f)))
            valid = valid | sel
            drill = torch.where(sel, torch.full_like(drill, i), drill)
        picked = (drill >= 0) & (self.teach_draw[b][:, None] < self.teach_share[drill.clamp(min=0)])
        return a, valid, drill, picked

    def _heavy_teacher(self, i, script, moments, b, live):
        """(Orders [B, N], moments [B, N]) of a heavy teacher computed only on the battles of the normal learner rows
        drawn below its share now (the rest: hold, no moment); (None, None) when there is none."""
        rows = self.row_normal & live & (self.teach_draw[b] < self.teach_share[i])
        keep = torch.unique(b[rows])
        if keep.numel() == 0:
            return None, None
        sub = S.State({k: v[keep] for k, v in self.st.u.items()}, self.st.t[keep], self.st.attacker[keep],
                      self.st.done[keep], self.st.winner[keep], self.st.lord_dead_s[keep], self.st.bounds,
                      [self.st.keys[j] for j in keep.tolist()] if self.st.keys else [])
        o_s = script(sub)
        m_s = moments(sub, o_s)
        o = O.hold(self.B, self.N, self.device)
        for f in O.FIELDS:
            getattr(o, f)[keep] = getattr(o_s, f)
        mom = torch.zeros((self.B, self.N), dtype=torch.bool, device=self.device)
        mom[keep] = m_s
        return o, mom

    def _sim_step(self, parts, attacks, rb, rs, was_done):
        """One simulator step with the networks' orders `parts` and the scripts': -> (the learner rows'
        reward [R], finished [B]); the counters (in place). A battle that had ended before the step is
        frozen and gets no reward."""
        marks = self._ability_marks()
        orders = self.assemble(parts)
        cost, switched = self._orders_cost(self.st, orders, was_done, self.ctrl, self.weights, self.orders_stats,
                                           self.order_battle)
        self.advance(self.st, orders, self.params, self.params.dt)
        # each unit's worst loss so far: a rally gives nothing back, a new rout counts beyond it (reward.track)
        self.st.u["lost_worst"] = self._track(self.st.u, self.weights.rout_share)
        after, finished, terms, r, self.hit_rate, self.last_hit = self._rewards(
            self.st, self.health, was_done, self.hit_rate, self.last_hit, cost, self.weights, self.params.dt, rb, rs,
            attacks, self.part_stats, self.part_steps, self.timeout_count)
        r_rows = r[rb, rs] * (~was_done[rb]).float()
        d_rows = finished[rb]
        self._count(finished, rb, rs, d_rows)
        lord_dead = after[:, :, 2] < 0.5 if after.shape[-1] > 2 else torch.zeros_like(after[:, :, 0], dtype=torch.bool)
        d = d_rows.float()
        self.lord_stats += torch.stack([d.sum(), (d * lord_dead[rb, rs].float()).sum(),
                                        (d * lord_dead[rb, 1 - rs].float()).sum()])
        self.ability_stats += torch.stack([((self._ability_marks() > marks + 1e-3).float().sum((1, 2))
                                            * (~was_done).float()).sum(), d.sum()])
        sw = self._learner_units(switched).float().sum(1)
        self.switch_stats += sw.sum()
        self.order_battle[:, 1] += sw
        self.health = after                          # = reward.measure of the state now
        return r_rows, finished

    def reward_parts(self, reset=True):
        """{role: {term: the learner's mean reward a minute of battle}} since the last call (reward.PARTS;
        the sum of the terms is the reward PPO learns from)."""
        per_min = 60.0 / self.params.dt
        p, n = self.part_stats.cpu().numpy(), self.part_steps.cpu().numpy()
        out = {role: {k: float(p[i, j] / max(1.0, n[i]) * per_min) for j, k in enumerate(reward.PARTS)}
               for i, role in enumerate(ROLES) if n[i] > 0}
        if reset:
            self.part_stats.zero_()
            self.part_steps.zero_()
        return out

    def _count(self, finished, rb, rs, d_rows):
        won = (self.st.winner[rb] == rs + 1).float()
        code = self.row_opp if not self.past_untrained else torch.where(
            self.row_opp == league.CODE["past"], torch.full_like(self.row_opp, UNTRAINED), self.row_opp)
        d = d_rows.float()
        defends = (self.st.attacker[rb] != rs + 1).long()
        self.stats.index_add_(0, 2 * code + defends, torch.stack([d, d * won, d * self.st.t[rb]], 1))
        self.battle_count += finished.sum()

    # Battles and timeouts so far: counted on the GPU (reading them back every decision waited for it).
    @property
    def battles(self):
        return int(self.battle_count)

    @battles.setter
    def battles(self, n):
        self.battle_count = torch.full((), n, dtype=torch.long, device=self.device)

    @property
    def timeouts(self):
        return int(self.timeout_count)

    @timeouts.setter
    def timeouts(self, n):
        self.timeout_count = torch.full((), n, dtype=torch.long, device=self.device)

    def _learner_units(self, mask):
        """mask [B, N] limited to the learner's units."""
        return mask & learner_units(self.st.u, self.ctrl)

    def _ability_marks(self):
        """[B, N, SLOTS] the learner's units' ability cooldowns (a use sets one up)."""
        u = self.st.u
        cd = torch.stack([u[f"ab{k}_cd"] for k in range(sim_abilities.SLOTS)], -1)
        return cd * self._learner_units(torch.ones_like(u["m"])).float()[..., None]

    def abilities(self, reset=True):
        """The learner's ability uses per its ended battle since the last call (uses / battles)."""
        uses, n = (float(x) for x in self.ability_stats)
        if reset:
            self.ability_stats.zero_()
        return uses / max(1.0, n)

    def lords(self, reset=True):
        """{"own", "enemy"}: shares of the learner's ended battles in which its own / the enemy's lord
        was dead at the end, since the last call; "switches_per_minute": attack target switches per
        standing learner unit per minute (read before orders_per_minute resets its unit-steps)."""
        n, own, enemy = (float(x) for x in self.lord_stats)
        steps = float(self.orders_stats[1])
        out = {"own": own / max(1.0, n), "enemy": enemy / max(1.0, n),
               "switches_per_minute": float(self.switch_stats[0]) / max(1e-9, steps * self.params.dt / 60)}
        if reset:
            self.lord_stats.zero_()
            self.switch_stats.zero_()
        return out

    def orders_per_minute(self, reset=True):
        """Real order changes per standing learner unit per minute of battle since the last call."""
        c, n = (float(x) for x in self.orders_stats)
        if reset:
            self.orders_stats.zero_()
        return c / max(1e-9, n * self.params.dt / 60)

    def kinds(self, reset=True):
        """{kind: share of the learner's unit decisions} since the last call."""
        k = self.kind_stats.cpu().clone().numpy()
        if reset:
            self.kind_stats.zero_()
        return {name: float(k[i] / max(1, k.sum())) for i, name in enumerate(O.KINDS)}

    def _reset(self, finished):
        if not bool(finished.any()):
            return
        idx = restart_rows(self.st, self.setup, self.source, finished, self.want)
        self.last_hit = torch.where(finished, torch.full_like(self.last_hit, -1.0), self.last_hit)
        self.hit_rate = torch.where(finished, torch.zeros_like(self.hit_rate), self.hit_rate)
        sim_abilities.set_rule(self.st.u, self.by_rule)
        self.bank_row = torch.where(finished, idx, self.bank_row)
        randomise.apply(self.st, finished, self.spread, self.gen)
        if self.teach_names:                     # a new battle: a new draw for the teacher's share
            self.teach_draw = torch.where(finished, torch.rand(self.B, generator=self.teach_gen, device=self.device),
                                          self.teach_draw)
        state = self.st.observation()
        for s in (1, 2):
            self.mem[s] = merge_memory(self.mem[s], ob.start(state, self.setup, s), finished)
            self.cmem[s] = merge_memory(self.cmem[s], ob.start(state, self.setup, s), finished)

    def narrow(self, keep):
        """Only the battles keep (indices into the batch) stay, in ascending order (in place) -> keep
        sorted, as a tensor on the device: new battle i is old battle keep[i]. For evaluation
        (auto_reset off): a batch whose battles have mostly ended steps the live ones (and as many
        ended ones as fill a fixed size: each new size is compiled once, tools/nn/train/evaluate.py
        BUCKETS) instead of all. Everything per
        battle goes along: the state, the setup, both sides' memories, the networks' memories, the
        pending observation; the counters over all battles (stats, kind_stats, ...) stay."""
        B = self.B
        keep = torch.as_tensor(keep, device=self.device).long().sort().values
        two = torch.cat([keep, keep + B])                    # rows [2B]: (side - 1) * B + battle
        at = torch.full((B,), -1, dtype=torch.long, device=self.device)
        at[keep] = torch.arange(len(keep), device=self.device)
        sel_learn = (at[self.rows_learn % B] >= 0).nonzero().squeeze(1)       # same order: (side, battle)
        sel_past = (at[self.rows_past % B] >= 0).nonzero().squeeze(1)
        take = (lambda x: None if x is None else x[keep])
        st, setup = self.st, self.setup
        kl = keep.tolist()
        self.st = S.State({k: v[keep] for k, v in st.u.items()}, st.t[keep], st.attacker[keep], st.done[keep],
                          st.winner[keep], st.lord_dead_s[keep], st.bounds, [st.keys[i] for i in kl])
        arrays = ob._Arrays(**{k: take(getattr(setup.arrays, k)) for k in scenes.LiveSetup.FIELDS})
        self.setup = scenes.LiveSetup(arrays, {s: setup.char[s][keep] for s in (1, 2)},
                                      [setup.factions[i] for i in kl])
        self.bounds2 = torch.cat([self.setup.bounds, self.setup.bounds])

        def memory(m):
            return ob.Memory(Frame(*(getattr(m.frame, k)[keep] for k in FRAME)), *(take(getattr(m, k)) for k in MEMORY))
        self.mem = {s: memory(m) for s, m in self.mem.items()}
        self.cmem = {s: memory(m) for s, m in self.cmem.items()}
        if self.h_learn is not None:
            self.h_learn = self.h_learn[sel_learn]
        if self.h_past is not None:
            self.h_past = self.h_past[sel_past]
        if self.cur is not None:
            a, frame, c = self.cur
            self.cur = ({k: v[two] for k, v in a.items()}, frame_rows(frame, two),
                        None if c is None else {k: v[sel_learn] for k, v in c.items()})
        for k in ("health", "last_hit", "hit_rate", "bank_row", "want", "ctrl", "by_rule", "kind_battle",
                  "order_battle", "teach_draw"):
            setattr(self, k, getattr(self, k)[keep])
        self.row_opp = self.row_opp[sel_learn]
        self.row_normal = self.row_normal[sel_learn]
        lay = self.layout
        self.layout = league.Layout(lay.scene[kl], lay.learner[kl], lay.opponent[kl])
        self.B = len(kl)
        flat = torch.cat([self.ctrl[:, 0], self.ctrl[:, 1]])
        self.rows_learn = (flat == league.LEARNER).nonzero().squeeze(1)
        self.rows_past = (flat == league.CODE["past"]).nonzero().squeeze(1)
        self.place_learn = placement(self.rows_learn, self.B)
        self.place_past = placement(self.rows_past, self.B)
        # self.scripts stays: the compiled assembly is keyed by it (an unused script costs little)
        return keep

    def take_stats(self):
        """{"opponent/role": (games, wins, mean seconds)} since the last call (role of the learner)."""
        s = self.stats.cpu().clone().numpy()
        self.stats.zero_()
        out = {}
        for i, name in enumerate(STAT_NAMES):
            for j, role in enumerate(ROLES):
                g, w, sec = s[2 * (i + 1) + j]
                if g > 0:
                    out[f"{name}/{role}"] = (int(g), int(w), float(sec / g))
        return out

    @torch.no_grad()
    def value(self, critic):
        """The side's value [R] of the learner rows now."""
        if self.cur is None:
            self.cur = self.observe(True)
        return self._values(critic, self.cur[2])


def collect(env, actor, critic, T):
    """T decisions (steps) of every battle -> a dict of stacked tensors [T, R, ...], the memory the chunk began
    with h0 [R, 1 + N, d], reset [T, R] (a new battle began at step t: its memory starts empty), the
    bootstrap value [R], and abil_static: the bank's ability passports (full_obs puts them back into obs)."""
    h0 = env.h_learn
    steps = [env.step(actor, critic) for _ in range(T)]
    out = {}
    for k in ("obs", "critic_obs"):
        out[k] = {n: torch.stack([s[k][n] for s in steps]) for n in steps[0][k]}
    out["h0"] = h0 if h0 is not None else actor.initial(rows_of(steps[0]["obs"], slice(None)))
    done = torch.stack([s["done"] for s in steps])
    out["reset"] = torch.cat([torch.zeros_like(done[:1]), done[:-1]])
    fields = ("kind", "point", "target", "run") + (("ability",) if steps[0]["action"].ability is not None else ())
    out["action"] = hd.Action(*(torch.stack([getattr(s["action"], f) for s in steps]) for f in fields))
    out["abil_static"] = getattr(env.bank.setup.arrays, "abil", None)
    for k in ("lp", "value", "reward", "done", "attacks"):
        out[k] = torch.stack([s[k] for s in steps])
    if steps[0].get("teach") is not None:
        # the teachers: labels [T, R, N], where they count [T, R, N], the unit's teacher [T, R, N] (index into
        # names), the unit's battle labelled [T, R, N] (the share), the shares
        out["teach"] = {"action": hd.Action(*(torch.stack([getattr(s["teach"][0], f) for s in steps])
                                              for f in ("kind", "point", "target", "run"))),
                        "valid": torch.stack([s["teach"][1] for s in steps]),
                        "drill": torch.stack([s["teach"][2] for s in steps]),
                        "picked": torch.stack([s["teach"][3] for s in steps]), "names": env.teach_names,
                        "shares": env.teach_shares()}
    out["last_value"] = env.value(critic)
    return out

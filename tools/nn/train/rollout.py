"""Many simulated battles at once, played by the learner and its opponents (docs/en/training/training.md).

Battles(layout) holds B battles on one device. A row is one side of one battle: row = (side - 1) * B
+ battle, so both sides' observations stack into one batch of 2B rows. Each row has a controller
(tools/nn/train/league.py): the learner, a past version, or a script. Every step (one decision,
0.5 s of battle - the simulator's step):

    observe both sides -> the learner acts for its rows (and the past version for its rows,
    the scripts for theirs) -> one simulator step -> reward -> finished battles start again
    (auto_reset) with fresh randomised numbers.

step() returns the learner's transition: its observation, action, log-probabilities, the critic's
value, reward and done, per learner row. collect() stacks T of them with the memory the chunk
began with, so the update can run the memory (GRU) through the chunk again.
"""
import numpy as np
import torch

from tools.nn.model import heads as hd
from tools.nn.model import observation as ob
from tools.nn.model import policy
from tools.nn.model.decide import to_orders
from tools.nn.model.frame import Frame
from tools.nn.sim import battle
from tools.nn.sim import orders as O
from tools.nn.sim import state as S
from tools.nn.sim.params import load
from tools.nn.train import league, opponents, randomise, reward, scenes

CRITIC_KEYS = ("tokens", "ctx", "own", "attend", "pos")
FRAME = ("cx", "cz", "ux", "uz")
MEMORY = ("last_x", "last_z", "last_t", "seen", "dead", "prev_x", "prev_z", "prev_t", "prev_vis", "last_melee_t",
          "last_rout_t", "lord_dead_t")
ROLES = ("attack", "defend")
# Outcome counters: the opponents, and the untrained network as "past" on its own.
STAT_NAMES = league.OPPONENTS + ("untrained",)
UNTRAINED = len(league.OPPONENTS) + 1


_OBSERVE = {}


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


def open_rows(source, want=None):
    """(State, LiveSetup, bank rows) of a batch that starts with the battles the source picks."""
    idx = source.pick(want)
    b, bs = source.bank.state, source.bank.setup
    st = S.State({k: v[idx].clone() for k, v in b.u.items()}, b.t[idx].clone(), b.attacker[idx].clone(),
                 b.done[idx].clone(), b.winner[idx].clone(), b.lord_dead_s[idx].clone(), b.bounds,
                 [list(b.keys[int(i)]) for i in idx.tolist()])
    arrays = ob._Arrays(**{k: getattr(bs.arrays, k)[idx].clone() for k in scenes.LiveSetup.FIELDS})
    setup = scenes.LiveSetup(arrays, {s: bs.char[s][idx].clone() for s in (1, 2)},
                             {s: bs.adapt[s][idx].clone() for s in (1, 2)}, [bs.factions[int(i)] for i in idx.tolist()])
    return st, setup, idx


def restart_rows(st, setup, source, rows, want=None):
    """Battles where rows [B] is true start again as battles the source picks (in place). -> bank rows."""
    idx = source.pick(want)
    scenes.take_rows(st, source.bank.state, rows, idx)
    setup.take(source.bank.setup, rows, idx)
    return idx


class Battles:
    def __init__(self, layout, scene_list=scenes.SCENES, device="cpu", params=None, spread=randomise.Spread(),
                 weights=reward.Weights(), seed=0, auto_reset=True, compile=None, source=None):
        self.device = torch.device(device)
        self.params = params or load()
        self.layout = layout
        self.spread = spread
        self.weights = weights
        self.auto_reset = auto_reset
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        # Where battles come from: the fixed scenes (battle b plays layout.scene[b]) or generated armies.
        self.source = source or scenes.Fixed(layout.scene, scene_list, self.params, self.device)
        self.bank = self.source.bank
        self.B, self.N = layout.B, self.bank.N
        self.advance = battle.stepper(self.device, compile)
        self.look = observer(self.device, compile)
        # `hold` is met only as the defender: its battles must have the learner attacking.
        only = np.isin(layout.opponent, [league.CODE[n] for n in league.ATTACK_ONLY])
        self.want = torch.as_tensor(np.where(only, layout.learner, 0), device=self.device)
        self.ctrl = torch.as_tensor(layout.controllers(), device=self.device)           # [B, 2]
        flat = torch.cat([self.ctrl[:, 0], self.ctrl[:, 1]])
        self.rows_learn = (flat == league.LEARNER).nonzero().squeeze(1)
        self.rows_past = (flat == league.CODE["past"]).nonzero().squeeze(1)
        self.scripts = {league.CODE[n]: f for n, f in opponents.SCRIPTS.items() if bool((self.ctrl == league.CODE[n]).any())}
        opp = torch.as_tensor(layout.opponent, device=self.device)
        self.row_opp = torch.cat([opp, opp])[self.rows_learn]                          # [R]
        self.past_actor = None
        self.past_untrained = False
        self.stats = torch.zeros(2 * (len(STAT_NAMES) + 1), 3, device=self.device)    # games, wins, seconds
        # by (opponent, the learner's role)
        self.row_attacks = None
        self.orders_stats = torch.zeros(2, device=self.device)                         # changes, unit-steps
        self.kind_stats = torch.zeros(len(O.KINDS), dtype=torch.long, device=self.device)   # decisions by kind
        self.lord_stats = torch.zeros(3, device=self.device)        # learner battles ended, own / enemy lord dead
        self.switch_stats = torch.zeros(1, device=self.device)      # the learner's attack target switches
        self.start()

    @property
    def R(self):
        return len(self.rows_learn)

    def start(self):
        self.st, self.setup, idx = open_rows(self.source, self.want)
        self.bank_row = idx.clone()
        self.bounds2 = torch.cat([self.setup.bounds, self.setup.bounds])
        every = torch.ones(self.B, dtype=torch.bool, device=self.device)
        randomise.apply(self.st, every, self.spread, self.gen)
        state = self.st.observation()
        self.mem = {s: ob.start(state, self.setup, s) for s in (1, 2)}
        self.cmem = {s: ob.start(state, self.setup, s) for s in (1, 2)}
        self.h_learn = None
        self.h_past = None
        self.health = reward.measure(self.st)
        self.cur = None
        self.battles = 0
        self.timeouts = 0

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
        logits, h_new = actor(obs_r, h)
        action = hd.sample(logits, greedy)
        orders = to_orders(actor.cfg, action, obs_r, frame_rows(frame, rows), self.bounds2[rows])
        return obs_r, h, logits, action, orders, h_new

    def assemble(self, parts):
        """[(rows, Orders [len(rows), N])] and the scripts -> the batch's Orders [B, N]."""
        B, N = self.B, self.N
        side = {s: O.hold(B, N, self.device) for s in (1, 2)}
        for code, script in self.scripts.items():
            o = script(self.st)
            for s in (1, 2):
                use = (self.ctrl[:, s - 1] == code)[:, None].expand(B, N)
                side[s] = O.merge(side[s], o, use)
        for rows, o in parts:
            b, two = rows % B, rows >= B
            for s, sel in ((1, ~two), (2, two)):
                for k in O.FIELDS:
                    getattr(side[s], k)[b[sel]] = getattr(o, k)[sel]
        return O.merge(side[1], side[2], self.st.u["side"] == 2)

    # --- one decision and one simulator step ---
    @torch.no_grad()
    def step(self, actor, critic=None, greedy=False):
        if self.cur is None:
            self.cur = self.observe(critic is not None)
        a, frame, c = self.cur
        obs_r, h_prev, logits, action, orders, h_new = self._act(actor, a, frame, self.rows_learn, self.h_learn, greedy)
        parts = [(self.rows_learn, orders)]
        h_past_new = None
        if len(self.rows_past) and self.past_actor is not None:
            *_, o_past, h_past_new = self._act(self.past_actor, a, frame, self.rows_past, self.h_past, False)
            parts.append((self.rows_past, o_past))
        lp, _ = hd.log_prob(logits, action, obs_r["ctrl"])
        value = critic(c) if critic is not None else None
        acting = obs_r["ctrl"] & ~self.st.done[self.rows_learn % self.B][:, None]
        self.kind_stats += torch.bincount(action.kind[acting], minlength=len(O.KINDS))[:len(O.KINDS)]

        was_done = self.st.done.clone()
        orders = self.assemble(parts)
        changes = reward.order_changes(self.st.u, orders, self.weights.order_move_m) & ~was_done[:, None]
        switched = reward.retargets(self.st.u, orders) & ~was_done[:, None]
        cost = reward.order_cost(changes, self.st.u["side"], self.weights, switched)
        self._count_orders(changes, was_done)
        self.advance(self.st, orders, self.params, self.params.dt)
        after = reward.measure(self.st)
        finished = self.st.done & ~was_done
        r = reward.step(self.health, after, finished, self.st.winner, self.st.attacker, self.weights) - cost
        r = r - reward.idle_cost(self.st, self.weights) * (~finished).float()[:, None]
        self.timeouts += int((finished & (after[:, 0, 1] > 0) & (after[:, 1, 1] > 0)).sum())
        rb, rs = self.rows_learn % self.B, self.rows_learn // self.B
        r_rows, d_rows = r[rb, rs], finished[rb]
        self._count(finished, rb, rs, d_rows)
        lord_dead = after[:, :, 2] < 0.5 if after.shape[-1] > 2 else torch.zeros_like(after[:, :, 0], dtype=torch.bool)
        d = d_rows.float()
        self.lord_stats += torch.stack([d.sum(), (d * lord_dead[rb, rs].float()).sum(),
                                        (d * lord_dead[rb, 1 - rs].float()).sum()])
        self.switch_stats += self._learner_units(switched).float().sum()

        if self.auto_reset:
            self._reset(finished)
        self.health = reward.measure(self.st)
        keep = (~finished).float()
        self.h_learn = h_new * keep[rb][:, None, None]
        if h_past_new is not None:
            self.h_past = h_past_new * keep[self.rows_past % self.B][:, None, None]
        self.cur = self.observe(critic is not None)
        return {"obs": obs_r, "action": action, "lp": lp, "value": value, "reward": r_rows,
                "done": d_rows, "critic_obs": c}

    def _count(self, finished, rb, rs, d_rows):
        won = (self.st.winner[rb] == rs + 1).float()
        code = torch.where((self.row_opp == league.CODE["past"]) & torch.tensor(self.past_untrained, device=self.device),
                           torch.full_like(self.row_opp, UNTRAINED), self.row_opp)
        d = d_rows.float()
        defends = (self.st.attacker[rb] != rs + 1).long()
        self.stats.index_add_(0, 2 * code + defends, torch.stack([d, d * won, d * self.st.t[rb]], 1))
        self.battles += int(finished.sum())

    def _learner_units(self, mask):
        """mask [B, N] limited to the learner's units."""
        side = self.st.u["side"]
        learner = torch.zeros_like(mask)
        for s in (1, 2):
            learner = learner | ((side == s) & (self.ctrl[:, s - 1] == league.LEARNER)[:, None])
        return mask & learner

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

    def _count_orders(self, changes, was_done):
        """Order changes of the learner's units that stand, and their unit-steps."""
        u = self.st.u
        side = u["side"]
        learner = torch.zeros_like(changes)
        for s in (1, 2):
            learner = learner | ((side == s) & (self.ctrl[:, s - 1] == league.LEARNER)[:, None])
        standing = (side > 0) & (u["men"] > 0) & ~u["gone"] & ~u["r"] & learner & ~was_done[:, None]
        self.orders_stats += torch.stack([(changes & learner).float().sum(), standing.float().sum()])

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
        self.bank_row = torch.where(finished, idx, self.bank_row)
        randomise.apply(self.st, finished, self.spread, self.gen)
        state = self.st.observation()
        for s in (1, 2):
            self.mem[s] = merge_memory(self.mem[s], ob.start(state, self.setup, s), finished)
            self.cmem[s] = merge_memory(self.cmem[s], ob.start(state, self.setup, s), finished)

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
        if self.cur is None:
            self.cur = self.observe(True)
        return critic(self.cur[2])


def collect(env, actor, critic, T):
    """T steps of every battle -> a dict of stacked tensors [T, R, ...], the memory the chunk began
    with h0 [R, 1 + N, d], reset [T, R] (a new battle began at step t: its memory starts empty) and
    the bootstrap value [R]."""
    h0 = env.h_learn
    steps = [env.step(actor, critic) for _ in range(T)]
    out = {}
    for k in ("obs", "critic_obs"):
        out[k] = {n: torch.stack([s[k][n] for s in steps]) for n in steps[0][k]}
    out["h0"] = h0 if h0 is not None else actor.initial(rows_of(steps[0]["obs"], slice(None)))
    done = torch.stack([s["done"] for s in steps])
    out["reset"] = torch.cat([torch.zeros_like(done[:1]), done[:-1]])
    out["action"] = hd.Action(*(torch.stack([getattr(s["action"], f) for s in steps])
                                for f in ("kind", "point", "target", "run")))
    for k in ("lp", "value", "reward", "done"):
        out[k] = torch.stack([s[k] for s in steps])
    out["last_value"] = env.value(critic)
    return out

"""The general in the battles (rollout.Battles): the plan of every side a network with a general plays, kept per row
(row = (side - 1) * B + battle, as rollout's), the executor's task input every decision, and the plans' records for
training (docs/en/training/general.md "How it runs").

* Deployment (a battle's start, Battles._reset / the first step): the general of the side's network plans (groups,
  missions, places in the zone) on the armies and the tiles of the scripted placement, the layout rule puts the units
  on their groups' tiles (deploy.py). Scripted sides keep place.py's placement; so does a side whose units do not stand
  in their zone (a kite battle, a drill) or the config's deploy "script" - the general still plans its groups and
  missions.
* Re-planning ("Пересмотр замысла"): every replan_s of battle, and on an event - one of our groups broken (more than
  `broken` of its units at the plan routs or is destroyed), a lord fallen (ours or theirs) - but not sooner than
  replan_min_s after the last plan. Between plans the groups, missions and targets are locked. A re-plan runs the whole
  network again on the armies now (men and health left), the side's observation's tiles and the places' cards of the
  last plan (inertia); the routing take no part in the groups. Every unit's place (net.POS_FEATURES: where it stands,
  its facing and velocity in the side's frame) comes from the side's observation's tokens (at the deployment, from the
  state: everything seen, standing). The new groups keep the last plan's numbers by their units (net.match_groups): a
  number is a group while it keeps its units - its prev card, its task, the logs' gN.
* The task (inputs): per own unit with a group - its group card (the mix of our places' cards by its shares), its
  group's mission kind card, and its group's target from the unit now: direction from its facing (cos, sin), distance
  / dist_m (capped), "none" (zeros and the flag). A target group's point is the centre of its units now (as the side
  sees them); a coarse tile's, its centre (a fixed point of the plan). Computed every decision (the targets move, the
  unit turns): a function of the state, no noise.
* Records (training): every plan the learner's general makes keeps its inputs, choices, log-probabilities and value;
  when it ends (the next plan or the battle's end) it is "ready" with its return - the reward summed from its decision
  to its end (run.py --general-return "span"), or that plus the next plan's discounted value ("td": the value is then of
  the rest of the battle), or ("critic") the reward discounted per decision + gamma^n x the unit network's critic's
  value of the side where the plan ended (0 at the battle's end), with "base" = the critic's value where it began (at a
  battle's first decision for the deployment's plan): the advantage is target - base (signal.plan_advantage;
  note_value gives the runtime the critic's value of the decision before its re-plans). The plans in force during a
  chunk of decisions form its table (rollout.collect): the update computes their cards again with gradients, so the
  executor's loss reaches the general (learn.py).
* Behaviour (run.py --general-distinct W: `distinct` is a distinct.Learner): every decision the learner's groups' order
  kinds, run, centre and path are summed (behave); a plan's end turns them into its groups' behaviour (behaviour: BEH
  numbers a group) and, at the bonus's weight, the classifier's bonus is added to the row's reward of that decision
  (take_bonus: after the reward went into the plans - the general's return has no bonus).
* Carrying out the missions (run.py --general-exec W: `measure` on, `exec_w` = W): every decision, after its simulator
  steps, each learner unit with a mission gets its measure 0..1 (execute; mission.py), summed by kind for the log; the
  executor's bonus of the decision, W x (the measure - its kind's running mean), goes out with the decision (the
  rollout's step "exec" [R, N]) into the units' own advantage (ppo.update, signal.exec_advantage) - never into the
  reward: the unit critic and the general's return do not see it.
* The teacher (run.py --general-teacher P: `teacher` = P): a share P of the battles (drawn at their start) has the
  learner's missions made by the rule (teacher.py; the groups stay the network's); such plans ("teach") play and train
  the executor but leave no record for the general's PPO, the classifier or the branches.
* Mission branches (run.py --general-branches P: `brancher` is a branch.Brancher): every plan of the learner marks its
  rows "fresh" (every kind's log-probability is kept with the plan: kind_logp); the next decision's capture snapshots
  a share of them (branch.py) and clears the marks.
* The game (tools/nn/companion/general.py GameBattles): the companion runs this Runtime on a view of the bridge's state
  shaped like a Battles of one battle - what it reads of `env`: st.u (x, z, b, side, men, hp, hp0, ms, gone, vis, cost,
  width, sp_h, sp_v), st.t, st.done, st.lord_dead_s, st.keys, setup.bounds, mem[s].frame, ctrl, past, battle_slot,
  B, N, device (rows_learn: the training's hooks only). A change of what it reads changes GameBattles too
  (test_companion_general_torch.py: the game's plans and tasks are the simulator's).
"""
import math
import numpy as np
import torch

from tools.nn.general import deploy as gdeploy
from tools.nn.general import mission as gmission
from tools.nn.general import signal as gsignal
from tools.nn.general import teacher as gteacher
from tools.nn.general import tiles as gtiles
from tools.nn.general import units as gunits
from tools.nn.general.config import mission_kinds
from tools.nn.general.executor import has_task
from tools.nn.general.net import CHOICES, INPUTS, POS_FEATURES, target_kind
from tools.nn.model import observation as ob
from tools.nn.model import sectors as msectors
from tools.nn.model.frame import Frame
from tools.nn.sim import orders as O

LEARNER, PAST = 0, 2                    # league.LEARNER, league.CODE["past"] (league imports drills: kept light here)
# the log's counters (Counts): summed on the device, read once an update (take_stats)
COUNTS = ("plans", "plans_deploy", "plans_replan", "groups_own", "groups_enemy", "deploys", "deploys_net", "laid_units",
          "replan_timer", "replan_broken", "replan_lord", "battles", "replans_ended", "distinct_bonus",
          "distinct_groups", "churn_units", "churn_moved", "kind_kept_groups", "kind_switched", "target_own",
          "target_enemy", "target_tile", "exec_bonus", "exec_bonus_abs", "exec_units", "pairs_ok", "pairs_all",
          "teach_plans", "battle_min")
# a group's behaviour during a plan (Runtime.behaviour): its units' order kinds' shares, the run share, its centre's
# displacement forward and sideways from the side's facing at the plan's start (/ 100 m, within +-5), its centre's path
# (/ 100 m, at most 10), the plan's length (decisions / 60)
BEH = tuple("kind_" + k for k in O.KINDS) + ("run", "forward", "sideways", "path", "length")
PLAN = ("main", "perm", "kind", "target", "place", "lp", "value", "active")    # a plan's record besides its inputs
REC = INPUTS + PLAN
_TABLE = None


def unit_table():
    global _TABLE
    if _TABLE is None:
        _TABLE = gunits.UnitTable()
    return _TABLE


def _cat(parts):
    if not parts:
        return None
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}


def _take(d, idx):
    return {k: v[idx] for k, v in d.items()}


class Counts:
    """The runtime's counters on the device (no read-back a decision: a float() each made the CPU wait for the GPU);
    counts["name"] reads one (tests), take() all of them and zeroes them."""

    def __init__(self, dev, kinds, tiles):
        self.v = torch.zeros(len(COUNTS), dtype=torch.float64, device=dev)
        self.kinds = torch.zeros(kinds, dtype=torch.float64, device=dev)
        self.places = torch.zeros(tiles, dtype=torch.float64, device=dev)
        self.exec = torch.zeros(2, kinds, dtype=torch.float64, device=dev)     # the measures' sum and units a kind

    def add(self, name, x):
        self.v[COUNTS.index(name)] += x

    def __getitem__(self, name):
        return float(self.v[COUNTS.index(name)])

    def take(self):
        # (copies: on the CPU .cpu().numpy() is the tensor's own memory, zeroed below)
        out = (dict(zip(COUNTS, self.v.tolist())), self.kinds.cpu().numpy().copy(), self.places.cpu().numpy().copy(),
               self.exec.cpu().numpy().copy())
        for t in (self.v, self.kinds, self.places, self.exec):
            t.zero_()
        return out


class Runtime:
    """The general's state of a Battles (module doc). env: the Battles; mcfg: the unit network's ModelConfig (its
    sectors); gcfg: the GeneralConfig (the learner's)."""

    def __init__(self, env, mcfg, gcfg):
        self.mcfg, self.gcfg = mcfg, gcfg
        self.zone = gtiles.ZoneRule.of_arena()
        self.dev = env.device
        self.B, self.N = env.B, env.N
        self.G, self.S, self.D = gcfg.places, mcfg.sectors ** 2, gcfg.card
        self.gamma = 0.9994                       # per decision (run.py sets the trainer's)
        self.ret = "span"                         # the return of a plan: "span", "td" or "critic" (--general-return)
        self.training = False                     # records kept (rollout.collect turns it on)
        self.frozen = False                       # a branch's copy: no plans of its own (branch)
        self.greedy = False
        self.learner = None                       # the learner's actor (Battles.step sets it)
        self.r = self._rows(2 * self.B)
        self.b = self._battles(self.B)
        self.pending = torch.ones(self.B, dtype=torch.bool, device=self.dev)   # the first start's deployment
        self.table, self.tix = None, torch.full((2 * self.B,), -1, dtype=torch.long, device=self.dev)
        self.ready = []
        self.stats = Counts(self.dev, gcfg.kinds, self.S)
        self.v_now = None                         # the critic's value [2B] of the decision now (note_value), NaN: none
        self.distinct = None                      # --general-distinct: the classifier (distinct.Learner), behaviour on
        self.bonus = torch.zeros(2 * self.B, device=self.dev)    # the distinct bonus due to the rows' reward
        self.brancher = None                      # --general-branches: the mission branches (branch.Brancher)
        # --general-exec: the missions' measures (mission.py) every decision, and the executor's bonus at weight exec_w
        self.measure = False
        self.exec_w = 0.0
        self.exec_rate = 0.01                     # the kinds' running mean of the measure: this share of a decision's
        self.exec_base = torch.full((gcfg.kinds,), float("nan"), device=self.dev)   # (NaN: none yet)
        # --general-teacher P: the share of the learner's battles whose missions the teacher gives (teacher.py)
        self.teacher = 0.0
        self.teach_gen = None

    # --- storage ------------------------------------------------------------------------------------------------
    def _rows(self, R):
        G, N, S, D, dev = self.G, self.N, self.S, self.D, self.dev
        z = (lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=dev))
        L, Bo = torch.long, torch.bool
        r = {"has": z(R, dt=Bo), "owner": z(R, dt=L) - 1, "t_plan": z(R), "n_dec": z(R, dt=L),
             "main": z(R, N, dt=L) - 1, "perm": z(R, 2 * G, dt=L) + torch.arange(2 * G, device=dev),
             "kind": z(R, G, dt=L), "target": z(R, G, dt=L),
             "place": z(R, G, dt=L) - 1, "tgt": z(R, G, 2), "mix": z(R, N, D), "kc": z(R, N, D),
             "places": z(R, 2 * G, D), "members": z(R, G), "lords": z(R, 2, dt=Bo), "r_sum": z(R),
             "n_replan": z(R, dt=L), "lp": z(R, G), "value": z(R), "active": z(R, 2 * G, dt=Bo),
             "r_disc": z(R), "v_start": z(R) + float("nan"),
             "b_kind": z(R, G, len(O.KINDS)), "b_run": z(R, G), "b_n": z(R, G), "b_c0": z(R, G, 2),
             "b_ok0": z(R, G, dt=Bo), "b_last": z(R, G, 2), "b_okl": z(R, G, dt=Bo), "b_path": z(R, G),
             "b_fwd": z(R, 2), "b_fwd_ok": z(R, dt=Bo),
             "kind_logp": z(R, G, self.gcfg.kinds), "fresh": z(R, dt=Bo), "teach": z(R, dt=Bo),
             "new": z(R, dt=Bo)}
        from tools.nn.general import army_input as ai
        r.update({"in/num": z(R, N, ai.SIZE), "in/lab": z(R, N, len(ai.LABELS), dt=L),
                  "in/pos": z(R, N, len(POS_FEATURES)), "in/prev_main": z(R, N, dt=L) - 1, "in/on": z(R, N, dt=Bo),
                  "in/own": z(R, N, dt=Bo), "in/free": z(R, N, dt=Bo), "in/tiles": z(R, S, len(gtiles.FEATURES)),
                  "in/tile_ok": z(R, S, dt=Bo), "in/zone": z(R, S, dt=Bo), "in/deploy": z(R, dt=Bo),
                  "in/prev": z(R, 2 * G, D), "in/has_prev": z(R, dt=Bo)})
        return r

    def _battles(self, B):
        from tools.nn.general import army_input as ai
        z = (lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=self.dev))
        return {"num": z(B, self.N, ai.SIZE), "lab": z(B, self.N, len(ai.LABELS), dt=torch.long),
                "full": z(B, self.N, 2), "teach": z(B, dt=torch.bool)}

    def record(self, rows):
        """The plan records of rows [M] (net.INPUTS + PLAN: the plan's membership and numbering, choices,
        log-probabilities at the plan, value)."""
        out = {k[3:]: self.r[k][rows] for k in self.r if k.startswith("in/")}
        for k in PLAN:
            out[k] = self.r[k][rows]
        return out

    # --- who plans ------------------------------------------------------------------------------------------------
    def owners(self, env, rows):
        """[M] the planner of each row: 0 the learner, 1 + k the past slot k, -1 none (a script, or a network without a
        general)."""
        b, s = rows % self.B, rows // self.B
        ctrl = env.ctrl[b, s]
        out = torch.full_like(rows, -1)
        if self.learner is not None and has_task(self.learner):
            out = torch.where(ctrl == LEARNER, torch.zeros_like(out), out)
        for k, a in enumerate(env.past):
            if a is not None and has_task(a):
                out = torch.where((ctrl == PAST) & (env.battle_slot[b] == k), torch.full_like(out, 1 + k), out)
        return out

    def actor_of(self, env, owner):
        return self.learner if owner == 0 else env.past[owner - 1]

    # --- the plan's inputs ----------------------------------------------------------------------------------------
    def _frame(self, env, rows, deploy):
        b, s = rows % self.B, rows // self.B + 1
        u = {k: env.st.u[k][b] for k in ("x", "z", "side")}
        bounds = env.setup.bounds[b]
        if deploy:
            return gtiles.frame_at(u, s, bounds), bounds
        f1, f2 = env.mem[1].frame, env.mem[2].frame
        pick = (lambda a, c: torch.where(s == 1, torch.as_tensor(a, device=self.dev).float()[b],
                                         torch.as_tensor(c, device=self.dev).float()[b]))
        return Frame(pick(f1.cx, f2.cx), pick(f1.cz, f2.cz), pick(f1.ux, f2.ux), pick(f1.uz, f2.uz)), bounds

    def inputs(self, env, rows, deploy, obs=None):
        """(net input dict [M], frame of the rows, unit state fields [M, N]) of rows [M] now. deploy: the battle's
        start (tiles from the state, the zone); else a re-plan (tiles from the side's observation obs: the stacked
        [2B] dict)."""
        B = self.B
        b, s = rows % B, rows // B + 1
        u = {k: env.st.u[k][b] for k in ("x", "z", "b", "side", "men", "hp", "hp0", "ms", "gone", "vis", "cost",
                                          "width", "sp_h", "sp_v")}
        present = u["side"] > 0
        own = present & (u["side"] == s[:, None])
        alive = present & (u["men"] > 0) & ~u["gone"]
        rout = u["ms"].float() >= 5.5
        on = alive
        free = alive & ~rout
        num = gunits.dynamic(self.b["num"][b], self.b["full"][b], u, own, u["vis"] & present)
        frame, bounds = self._frame(env, rows, deploy)
        cells_in = msectors.inside(self.mcfg, msectors.geo(frame, bounds))
        if deploy:
            tiles = gtiles.from_state(self.mcfg, u, u["cost"], s, frame, cells_in)
            pos = self.pos_of_state(u, frame, present)
        else:
            o = {k: obs[k][rows] for k in ("tokens", "own", "attend", "pos")}
            tiles = gtiles.from_obs(self.mcfg, o, cells_in)
            pos = self.pos_of_obs(o)
        tile_ok = cells_in.any(-1)
        if deploy and self.gcfg.deploy == "net":
            valid = self.zone.valid(u, s)
            zone = self.zone.tiles(self.mcfg, frame, s, cells_in) & valid[:, None]
        else:
            zone = torch.zeros_like(tile_ok)
        has_prev = self.r["has"][rows] if not deploy else torch.zeros_like(self.r["has"][rows])
        prev = torch.where(has_prev.reshape(-1, 1, 1), self.r["places"][rows], torch.zeros_like(self.r["places"][rows]))
        prev_main = torch.where(has_prev[:, None], self.r["main"][rows], torch.full_like(self.r["main"][rows], -1))
        inp = {"num": num, "lab": self.b["lab"][b], "pos": pos, "on": on, "own": own, "free": free, "tiles": tiles,
               "tile_ok": tile_ok, "zone": zone, "deploy": zone.any(-1), "prev": prev, "has_prev": has_prev,
               "prev_main": prev_main}
        return inp, frame, u

    @staticmethod
    def pos_of_obs(o):
        """[M, N, 7] the units' places (net.POS_FEATURES) as the side's observation's tokens hold them: own units and
        the enemies seen (their last seen place); zeros for the others."""
        tok = o["tokens"].float()
        cols = torch.stack([tok[..., ob.INDEX[k]] for k in POS_FEATURES[:-1]], -1)
        known = o["attend"] & (o["own"] | (tok[..., ob.INDEX["seen"]] > 0.5))
        return torch.cat([cols, torch.ones_like(cols[..., :1])], -1) * known[..., None].float()

    @staticmethod
    def pos_of_state(u, frame, present):
        """[M, N, 7] the units' places at the deployment (the state: every unit seen, standing) in the side's frame
        (frame [M]), as the observation's token would show them."""
        f, l = frame.point(u["x"].float(), u["z"].float())
        fc, fs = frame.facing(u["b"].float())
        z = torch.zeros_like(f)
        p = torch.stack([f / ob.POS, l / ob.POS, fc, fs, z, z, torch.ones_like(f)], -1)
        return torch.nan_to_num(p) * present[..., None].float()

    # --- planning ---------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def plan(self, env, rows, deploy, obs=None, reason="deploy"):
        """Plan rows [M] (long, those with a planner) now. -> the deployment's {row: points} for the layout."""
        if rows.numel() == 0:
            return
        owner = self.owners(env, rows)
        keep = owner >= 0
        if not bool(keep.all()):
            rows, owner = rows[keep], owner[keep]
            if rows.numel() == 0:
                return
        inp, frame, u = self.inputs(env, rows, deploy, obs)
        outs = {}
        for o in torch.unique(owner).tolist():
            sel = (owner == o).nonzero().squeeze(1)
            gen = self.actor_of(env, o).general
            c, d = gen(_take(inp, sel), greedy=self.greedy)
            if o == 0 and self.teacher > 0:
                d = self._teach(env, rows[sel], c, d)
            mix, kc = gen.unit_cards(c, d["kind"], c["main"])
            outs[o] = (sel, c, d, mix, kc)
        M, G, N = rows.numel(), self.G, self.N
        res = {k: None for k in ("main", "perm", "kind", "target", "place", "lp", "value", "mix", "kc", "places",
                                 "active", "kind_logp", "pairs")}
        for o, (sel, c, d, mix, kc) in outs.items():
            vals = {"main": c["main"], "perm": c["perm"], "kind": d["kind"], "target": d["target"],
                    "place": d["place"], "lp": d["lp"], "value": d["value"], "mix": mix, "kc": kc,
                    "places": c["places"], "active": c["active"], "kind_logp": d["kind_logp"], "pairs": d["pairs"]}
            for k, v in vals.items():
                if res[k] is None:
                    res[k] = v.new_zeros((M,) + tuple(v.shape[1:]))
                res[k][sel] = v.to(res[k].dtype)
        # the previous plans end here: ready with their return (a re-plan; the battle's end: after())
        if not deploy:
            self._finish(rows, done=False, v_next=res["value"])
            self._churn(rows, owner, res)
        b, s = rows % self.B, rows // self.B + 1
        C = self.gcfg.target_grid ** 2
        tile_w = gtiles.coarse_world(self.mcfg, frame, self.gcfg.target_grid)  # [M, C, 2]
        ti = res["target"] - 2 * G
        is_tile = (ti >= 0) & (ti < C)
        w = tile_w.gather(1, ti.clamp(0, C - 1)[..., None].expand(-1, -1, 2))
        tgt = torch.where(is_tile[..., None], w, torch.full_like(w, float("nan")))
        members = (res["main"][..., None] == torch.arange(G, device=self.dev)).sum(1).float()
        lds = env.st.lord_dead_s[b]
        own_alive = lds.gather(1, (s - 1)[:, None])[:, 0] < 0
        foe_alive = lds.gather(1, (2 - s)[:, None])[:, 0] < 0
        r = self.r
        for k in ("main", "perm", "kind", "target", "place", "lp", "value", "mix", "kc", "places", "active",
                  "kind_logp"):
            r[k][rows] = res[k].to(r[k].dtype)
        for k, v in inp.items():
            r["in/" + k][rows] = v.to(r["in/" + k].dtype)
        r["tgt"][rows] = tgt
        r["members"][rows] = members
        r["lords"][rows] = torch.stack([own_alive, foe_alive], -1)
        r["t_plan"][rows] = env.st.t[b].float()
        r["n_dec"][rows] = 0
        r["new"][rows] = True                     # begun in this chunk (begin_chunk: the plans in force are old)
        r["r_sum"][rows] = 0.0
        r["r_disc"][rows] = 0.0
        # the critic's value at the plan (a deployment's: the battle's first decision fills it, note_value)
        nan = torch.full((M,), float("nan"), device=self.dev)
        r["v_start"][rows] = nan if deploy or self.v_now is None else self.v_now[rows]
        for k in ("b_kind", "b_run", "b_n", "b_c0", "b_last", "b_path", "b_fwd"):
            r[k][rows] = 0.0
        for k in ("b_ok0", "b_okl", "b_fwd_ok"):
            r[k][rows] = False
        r["has"][rows] = True
        r["owner"][rows] = owner
        r["teach"][rows] = ((owner == 0) & self.b["teach"][b]) if self.teacher > 0 else torch.zeros_like(owner,
                                                                                                         dtype=torch.bool)
        if self.brancher is not None:
            # the learner's new plans (not the teacher's): the mission branches' candidates
            r["fresh"][rows] = (owner == 0) & ~r["teach"][rows]
        if deploy:
            r["n_replan"][rows] = 0
        else:
            r["n_replan"][rows] += 1
        self._count(rows, owner, res, inp, deploy, reason)
        if self.table is not None:
            mine = (owner == 0).nonzero().squeeze(1)
            if mine.numel():
                base = sum(t["lp"].shape[0] for t in self.table)
                self.table.append(self.record(rows[mine]))
                self.tix[rows[mine]] = base + torch.arange(mine.numel(), device=self.dev)
        if deploy:
            self._layout(env, rows, res, inp, frame, u)

    def _teach(self, env, rows, c, d):
        """The general's choices d of the learner's rows [M] (cards c) with the teacher's missions where its battle is
        taught (--general-teacher; teacher.py): kind and target replaced on the groups the rule gives one."""
        b, s = rows % self.B, rows // self.B + 1
        u = env.st.u
        f = (lambda k: u[k][b])
        side = f("side")
        standing = (side > 0) & (f("men") > 0) & ~f("gone") & ~f("r")
        kind, target, use = gteacher.choose(c["main"], c["active"], f("x"), f("z"), side == s[:, None], standing,
                                            (f("range") > 0) & (f("a") > 0), f("lord"), self.G, self.gcfg.kinds)
        use = use & self.b["teach"][b][:, None]
        return dict(d, kind=torch.where(use, kind, d["kind"]), target=torch.where(use, target, d["target"]))

    def _count(self, rows, owner, res, inp, deploy, reason):
        """The log's counters of a planning call (the learner's plans only), on the device."""
        G, c = self.G, self.stats
        taught = self.r["teach"][rows]
        c.add("teach_plans", ((owner == 0) & taught).sum())
        mine = (owner == 0) & ~taught                  # the general's own plans (not the teacher's)
        act = res["active"]
        own_g = act[:, :G] & mine[:, None]
        pairs = res["pairs"] * own_g[..., None]
        c.add("pairs_ok", pairs[..., 0].sum())
        c.add("pairs_all", pairs[..., 1].sum())
        c.add("plans", mine.sum())
        c.add(f"plans_{reason}", mine.sum())
        c.add("groups_own", own_g.sum())
        c.add("groups_enemy", (act[:, G:] & mine[:, None]).sum())
        c.kinds.index_add_(0, res["kind"].reshape(-1), own_g.reshape(-1).double())
        tk = target_kind(res["target"], G, self.gcfg.target_grid ** 2)
        for i, name in enumerate(("target_own", "target_enemy", "target_tile")):
            c.add(name, ((tk == i) & own_g).sum())
        if deploy:
            dep = inp["deploy"] & mine
            c.add("deploys", mine.sum())
            c.add("deploys_net", dep.sum())
            pl = res["place"]
            m = own_g & dep[:, None] & (pl >= 0)
            c.places.index_add_(0, pl.clamp(min=0).reshape(-1), m.reshape(-1).double())

    def _churn(self, rows, owner, res):
        """The log's counters of a re-plan's numbering (the learner's plans; before the new plan is written): our units
        in a group in both plans and those whose group number changed (general_group_churn), our groups active in
        both plans and those whose mission kind changed (general_kind_switch)."""
        r, G, c = self.r, self.G, self.stats
        mine = (owner == 0) & r["has"][rows]
        old, new = r["main"][rows], res["main"]
        both = (old >= 0) & (old < G) & (new >= 0) & (new < G) & mine[:, None]
        c.add("churn_units", both.sum())
        c.add("churn_moved", (both & (old != new)).sum())
        kept = r["active"][rows][:, :G] & res["active"][:, :G] & mine[:, None]
        c.add("kind_kept_groups", kept.sum())
        c.add("kind_switched", (kept & (r["kind"][rows] != res["kind"])).sum())

    def _layout(self, env, rows, res, inp, frame, u):
        """The deployment of the rows whose plan places its groups (inp deploy): the layout rule, written into the
        state (x, z, b, ox, oz of the side's units)."""
        dep = inp["deploy"]
        if not bool(dep.any()):
            return
        sel = dep.nonzero().squeeze(1)
        G = self.G
        tile_w = gtiles.centres_world(self.mcfg, Frame(*(getattr(frame, k)[sel] for k in ("cx", "cz", "ux", "uz"))))
        host = {k: v[sel].cpu().numpy() for k, v in u.items() if k in ("x", "z", "side", "men", "width", "sp_h", "sp_v")}
        main = res["main"][sel].cpu().numpy()
        place = res["place"][sel].cpu().numpy()
        tw = tile_w.cpu().numpy()
        rr = rows[sel]
        b_np, s_np = (rr % self.B).cpu().numpy(), (rr // self.B + 1).cpu().numpy()
        # (both sides of a battle may deploy: the battles' units on the host, each row's units written in turn)
        bt = torch.as_tensor(np.unique(b_np), device=self.dev)
        at = {int(b): i for i, b in enumerate(bt.tolist())}
        fields = {k: env.st.u[k][bt].cpu().numpy().copy() for k in ("x", "z", "b", "ox", "oz")}
        laid_n = 0
        for i in range(len(sel)):
            side = int(s_np[i])
            own = host["side"][i] == side
            foe = (host["side"][i] > 0) & ~own
            enemy_c = (float(host["x"][i][foe].mean()), float(host["z"][i][foe].mean())) if foe.any() else (0.0, 0.0)
            units = []
            for n in np.nonzero(own)[0]:
                g = int(main[i, n])
                w = float(host["width"][i, n])
                units.append({"slot": int(n), "group": g if 0 <= g < G else -1, "width": w,
                              "depth": gdeploy.depth_of(float(host["men"][i, n]), w, float(host["sp_h"][i, n]),
                                                        float(host["sp_v"][i, n]))})
            groups = {g: tuple(tw[i, place[i, g]]) for g in range(G) if place[i, g] >= 0}
            pts = gdeploy.layout(units, groups, side, enemy_c, self.zone)
            x, z, bb, laid = gdeploy.as_arrays(pts, self.N)
            j = at[int(b_np[i])]
            for k, v in (("x", x), ("z", z), ("b", bb), ("ox", x), ("oz", z)):
                fields[k][j][laid] = v[laid]
            laid_n += int(laid.sum())
        for k, v in fields.items():
            t = env.st.u[k].clone()
            t[bt] = torch.as_tensor(v, device=self.dev, dtype=t.dtype)
            env.st.u[k] = t
        self.stats.add("laid_units", laid_n)

    # --- the battle's hooks ---------------------------------------------------------------------------------------
    def set_teacher(self, share):
        """--general-teacher P: the share of the battles taught; the battles running now are drawn at once (their next
        plans are the teacher's), the others at their start (start_battles)."""
        self.teacher = float(share)
        if self.teacher > 0:
            self._draw_teach(torch.arange(self.B, device=self.dev))

    def _draw_teach(self, idx):
        # (its own generator: P = 0 draws nothing, the training's other random numbers are as without it)
        if self.teach_gen is None:
            self.teach_gen = torch.Generator(device=self.dev).manual_seed(7121)
        self.b["teach"][idx] = torch.rand(len(idx), generator=self.teach_gen, device=self.dev) < self.teacher

    def start_battles(self, env, battles):
        """Battles [B] (bool) that (re)started: their units' slots, then the deployment's plans (both sides)."""
        if self.frozen or not bool(battles.any()):
            return
        idx = battles.nonzero().squeeze(1)
        keys = [env.st.keys[i] for i in idx.tolist()]
        num, lab, full = unit_table().battle(keys, self.dev)
        self.b["num"][idx], self.b["lab"][idx], self.b["full"][idx] = num, lab, full
        if self.teacher > 0:
            self._draw_teach(idx)                 # --general-teacher P: the battles whose learner plans it makes
        rows = torch.cat([idx, idx + self.B])
        self.r["has"][rows] = False
        self.r["owner"][rows] = -1
        self.tix[rows] = -1
        self.plan(env, rows, deploy=True)

    def deploy_pending(self, env):
        """The first start's battles (Battles.start ran before any network was known): their deployment now, then
        their memories, men and pending observation afresh. -> whether anything was done."""
        if self.pending is None or not bool(self.pending.any()):
            self.pending = None
            return False
        p = self.pending
        self.pending = None
        self.start_battles(env, p)
        return True

    def replan(self, env, obs):
        """The rows whose plan is due (module doc) plan again on the observation obs (the stacked [2B] dict)."""
        if self.frozen:
            return
        r = self.r
        has = r["has"] & (r["owner"] >= 0)
        B, G = self.B, self.G
        rows = torch.arange(2 * B, device=self.dev)
        b, s = rows % B, rows // B + 1
        t = env.st.t[b].float()
        age = t - r["t_plan"]
        live = has & ~env.st.done[b]
        timer = live & (age >= self.gcfg.replan_s)
        u = env.st.u
        side = u["side"][b]
        out = (u["ms"][b].float() >= 5.5) | (u["men"][b] <= 0) | u["gone"][b]
        own = side == s[:, None]
        mem = (r["main"][..., None] == torch.arange(G, device=self.dev)) & own[..., None]      # [2B, N, G]
        lost = (mem & out[..., None]).sum(1).float()
        broken = ((r["members"] > 0) & (lost > self.gcfg.broken * r["members"])).any(-1)
        lds = env.st.lord_dead_s[b]
        fell = torch.stack([lds.gather(1, (s - 1)[:, None])[:, 0] >= 0, lds.gather(1, (2 - s)[:, None])[:, 0] >= 0], -1)
        lord = (r["lords"] & fell).any(-1)
        event = live & (age >= self.gcfg.replan_min_s) & (broken | lord) & ~timer
        due = timer | event
        if not bool(due.any()):
            return
        mine = r["owner"] == 0
        self.stats.add("replan_timer", (timer & mine).sum())
        self.stats.add("replan_broken", (event & broken & mine).sum())
        self.stats.add("replan_lord", (event & lord & ~broken & mine).sum())
        self.plan(env, due.nonzero().squeeze(1), deploy=False, obs=obs, reason="replan")

    def after(self, env, r_rows, finished):
        """After a decision's simulator steps: the learner rows' reward r_rows [R] added to their plans; the plans of
        the battles that ended (finished [B]) are ready (done)."""
        if self.frozen:
            return
        r = self.r
        add = torch.zeros(2 * self.B, device=self.dev)
        add[env.rows_learn] = r_rows.float()
        r["r_sum"] += add * r["has"].float()
        r["r_disc"] += add * r["has"].float() * self.gamma ** r["n_dec"].float()
        r["n_dec"] += r["has"].long()
        if bool(finished.any()):
            rows = torch.cat([finished, finished]).nonzero().squeeze(1)
            self._finish(rows, done=True)
            mine = r["has"][rows] & (r["owner"][rows] == 0)
            self.stats.add("battles", mine.sum())
            self.stats.add("replans_ended", (r["n_replan"][rows] * mine).sum())
            self.stats.add("battle_min", (env.st.t[rows % self.B].double() * mine).sum() / 60.0)   # their length
            r["has"][rows] = False
            r["owner"][rows] = -1

    def _finish(self, rows, done, v_next=None):
        """The plans of rows end: ready records (the learner's, while training) with their return."""
        r = self.r
        sel = r["has"][rows] & (r["owner"][rows] == 0) & ~r["teach"][rows]    # (the teacher's plans: no record)
        if not self.training or not bool(sel.any()):
            return
        rr = rows[sel]
        rec = self.record(rr)
        target = r["r_sum"][rr].clone()
        if self.ret == "td" and not done and v_next is not None:
            target = target + (self.gamma ** r["n_dec"][rr].float()) * v_next[sel].float()
        if self.ret == "critic":
            target, rec["base"] = self.critic_target(rr, done)
        rec["ret"] = target                       # the plan's return (learn.update's target)
        rec["done"] = torch.full_like(target, float(done))
        rec["new"] = r["new"][rr].clone()         # begun in this chunk (learn.update: clip_fresh / clip_old)
        if self.distinct is not None:
            beh, ok = self.behaviour(rr)
            rec["beh"], rec["beh_ok"] = beh, ok
            if self.distinct.weight > 0:
                bonus = self.distinct.bonus(beh, rec["kind"], ok)                      # [M, G]
                self.bonus.index_add_(0, rr, bonus.sum(-1))
                self.stats.add("distinct_bonus", bonus.sum())
                self.stats.add("distinct_groups", ok.sum())
        self.ready.append(rec)

    def critic_target(self, rows, done):
        """(target, base) [M] of the plans of rows ending now by the critic (module doc; signal.plan_target): the
        critic's value now (v_now; 0 at the battle's end) and at the plan's start (a plan without one: base = target,
        no advantage)."""
        r = self.r
        n = r["n_dec"][rows].float()
        if done or self.v_now is None:
            v_end = torch.zeros_like(n)
        else:
            v_end = torch.nan_to_num(self.v_now[rows], nan=0.0)
        target = gsignal.plan_target(r["r_disc"][rows], n, v_end, float(done), self.gamma)
        v0 = r["v_start"][rows]
        return target, torch.where(torch.isnan(v0), target, v0)

    # --- the critic's value and the behaviour (module doc) -------------------------------------------------------
    def note_value(self, env, value):
        """The critic's value [R] of the learner rows at the decision now (before its re-plans; None: no critic): the
        plans that wait for theirs (a deployment's) take it, the re-plans end and begin with it."""
        if self.frozen:
            return
        if value is None:
            self.v_now = None
            return
        v = torch.full((2 * self.B,), float("nan"), device=self.dev)
        v[env.rows_learn] = value.float()
        self.v_now = v
        r = self.r
        r["v_start"] = torch.where(r["has"] & torch.isnan(r["v_start"]), v, r["v_start"])

    def behave(self, env, action, acting):
        """A decision of the learner rows (action: its Action [R, N]; acting [R, N]: the units that decide): their
        groups' order kinds, run, centre and path into the plans' sums (with --general-distinct only)."""
        if self.frozen or self.distinct is None:
            return
        rows, r, G, B = env.rows_learn, self.r, self.G, self.B
        has = r["has"][rows] & (r["owner"][rows] == 0)
        b, s = rows % B, rows // B + 1
        u = env.st.u
        side = u["side"][b]
        live = (u["men"][b] > 0) & ~u["gone"][b]
        own = (side == s[:, None]) & live
        foe = (side > 0) & (side != s[:, None]) & live
        main = r["main"][rows]
        grp = ((main[..., None] == torch.arange(G, device=self.dev)) & own[..., None] & has[:, None, None]).float()
        act = grp * acting[..., None].float()                                           # [R, N, G]
        kinds = (action.kind[..., None] == torch.arange(len(O.KINDS), device=self.dev)).float()      # [R, N, K]
        r["b_kind"][rows] = r["b_kind"][rows] + torch.einsum("rng,rnk->rgk", act, kinds)
        r["b_run"][rows] = r["b_run"][rows] + (act * action.run[..., None].float()).sum(1)
        r["b_n"][rows] = r["b_n"][rows] + act.sum(1)
        p = torch.stack([u["x"][b], u["z"][b]], -1).float()                             # [R, N, 2]
        cnt = grp.sum(1)
        cent = torch.einsum("rng,rnc->rgc", grp, p) / cnt.clamp(min=1)[..., None]
        ok = cnt > 0
        ok0, okl = r["b_ok0"][rows], r["b_okl"][rows]
        first = ok & ~ok0
        r["b_c0"][rows] = torch.where(first[..., None], cent, r["b_c0"][rows])
        step = (cent - r["b_last"][rows]).norm(dim=-1)
        r["b_path"][rows] = r["b_path"][rows] + torch.where(ok & okl, step, torch.zeros_like(step))
        r["b_last"][rows] = torch.where(ok[..., None], cent, r["b_last"][rows])
        r["b_ok0"][rows], r["b_okl"][rows] = ok0 | ok, okl | ok
        # the side's facing: from its units' centre to the enemy's, once a plan
        mo, mf = own.float(), foe.float()
        co = torch.einsum("rn,rnc->rc", mo, p) / mo.sum(1).clamp(min=1)[:, None]
        cf = torch.einsum("rn,rnc->rc", mf, p) / mf.sum(1).clamp(min=1)[:, None]
        d = cf - co
        nd = d.norm(dim=-1)
        can = (mo.sum(1) > 0) & (mf.sum(1) > 0) & (nd > 1e-3) & ~r["b_fwd_ok"][rows] & has
        r["b_fwd"][rows] = torch.where(can[:, None], d / nd.clamp(min=1e-3)[:, None], r["b_fwd"][rows])
        r["b_fwd_ok"][rows] = r["b_fwd_ok"][rows] | can

    def behaviour(self, rows):
        """(beh [M, G, len(BEH)], ok [M, G]) of the plans of rows ending now: their groups' behaviour (BEH) and which
        groups count (own active groups whose units decided during the plan)."""
        r, G = self.r, self.G
        n = r["b_n"][rows]
        n1 = n.clamp(min=1)
        share = r["b_kind"][rows] / n1[..., None]
        run = r["b_run"][rows] / n1
        d = r["b_last"][rows] - r["b_c0"][rows]
        f = r["b_fwd"][rows][:, None]                                                    # [M, 1, 2]
        fwd = d[..., 0] * f[..., 0] + d[..., 1] * f[..., 1]
        lat = -d[..., 0] * f[..., 1] + d[..., 1] * f[..., 0]
        seen = r["b_ok0"][rows] & r["b_fwd_ok"][rows][:, None]
        fwd = torch.where(seen, fwd / 100.0, torch.zeros_like(fwd)).clamp(-5.0, 5.0)
        lat = torch.where(seen, lat / 100.0, torch.zeros_like(lat)).clamp(-5.0, 5.0)
        path = (r["b_path"][rows] / 100.0).clamp(max=10.0)
        length = (r["n_dec"][rows].float() / 60.0)[:, None].expand_as(path)
        beh = torch.cat([share, torch.stack([run, fwd, lat, path, length], -1)], -1)
        ok = (n > 0) & r["active"][rows][:, :G]
        return beh, ok

    def execute(self, env, acting):
        """After a decision's simulator steps (before after(): the plans in force are the decision's): how well each of
        the learner's units with a mission carried it out (mission.measure, 0..1) into the log's sums by kind, and ->
        the executor's bonus [R, N] of the decision (signal.exec_bonus: exec_w x (the measure - its kind's running mean
        before this decision); 0 for the units not counted) or None (no measures, or exec_w 0: nothing to pay). acting
        [R, N]: the units that decided. The bonus goes into the units' advantage only (ppo.update), never the reward."""
        if self.frozen or not self.measure:
            return None
        rows, r, G, B = env.rows_learn, self.r, self.G, self.B
        has = r["has"][rows] & (r["owner"][rows] == 0)
        C = self.gcfg.target_grid ** 2
        half = self.mcfg.grid_half / self.gcfg.target_grid             # a coarse tile's half side, m
        m, ok, k = gmission.measure(env.st.u, rows % B, rows // B + 1, r["main"][rows], r["kind"][rows],
                                    r["target"][rows], r["tgt"][rows], acting & has[:, None], G, C, half, self.gcfg)
        K = self.gcfg.kinds
        okf = ok.to(torch.float64)
        kk = k.clamp(0, K - 1).reshape(-1)
        tot = torch.zeros(2, K, dtype=torch.float64, device=self.dev)
        tot[0].index_add_(0, kk, (m.double() * okf).reshape(-1))
        tot[1].index_add_(0, kk, okf.reshape(-1))
        self.stats.exec += tot
        mean = (tot[0] / tot[1].clamp(min=1)).float()
        seen = tot[1] > 0
        first = torch.isnan(self.exec_base) & seen
        base = torch.where(first, mean, self.exec_base)                 # (a kind's first decision: its own mean)
        self.exec_base = torch.where(seen, base + self.exec_rate * (mean - base), base)
        if not self.exec_w:
            return None
        bonus = gsignal.exec_bonus(m, torch.nan_to_num(base)[k.clamp(0, K - 1)], self.exec_w) * ok.float()
        self.stats.add("exec_bonus", bonus.double().sum())
        self.stats.add("exec_bonus_abs", bonus.double().abs().sum())
        self.stats.add("exec_units", okf.sum())
        return bonus

    def take_bonus(self, env):
        """The distinct bonus [R] due to the learner rows' reward of this decision (then zeroed)."""
        out = self.bonus[env.rows_learn]
        self.bonus.zero_()
        return out

    # --- the executor's input ---------------------------------------------------------------------------------------
    def task(self, rows, obs_r, fr):
        """{"task" [R, N, task_dim], "task_on" [R, N], "task_geo" [R, N, 4], "task_plan" [R]} of rows [R] on their
        observation obs_r (the rows' dict) and frame fr (module doc)."""
        r, G, cfg = self.r, self.G, self.gcfg
        C = cfg.target_grid ** 2
        has = r["has"][rows]
        main = r["main"][rows]
        own, attend = obs_r["own"], obs_r["attend"]
        tok = obs_r["tokens"]
        pos = obs_r["pos"].float() * ob.POS                                          # [R, N, 2] forward, lateral
        fc, fs = tok[..., ob.INDEX["face_cos"]].float(), tok[..., ob.INDEX["face_sin"]].float()
        seen = tok[..., ob.INDEX["seen"]] > 0.5
        grp = main[..., None] == torch.arange(2 * G, device=self.dev)               # [R, N, 2G]
        w = (grp & (attend & (own | seen))[..., None]).float()
        n = w.sum(1)                                                                 # [R, 2G]
        cent = torch.einsum("rng,rnc->rgc", w, pos) / n.clamp(min=1)[..., None]
        cent_ok = n > 0
        t = r["target"][rows]
        is_g = t < 2 * G
        gi = t.clamp(max=2 * G - 1)
        gp = cent.gather(1, gi[..., None].expand(-1, -1, 2))
        gok = cent_ok.gather(1, gi)
        tw = r["tgt"][rows]                                                          # [R, G, 2] world (a tile)
        tf, tl = fr.point(tw[..., 0], tw[..., 1])
        tp = torch.stack([tf, tl], -1)
        tok_t = torch.isfinite(tp).all(-1) & (t >= 2 * G) & (t < 2 * G + C)       # a coarse tile
        p = torch.where(is_g[..., None], gp, torch.nan_to_num(tp))
        ok = torch.where(is_g, gok, tok_t)                                           # [R, G]
        mi = main.clamp(0, G - 1)
        up = p.gather(1, mi[..., None].expand(-1, -1, 2))                           # [R, N, 2] the unit's target
        uok = ok.gather(1, mi) & (main >= 0) & (main < G)
        d = up - pos
        df, dl = d[..., 0], d[..., 1]
        rf, rl = df * fc + dl * fs, -df * fs + dl * fc
        dist = torch.sqrt(rf * rf + rl * rl)
        far = dist > 1e-3
        cos = torch.where(far, rf / dist.clamp(min=1e-3), torch.ones_like(rf))
        sin = torch.where(far, rl / dist.clamp(min=1e-3), torch.zeros_like(rl))
        dn = (dist / cfg.dist_m).clamp(max=cfg.dist_cap)
        z = torch.zeros_like(dn)
        geo = torch.stack([torch.where(uok, cos, z), torch.where(uok, sin, z), torch.where(uok, dn, z),
                           (~uok).float()], -1)
        on = has[:, None] & own & obs_r["ctrl"] & (main >= 0) & (main < G)
        geo = geo * on[..., None].float()
        task = torch.cat([r["mix"][rows], r["kc"][rows], geo], -1) * on[..., None].float()
        return {"task": task, "task_on": on, "task_geo": geo, "task_plan": torch.where(has, self.tix[rows],
                                                                                      torch.full_like(rows, -1))}

    # --- chunks (rollout.collect) -----------------------------------------------------------------------------------
    @torch.no_grad()
    def begin_chunk(self, env):
        """A chunk's start: records on; the learner's plans in force get their cards from the general's weights now
        (the update computes them again from the same weights: the executor's probabilities in the rollout and in the
        update agree) and their value; their choices keep the log-probability they were made with (PPO's honest
        ratio for a plan made before the last update: the policy now / the policy that chose), and they open the
        chunk's table."""
        self.training = True
        r = self.r
        rows = (r["has"] & (r["owner"] == 0)).nonzero().squeeze(1)
        self.table = []
        r["new"].zero_()                          # the plans in force were begun before this chunk
        self.tix.fill_(-1)
        if rows.numel() and self.learner is not None:
            rec = self.record(rows)
            gen = self.learner.general
            c = gen.cards(rec)
            mix, kc = gen.unit_cards(c, rec["kind"], rec["main"])
            d = gen.decide(rec, c, act={k: rec[k] for k in CHOICES})
            r["mix"][rows], r["kc"][rows], r["places"][rows] = mix, kc, c["places"]
            r["value"][rows] = d["value"]
            rec = self.record(rows)
            self.table.append(rec)
            self.tix[rows] = torch.arange(rows.numel(), device=self.dev)

    def end_chunk(self):
        """-> {"table": the chunk's plan records [P, ..] (task_plan indexes it), "ready": the ended plans' records with
        their return since the last chunk}; both None when empty."""
        table = _cat(self.table) if self.table else None
        ready = _cat(self.ready)
        self.table, self.ready = None, []
        return {"table": table, "ready": ready}

    # --- the batch's changes ----------------------------------------------------------------------------------------
    def narrow(self, keep):
        """Only the battles keep [b] stay (Battles.narrow)."""
        two = torch.cat([keep, keep + self.B])
        self.r = {k: v[two] for k, v in self.r.items()}
        self.b = {k: v[keep] for k, v in self.b.items()}
        self.tix = self.tix[two]
        self.bonus = self.bonus[two]
        if self.v_now is not None:
            self.v_now = self.v_now[two]
        if self.pending is not None:
            self.pending = self.pending[keep]
        self.B = len(keep)

    def snap(self, idx):
        """The rows of the battles idx as a snapshot's {"g/<k>": [b, 2, ...]} (ability_branch.snapshot)."""
        B = self.B
        out = {}
        for k in ("has", "main", "target", "tgt", "mix", "kc"):
            v = self.r[k]
            out["g/" + k] = v.reshape(2, B, *v.shape[1:]).transpose(0, 1)[idx]
        return out

    def branch(self, snap, B):
        """A frozen copy for a branch env of B battles from a snapshot (ability_branch.branch_env): the plans of the
        snapshot, no re-plans, no deployment, no records."""
        new = Runtime.__new__(Runtime)
        new.__dict__.update({k: v for k, v in self.__dict__.items() if k not in ("r", "b", "tix", "ready", "table")})
        new.B = B
        new.r = self._rows(2 * B)
        for k in ("has", "main", "target", "tgt", "mix", "kc"):
            v = snap["g/" + k]
            new.r[k] = v.transpose(0, 1).reshape(-1, *v.shape[2:]).clone()
        new.r["owner"] = torch.where(new.r["has"], torch.zeros_like(new.r["owner"]), new.r["owner"])
        new.b = self._battles(B)
        new.tix = torch.full((2 * B,), -1, dtype=torch.long, device=self.dev)
        new.ready, new.table, new.pending = [], None, None
        new.v_now, new.distinct, new.bonus = None, None, torch.zeros(2 * B, device=self.dev)
        new.brancher = None
        new.frozen = True
        new.training = False
        new.stats = Counts(self.dev, self.gcfg.kinds, self.S)
        return new

    # --- the log -----------------------------------------------------------------------------------------------------
    def take_stats(self):
        """The log's numbers since the last call (one read of the device): plans (all, at the deployment, the share
        placed by the general), groups a plan (ours, theirs), the mission kinds' shares and their entropy, re-plans a
        battle (of the battles that ended since the last call: early in a run only the short ones have, so it grows
        with their length) and a minute of those battles (replans_per_min: ~2 at 30 s plans, the events above it;
        battle_min: their mean length in minutes) and by reason, the most chosen deployment tiles ("row_column" of the side's 16 x 16), units laid out,
        the targets' shares (our group, theirs, a coarse tile; groups: both), the share of our units whose group number
        changed at a re-plan (group_churn) and of our groups kept whose mission kind changed (kind_switch); each kind's
        share by name (kind_share_<kind>), the share of the (kind, target) pairs masked (masked_pairs); the teacher's
        share of the learner's plans (teach_share); with the measures, each kind's mean measure a unit and decision
        (exec_<kind>), the running means (exec_base) and the bonus paid a unit and decision (exec_bonus: signed, ~0 by
        its baseline; exec_bonus_abs). The general's numbers are of its own plans (not the teacher's); the measures of
        all the learner's."""
        s, kinds, places, ex = self.stats.take()
        names = mission_kinds(self.gcfg)
        n = max(1.0, s["plans"])
        tot = kinds.sum()
        p = kinds / tot if tot else kinds
        ent = float(-(p[p > 0] * np.log(p[p > 0])).sum()) if tot else 0.0
        n_s = self.mcfg.sectors
        top = np.argsort(-places)[:6]
        tg = np.array([s["target_own"], s["target_enemy"], s["target_tile"]])
        tg = tg / tg.sum() if tg.sum() else tg
        return {"plans": int(s["plans"]), "deploy_plans": int(s["deploys"]),
                "deploy_net_share": round(s["deploys_net"] / max(1.0, s["deploys"]), 3),
                "groups_own": round(s["groups_own"] / n, 2), "groups_enemy": round(s["groups_enemy"] / n, 2),
                "kinds": [round(float(x), 3) for x in p], "kind_entropy": round(ent, 3),
                "kind_entropy_max": round(math.log(self.gcfg.kinds), 3),
                "replans_per_battle": round(s["replans_ended"] / max(1.0, s["battles"]), 2),
                "replans_per_min": round(s["replans_ended"] / s["battle_min"], 3) if s["battle_min"] else 0.0,
                "battle_min": round(s["battle_min"] / max(1.0, s["battles"]), 2),
                "replan_reasons": {k: int(s[f"replan_{k}"]) for k in ("timer", "broken", "lord")},
                "places_top": {f"{int(i) // n_s}_{int(i) % n_s}": int(places[i]) for i in top if places[i] > 0},
                "laid_units": int(s["laid_units"]),
                "target_own": round(float(tg[0]), 3), "target_enemy": round(float(tg[1]), 3),
                "target_tile": round(float(tg[2]), 3), "target_group": round(float(tg[0] + tg[1]), 3),
                "group_churn": round(s["churn_moved"] / max(1.0, s["churn_units"]), 4),
                "kind_switch": round(s["kind_switched"] / max(1.0, s["kind_kept_groups"]), 4),
                **({"distinct_bonus": round(s["distinct_bonus"] / max(1.0, s["distinct_groups"]), 5),
                    "distinct_weight": round(self.distinct.weight, 6)} if self.distinct is not None else {}),
                **{f"kind_share_{k.name}": round(float(p[k.index]), 3) for k in names},
                "masked_pairs": round(1.0 - s["pairs_ok"] / s["pairs_all"], 4) if s["pairs_all"] else 0.0,
                **({"teach_share": round(s["teach_plans"] / (s["plans"] + s["teach_plans"]), 4)} if self.teacher > 0
                   else {}),
                **({f"exec_{k.name}": round(float(ex[0, k.index] / ex[1, k.index]), 4) for k in names
                    if ex[1, k.index] > 0} if self.measure else {}),
                **({"exec_base": [round(float(x), 4) for x in self.exec_base.tolist()],
                    "exec_units": int(ex[1].sum()), "exec_weight": round(float(self.exec_w), 6)} if self.measure else {}),
                **({"exec_bonus": round(s["exec_bonus"] / max(1.0, s["exec_units"]), 5),
                    "exec_bonus_abs": round(s["exec_bonus_abs"] / max(1.0, s["exec_units"]), 5)}
                   if self.measure and self.exec_w else {})}

"""The general in the companion (S28): a network with a general (tools/nn/general/executor.py GeneralActor) plans its
side of a real battle as in the simulator - the same Runtime (tools/nn/general/runtime.py) on a view of the game's state
shaped like the simulator's batch: one battle, its units in the state document's order (GameBattles)
(docs/en/training/general.md "In the game").

    the battle's first state   the plan ("Замысел") on the armies and the tiles of the state; the deployment stays the
                               bridge's (deploy "script": the general forms groups and gives missions, places nobody)
    every state                the re-plans by Runtime.replan's rules on the battle's time (the state's t: the game's
                               speed-up changes nothing) - every replan_s, a broken group, a lord fallen; then the task
                               of every own unit, the executor's input ("Задание отряда")
    the log                    every plan: its groups (units), mission kind and targets; the re-plan's reason

    gen = GameGeneral(actor, gcfg, greedy=False, side=1)
    task, plans = gen.decide(battle, state, obs, memory)   # exchange.Battle / arrays' state, the side's observation
    actor = with_task(actor, task)                          # decide.act's actor: the task joins its input
"""
import dataclasses
from types import SimpleNamespace

import numpy as np
import torch

from tools.nn.general import runtime as gruntime
from tools.nn.model import policy as model_policy
from tools.nn.model.decide import frame_to

SCRIPT = 1                 # league's code of a scripted side (runtime.owners: no general plans it)


def config(actor, sets=None):
    """The GeneralConfig the companion plays the actor's general with: the checkpoint's, deploy "script" (the bridge
    does not move units: the deployment is the game's), and sets {field: value} over it (the companion's
    --general-set). None for an actor without a general."""
    gcfg = getattr(actor, "gcfg", None)
    if gcfg is None:
        return None
    fields = {f.name: f for f in dataclasses.fields(gcfg)}
    over = {"deploy": "script"}
    for k, v in (sets or {}).items():
        if k not in fields:
            raise ValueError(f"general setting {k!r}: not a GeneralConfig field ({sorted(fields)})")
        cur = getattr(gcfg, k)
        over[k] = type(cur)(v) if not isinstance(cur, bool) else str(v).lower() in ("1", "true", "yes")
    if over["deploy"] != "script":
        raise ValueError("general deploy: only 'script' in the game (the bridge does not move units yet)")
    return dataclasses.replace(gcfg, **over)


def parse_sets(items):
    """{field: text} of --general-set KEY=VALUE items."""
    out = {}
    for it in items or ():
        k, sep, v = it.partition("=")
        if not sep:
            raise ValueError(f"--general-set {it!r}: KEY=VALUE")
        out[k.strip().replace("-", "_")] = v.strip()
    return out


class GameBattles:
    """What Runtime reads of a rollout.Battles, for one game battle (B = 1, its N units in the state document's order):
    st.u - the unit fields [1, N] it reads (x, z, b, side, men, hp, hp0, ms, gone, vis, cost, width, sp_h, sp_v), st.t
    [1] (the battle's time, s), st.done, st.lord_dead_s [1, 2] (s since the side's lord fell, -1 standing), st.keys;
    setup.bounds [1, 4]; mem[s].frame (the side's observation frame); ctrl [1, 2] (the side the network commands is the
    learner, the other a script); past (none); B, N, device, rows_learn.

    The game's state over it (update): positions, men, health share, morale state, visibility as read; a reading that
    is missing keeps the last one; a unit with men and no position has left the map (gone, as the enemy's script reads
    it); the static numbers (start health, cost, formation) are the simulator's of the unit's key (params.static, the
    formation of exchange.formation_shape); a lord falls - dead, shattered or gone - as the simulator's lord_dead_s."""

    def __init__(self, battle, side, params, device="cpu"):
        names = battle.names
        N = len(names)
        self.B, self.N, self.device, self.side = 1, N, torch.device(device), side
        self.past, self.battle_slot = [], torch.zeros(1, dtype=torch.long, device=self.device)
        self.rows_learn = torch.tensor([side - 1], device=self.device)
        self.ctrl = torch.full((1, 2), SCRIPT, dtype=torch.long, device=self.device)
        self.ctrl[0, side - 1] = gruntime.LEARNER
        self.setup = SimpleNamespace(bounds=torch.as_tensor(np.asarray(battle.setup.bounds), dtype=torch.float32,
                                                            device=self.device))
        keys = list(battle.setup.keys[0])
        factions = battle.setup.factions[0]
        sides = np.asarray(battle.side)
        hp0, cost = np.zeros(N, np.float32), np.zeros(N, np.float32)
        for i, k in enumerate(keys):
            if k in params.units:
                row = params.static(k, factions[int(sides[i]) - 1] if sides[i] in (1, 2) else None)
                hp0[i], cost[i] = row["hp0"], row["cost"]
        sh = battle.shape or {}
        f = (lambda a: torch.as_tensor(np.nan_to_num(np.asarray(a, np.float32)), device=self.device)[None])
        z = torch.zeros(1, N, device=self.device)
        self.u = {"side": torch.as_tensor(sides, dtype=torch.long, device=self.device)[None],
                  "hp0": f(hp0), "cost": f(cost),
                  "width": f(sh.get("width", np.zeros(N))), "sp_h": f(sh.get("sp_h", np.zeros(N))),
                  "sp_v": f(sh.get("sp_v", np.zeros(N))),
                  "x": z.clone(), "z": z.clone(), "b": z.clone(), "men": z.clone(), "hp": z + 1.0, "ms": z + 1.0,
                  "gone": torch.zeros(1, N, dtype=torch.bool, device=self.device),
                  "vis": torch.ones(1, N, dtype=torch.bool, device=self.device)}
        self.lord = torch.as_tensor(np.asarray(battle.setup.lord[0], bool), device=self.device)[None]
        self.st = SimpleNamespace(u=self.u, t=torch.zeros(1, device=self.device),
                                  done=torch.zeros(1, dtype=torch.bool, device=self.device),
                                  lord_dead_s=torch.full((1, 2), -1.0, device=self.device), keys=[keys])
        self.mem = {1: SimpleNamespace(frame=None), 2: SimpleNamespace(frame=None)}

    def update(self, state, memory=None):
        """The game's state now (exchange.arrays' dict [1, N]) and the side's observation memory (its frame)."""
        u, dev = self.u, self.device
        t = (lambda k: torch.as_tensor(np.asarray(state[k], np.float32), device=dev))
        x, z = t("x"), t("z")
        seen = torch.isfinite(x) & torch.isfinite(z)
        u["x"] = torch.where(seen, x, u["x"])
        u["z"] = torch.where(seen, z, u["z"])
        for k in ("b", "hp"):
            v = t(k)
            u[k] = torch.where(torch.isfinite(v), v, u[k])
        u["men"] = torch.nan_to_num(t("men"), nan=0.0)
        # the morale state (MORALE_STATES 1-7; >= 5.5 routs): not read - the flags' (shattered 7, routing 6), else the
        # last one
        ms = t("ms")
        rs = torch.as_tensor(np.asarray(state["r"], bool), device=dev)
        sh = torch.as_tensor(np.asarray(state["s"], bool), device=dev)
        flags = torch.where(sh, torch.full_like(ms, 7.0), torch.where(rs, torch.full_like(ms, 6.0), u["ms"]))
        u["ms"] = torch.where(torch.isfinite(ms), ms, flags)
        u["vis"] = torch.as_tensor(np.asarray(state["vis"], bool), device=dev)
        u["gone"] = (u["men"] > 0) & ~seen
        st = self.st
        st.t = torch.as_tensor(np.asarray(state["t"], np.float32).reshape(1), device=dev)
        present = u["side"] > 0
        standing = present & (u["men"] > 0) & ~u["gone"] & ~sh
        has = torch.stack([(self.lord & (u["side"] == s)).any(1) for s in (1, 2)], 1)
        up = torch.stack([(self.lord & (u["side"] == s) & standing).any(1) for s in (1, 2)], 1)
        fell = has & ~up
        # (sticky: a lord who fell stays fallen; the simulator's seconds since the fall)
        since = torch.where(st.lord_dead_s >= 0, st.lord_dead_s, torch.zeros_like(st.lord_dead_s))
        st.lord_dead_s = torch.where(fell | (st.lord_dead_s >= 0), since + self._dt(st.t), st.lord_dead_s)
        self._t_last = float(st.t[0])
        if memory is not None:
            for s in (1, 2):
                self.mem[s].frame = memory.frame

    def _dt(self, t):
        last = getattr(self, "_t_last", None)
        return 0.0 if last is None else max(0.0, float(t[0]) - last)


class GameGeneral:
    """The general of one side in the companion (module doc): a Runtime a battle on GameBattles; its plans' records
    (plan_record) for the log."""

    def __init__(self, actor, gcfg=None, greedy=False, side=1, params=None):
        from tools.nn.sim.params import load as load_params
        assert getattr(actor, "gcfg", None) is not None, "an actor without a general"
        self.actor, self.gcfg = actor, gcfg or config(actor)
        self.greedy, self.side = greedy, side
        self.params = params or load_params()
        self.device = next(actor.parameters()).device
        gruntime.unit_table()                      # every passport's slot, once (seconds): not at the battle's start
        self.batch, self.env, self.rt, self.off = None, None, None, None
        self.plans = 0

    @property
    def row(self):
        return self.side - 1

    def start(self, battle, state, memory=None):
        """A new battle: its view, the Runtime and the first plan (Runtime.start_battles: deploy, on the state).
        -> its records."""
        self.batch, self.plans, self.off = battle.batch, 0, None
        self.env = GameBattles(battle, self.side, self.params, self.device)
        missing = [k for k in battle.setup.keys[0] if k and k not in gruntime.unit_table().index]
        if missing:
            self.rt = None
            self.off = f"unit {missing[0]!r} has no passport (config/nn/units.json): the battle without the general"
            return [{"t": float(np.asarray(state["t"]).reshape(-1)[0]), "off": self.off}]
        self.rt = gruntime.Runtime(self.env, self.actor.cfg, self.gcfg)
        self.rt.learner, self.rt.greedy, self.rt.pending = self.actor, self.greedy, None
        self.env.update(state, memory)
        n0 = self._counts()
        self.rt.start_battles(self.env, torch.ones(1, dtype=torch.bool, device=self.device))
        return self._records(n0, "deploy")

    def _counts(self):
        s = self.rt.stats
        return {k: s[k] for k in ("plans", "replan_timer", "replan_broken", "replan_lord")}

    def _records(self, n0, reason=None):
        n1 = self._counts()
        if n1["plans"] <= n0["plans"]:
            return []
        if reason is None:
            why = [k for k in ("timer", "broken", "lord") if n1[f"replan_{k}"] > n0[f"replan_{k}"]]
            reason = why[0] if why else "replan"
        self.plans += 1
        return [plan_record(self.rt, self.row, self.env, self._names, self.plans, reason)]

    @torch.no_grad()
    def decide(self, battle, state, obs, memory):
        """One decision: the battle's state now (a new battle: start), the re-plans due (on the side's observation
        obs), the task. -> (task dict for the executor's input or None, [plan records made now])."""
        self._names = list(battle.names)
        recs = []
        if battle.batch != self.batch:
            recs = self.start(battle, state, memory)
        if self.rt is None:
            return None, recs
        self.env.update(state, memory)
        obs_t = model_policy.to_torch(obs, self.device)
        two = {k: torch.cat([obs_t[k], obs_t[k]]) for k in ("tokens", "own", "attend", "pos")}
        n0 = self._counts()
        self.rt.replan(self.env, two)
        recs += self._records(n0)
        fr = frame_to(obs.frame, self.device)
        task = self.rt.task(self.env.rows_learn, obs_t, fr)
        return task, recs


def target_name(i, G, S, n_sectors):
    """A mission's target index (net.General.decide): our group, their group, a tile ("row_column" of the side's
    sectors) or none."""
    i = int(i)
    if i < G:
        return f"g{i}"
    if i < 2 * G:
        return f"e{i - G}"
    if i < 2 * G + S:
        k = i - 2 * G
        return f"tile {k // n_sectors}_{k % n_sectors}"
    return "none"


def plan_record(rt, row, env, names, n, reason):
    """The log's record of the plan in force of a row: {"n", "t", "reason", "groups": [{"g", "units", "kind", "t1",
    "t2"}], "enemy": [{"g", "units"}]} - our groups gN, theirs eN (as the targets name them)."""
    r, G, S = rt.r, rt.G, rt.S
    main = r["main"][row].tolist()
    active = r["active"][row].tolist()
    kind, t1, t2 = r["kind"][row].tolist(), r["t1"][row].tolist(), r["t2"][row].tolist()
    ns = rt.mcfg.sectors
    members = {g: [names[i] for i, m in enumerate(main) if m == g] for g in range(2 * G)}
    ours = [{"g": f"g{g}", "units": members[g], "kind": int(kind[g]), "t1": target_name(t1[g], G, S, ns),
             "t2": target_name(t2[g], G, S, ns)} for g in range(G) if active[g]]
    theirs = [{"g": f"e{g - G}", "units": members[g]} for g in range(G, 2 * G) if active[g]]
    return {"n": n, "t": round(float(r["t_plan"][row]), 2), "reason": reason, "groups": ours, "enemy": theirs}


def describe(rec, limit=6):
    """One line of a plan record (the companion's console)."""
    if "off" in rec:
        return f"general off: {rec['off']}"
    short = (lambda u: u.removeprefix("own_").removeprefix("enemy_"))
    sizes = {e["g"]: len(e["units"]) for e in rec["enemy"]}
    tg = (lambda t: f"{t}({sizes[t]})" if t in sizes else t)
    parts = []
    for g in rec["groups"]:
        units = ", ".join(short(u) for u in g["units"][:limit])
        more = f" +{len(g['units']) - limit}" if len(g["units"]) > limit else ""
        parts.append(f"{g['g']} [{units}{more}] kind {g['kind']} -> {tg(g['t1'])}, {tg(g['t2'])}")
    return (f"general plan {rec['n']} t {rec['t']:.1f} s ({rec['reason']}): {len(rec['groups'])} groups, enemy "
            f"{len(rec['enemy'])} | " + " | ".join(parts))


def with_task(actor, task):
    """decide.act's actor with the task joined to its input (TaskActor); the actor itself without a task."""
    return actor if task is None else TaskActor(actor, task)


class TaskActor:
    """An actor whose input gets the task (the general's executor reads "task", "task_on"): the wrapped actor's cfg
    and parameters, its act on obs_t + task (decide.act: tools/nn/model stays as it is)."""

    def __init__(self, actor, task):
        self.actor, self.task, self.cfg = actor, task, actor.cfg

    def parameters(self):
        return self.actor.parameters()

    def act(self, obs_t, *args, **kwargs):
        return self.actor.act(dict(obs_t, **self.task), *args, **kwargs)

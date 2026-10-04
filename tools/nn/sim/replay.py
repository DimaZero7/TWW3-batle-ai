"""Policies without a network: hold, attack the nearest enemy, and replaying the orders of recorded
battles with contact-phase synchronisation (docs/en/training/simulator.md).

Replay: for every recorded second and unit, the order that recording implies -
    a recorded target (fought or shot)   -> ATTACK it (a missile unit closes to its range)
    in melee without a target            -> ATTACK the nearest enemy (CA's planner leaves the
                                            target empty in ~70 % of its melee seconds, the
                                            game's AI in ~6 %: HOLD made the two sides differ)
    otherwise                            -> MOVE to the recorded order point (ox, oz),
                                            running if the unit was running.
Each unit's replay clock waits for actual contact/separation at recorded approaches/break-offs.
The approach's running flag and target survive a late contact; melee keeps that phase's target.
Phases continue until the wall-clock recording end plus `grace_s` (120 s), then every unit attacks the nearest enemy
(check.py stops a whole battle when its recording ends, and keeps a pair on its last orders).
"""
import numpy as np
import torch

from tools.nn.sim import orders as O


def hold(st):
    return O.hold(st.B, st.N, st.device)


def nearest_attack(st, run=True):
    """Every standing unit attacks the nearest standing enemy (missile units close to their range)."""
    u = st.u
    alive = (u["side"] > 0) & (u["men"] > 0) & ~u["gone"]
    standing = alive & ~u["r"]
    dx = u["x"][:, None, :] - u["x"][:, :, None]
    dz = u["z"][:, None, :] - u["z"][:, :, None]
    d = torch.sqrt(dx * dx + dz * dz)
    enemy = (u["side"][:, :, None] != u["side"][:, None, :]) & standing[:, None, :]
    d = torch.where(enemy, d, torch.full_like(d, 1e9))
    near = d.argmin(2)
    has = d.min(2).values < 1e9
    o = O.hold(st.B, st.N, st.device)
    o.kind = torch.where(has & standing, O.ATTACK, O.HOLD)
    o.target = torch.where(has & standing, near, torch.full_like(near, -1))
    o.run = torch.full_like(o.run, run)
    return o


def half_depth(men, width, spacing=1.5):
    """Half the depth of a formation of `men` with a `width` front (numpy); 0 for a single man."""
    files = np.maximum(1, np.round(np.asarray(width, float) / spacing))
    files = np.minimum(files, np.maximum(men, 1))
    return np.where(men > 1, np.ceil(np.maximum(men, 1) / files) * spacing / 2, 0.0)


def recorded_orders(battle, slot_of, N, widths=None, spacing=1.5, fight_nearest=True, leave_m=10.0, leavers=None):
    """Orders implied by a recorded battle (tools/nn/gamedata.Battle), one row per recorded second:
    dict of arrays [T, N] kind, x, z, target, run in the simulator's slots (slot_of: recorded index
    -> slot). The game records the order's point at the formation's front (measured: half a depth
    ahead of the centre); widths [recorded index] (m) turn it into the centre the simulator goes to.
    phase (0: ordinary, 1: approach, 2: separation), end (exclusive row), and phase_target
    describe the contact clock; Replay consumes them, not the combat engine.
    leavers [recorded index] (None: all): units whose recorded point in melee is a move order in force;
    such a unit in melee without a recorded target whose point is leave_m or more away moves there (it
    walks out of the fight, as in the game; leave_m 0: never) instead of attacking the nearest enemy.
    fight_nearest: a bool, or one per recorded index: a unit in melee without a recorded target (and not
    leaving) attacks the nearest enemy; where false it holds (HOLD). The network's units hold: the game
    records the engine target of its attack orders in melee 97 % of the time (all fair network runs: 75 %
    of their melee unit-seconds with one, 2.7 % without), so in melee without one they are under its HOLD
    (12 %) or moving (4 %).
    The network's units and every missile unit are leavers; the melee units of CA's planner and the
    game's AI are not: in melee their recorded point lies anywhere (the planner's: often the enemy's
    start beyond it) while they fight on at the attack rate (build/simbatch/leave_dir.py: the planner's
    melee units with a point 10 m or more away kill 0.17-0.29 a second in every direction, the
    network's under such a move 0.003-0.04 against 0.19 attacking)."""
    f = battle.f
    T = len(battle.t)
    kind = np.full((T, N), O.HOLD, dtype=np.int64)
    x = np.zeros((T, N), dtype=np.float32)
    z = np.zeros((T, N), dtype=np.float32)
    target = np.full((T, N), -1, dtype=np.int64)
    run = np.zeros((T, N), dtype=bool)
    phase = np.zeros((T, N), dtype=np.int64)
    end = np.broadcast_to(np.arange(1, T + 1)[:, None], (T, N)).copy()
    phase_target = np.full((T, N), -1, dtype=np.int64)
    # The nearest living enemy of each unit each second (for fights without a recorded target).
    xs, zs = np.nan_to_num(f["x"], nan=1e6), np.nan_to_num(f["z"], nan=1e6)
    dist = np.hypot(xs[:, :, None] - xs[:, None, :], zs[:, :, None] - zs[:, None, :])
    foe = (battle.side[:, None] != battle.side[None, :])[None] & (np.nan_to_num(f["men"]) > 0)[:, None, :]
    nearest = np.where(foe, dist, np.inf).argmin(axis=2)
    nearest_of = list(fight_nearest) if isinstance(fight_nearest, (list, tuple, np.ndarray)) else [fight_nearest] * len(slot_of)
    for i, s in enumerate(slot_of):
        tg = battle.target[:, i]
        ox, oz = np.nan_to_num(f["ox"][:, i]), np.nan_to_num(f["oz"][:, i])
        # In melee without a recorded target (CA's planner leaves it empty most of the time):
        # the nearest enemy, as the game's own AI records it - unless the unit is told to go
        # somewhere else (it leaves the fight).
        leaving = np.zeros(T, dtype=bool)
        if leave_m > 0 and (leavers is None or leavers[i]):
            leaving = np.hypot(ox - np.nan_to_num(f["x"][:, i]), oz - np.nan_to_num(f["z"][:, i])) >= leave_m
        if nearest_of[i]:
            tg = np.where((tg < 0) & f["m"][:, i] & ~leaving, nearest[:, i], tg)
        ok_t = tg >= 0
        mapped = np.where(ok_t, np.array(slot_of)[np.clip(tg, 0, None)], -1)
        attack = ok_t & (battle.side[np.clip(tg, 0, None)] != battle.side[i])
        if widths is not None and widths[i]:
            back = half_depth(np.nan_to_num(f["men"][:, i]), widths[i], spacing)
            br = np.radians(np.nan_to_num(f["b"][:, i]))
            ox, oz = ox - back * np.sin(br), oz - back * np.cos(br)
        fighting = f["m"][:, i] & ~attack & ~leaving
        kind[:, s] = np.where(attack, O.ATTACK, np.where(fighting, O.HOLD, O.MOVE))
        target[:, s] = np.where(attack, mapped, -1)
        x[:, s], z[:, s] = ox, oz
        running = f["f"][:, i]
        # Look ahead without wrapping the recording's first flags into its last seconds.
        run[:, s] = running
        for ahead in (1, 2):
            run[:-ahead, s] |= running[ahead:]
        active = (f["men"][:, i] > 0) & ~f["r"][:, i] & ~f["s"][:, i]
        contact = f["m"][:, i] & active
        edges = np.diff(np.r_[False, contact, False].astype(int))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            attacks = np.flatnonzero(attack[a:b]) + a
            t = int(target[attacks[0], s]) if len(attacks) else -1
            phase_target[a:b, s] = t
            # Only a recorded approach to a recorded melee contact is latched: shooting
            # ATTACKs and unrelated moves must not wait for a contact that never existed.
            start = a
            while (start > 0 and active[start - 1] and not contact[start - 1]
                   and not f["fire"][start - 1, i]
                   and (run[start - 1, s] or attack[start - 1])):
                start -= 1
            if start < a and t >= 0:
                phase[start:a, s] = 1
                end[start:a, s] = a
                phase_target[start:a, s] = t
                # Keep the recorded route and walk/run timing before contact is due.
                # Once a run begins it stays running until contact, rather than stopping
                # because the recording (but not this simulation) has reached the enemy.
                run[start:a, s] = np.maximum.accumulate(run[start:a, s])
            # Preserve leave_m's inference: only authorised leavers with no attack target
            # get a break-off. A far planner/AI melee point is still an attack, never a move.
            leaves = np.flatnonzero((kind[a:b, s] == O.MOVE) & leaving[a:b]) + a
            if len(leaves):
                first = int(leaves[0])
                # Do not turn a cancelled move (a later attack/hold) into a break-off.
                if np.all(kind[first:b, s] == O.MOVE):
                    phase[first:b, s] = 2
                    end[first:b, s] = b
        # A MOVE first observed after recorded separation is a break-off too. In
        # particular this does not reinterpret an AI melee unit's far point *in*
        # recorded melee: it only starts once that unit actually left in the game.
        # Assign after approaches: the first separated sample is a barrier even if
        # the next running approach begins there. Never skip separation for a new charge.
        for b in np.flatnonzero(edges == -1):
            if leave_m > 0 and b < T and active[b] and kind[b, s] == O.MOVE:
                phase[b, s] = 2
                end[b, s] = b + 1
    return {"kind": kind, "x": x, "z": z, "target": target, "run": run,
            "phase": phase, "end": end, "phase_target": phase_target}


class Replay:
    """Recorded orders with an independent contact clock per unit and batch copy.

    Approach and break-off phases finish on simulated contact/separation, respectively.
    Other orders retain their recorded duration. Routing/dead units and unavailable targets
    release a latch. Legacy rows without phase metadata remain clock-indexed. The wall-clock
    recording end plus grace_s still invokes `after`, even if a phase could not finish.
    One instance belongs to one simulation; decreasing battle time resets its clocks.
    """

    def __init__(self, rows, device="cpu", after=nearest_attack, grace_s=120.0):
        T = max(r["kind"].shape[0] for r in rows)
        N = rows[0]["kind"].shape[1]
        B = len(rows)
        self.length = torch.tensor([r["kind"].shape[0] for r in rows], device=device)
        stack = {}
        self.phased = all("phase" in r for r in rows)
        keys = ("kind", "x", "z", "target", "run")
        if self.phased:
            keys += ("phase", "end", "phase_target")
        for k in keys:
            fill = {"kind": O.HOLD, "target": -1}.get(k, 0)
            arr = np.full((B, T, N), fill, dtype=rows[0][k].dtype)
            for b, r in enumerate(rows):
                arr[b, :r[k].shape[0]] = r[k]
            stack[k] = torch.as_tensor(arr, device=device)
        self.stack = stack
        self.after = after
        self.grace = grace_s
        self.clock = None
        self.last_t = None

    def __call__(self, st):
        sec = st.t.floor().long()
        if self.clock is None:
            self.clock = st.t[:, None].expand(-1, st.N).clone()
            self.last_t = st.t.clone()
        elapsed = st.t - self.last_t
        self.clock = torch.where((elapsed < 0)[:, None], st.t[:, None],
                                 self.clock + elapsed.clamp(min=0)[:, None])
        self.last_t = st.t.clone()
        batch = torch.arange(st.B, device=st.device)[:, None]
        unit = torch.arange(st.N, device=st.device)[None, :]

        def index():
            return torch.minimum(self.clock.floor().long(), (self.length - 1)[:, None]).clamp(min=0)

        idx = index()
        gather = lambda a: a[batch, idx, unit]
        if self.phased:
            # A step crossing a boundary must first resolve the phase it was already in.
            prev = torch.minimum((self.clock - elapsed.clamp(min=0)[:, None]).floor().long(),
                                 (self.length - 1)[:, None]).clamp(min=0)
            p = self.stack["phase"][batch, prev, unit]
            stop = self.stack["end"][batch, prev, unit]
            tg = self.stack["phase_target"][batch, prev, unit]
            u = st.u
            standing = (u["side"] > 0) & (u["men"] > 0) & ~u["gone"] & ~u["r"] & ~u["s"]
            target_ok = (tg >= 0) & standing.gather(1, tg.clamp(min=0))
            complete = ((p == 1) & (u["m"] | ~target_ok)) | ((p == 2) & ~u["m"]) | ~standing
            latched = p > 0
            self.clock = torch.where(latched & complete, stop.float(), self.clock)
            self.clock = torch.where(latched & ~complete,
                                     torch.minimum(self.clock, stop.float() - 0.001), self.clock)
            idx = index()
        o = O.Orders(kind=gather(self.stack["kind"]), x=gather(self.stack["x"]), z=gather(self.stack["z"]),
                     target=gather(self.stack["target"]), run=gather(self.stack["run"]))
        if self.phased:
            tg = gather(self.stack["phase_target"])
            valid = (tg >= 0) & standing.gather(1, tg.clamp(min=0))
            # A delayed running MOVE has reached the recorded opponent's old point.
            # Follow that opponent until actual contact instead of standing at that point.
            waiting = (gather(self.stack["phase"]) == 1) & (self.clock >= gather(self.stack["end"]) - 0.0011)
            o.kind = torch.where(waiting & valid, O.ATTACK, o.kind)
            o.target = torch.where((o.kind == O.ATTACK) & valid, tg, o.target)
        over = (sec >= self.length + self.grace)[:, None].expand(-1, st.N)
        if bool(over.any()):
            o = O.merge(o, self.after(st), over)
        return o

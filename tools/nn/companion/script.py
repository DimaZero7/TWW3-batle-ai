"""A scripted opponent of the simulator (tools/nn/train/opponents.py SCRIPTS: ai_like) commands the enemy side
(side 2) of a real battle through the bridge, as the network commands ours (docs/en/apps/bridge.md "The enemy
under a script"; nn-arena --enemy-ai ai_like). Torch: the script runs on the simulator's state.

    the bridge's second instance  -> tww3_bai_nn_state_enemy.json   every decision, the same moment as ours
    the companion (this)          -> tww3_bai_nn_orders_enemy.txt   the answer, the orders file's format

The script sees what it sees in the simulator: everything (both sides, every field, the network's orders in
force). Each decision lays the game's state over a simulator state built once a battle (tools/nn/sim/scenario.py
build: the same slots, unit passports, widths and lords as the twin's), with what the simulator keeps and the
game does not show made the simulator's way:

    r          routing or shattered (the simulator's `r` counts both)
    target     the enemy fought or shot at now (exchange.engaged_targets, both sides)
    order_*, ox, oz   the order in force, the simulator's point (exchange.order_points): the script's own
               orders for its side, the network's orders in force before this decision for ours
    vx, vz     the speed between the last two states
    contact_s  seconds in melee since the contact began, through gaps shorter than contact.reset_s (battle.py)
    rout_s     seconds since the rout began (morale.py)
    gone       men alive but no position (off the map)

The lord's active abilities fire by the game-AI rule of the simulator (tools/nn/sim/abilities.py apply: the
script's side has `ai`): ready (the bridge's uses, exchange.ability_timers), not switched off, and its trigger
(melee, an enemy within near_m, a friend in range wavering, ready) with friends_min friends in range. The rule's
`losing` trigger (the morale's fight balance) has no reading in the game: never (no ability uses it now).
"""
import math
import time

import numpy as np
import torch

from tools.nn.companion import exchange
from tools.nn.sim import abilities as sim_abilities
from tools.nn.sim import orders as O
from tools.nn.sim import scenario as sim_scenario
from tools.nn.sim.params import load as load_params
from tools.nn.train import opponents

SIDE = 2   # the enemy side in the arena (src/entries/nn_arena.lua)
FLOATS = ("x", "z", "b", "hp", "mp", "ms", "a", "k")       # kept from the last state when not read
BOOLS = ("w", "m", "mv", "f", "fire", "lf", "rf", "bf")


class ScriptSide:
    """The script with what it keeps for a battle: the simulator state, the orders in force."""

    def __init__(self, script="ai_like", side=SIDE, params=None):
        if script not in opponents.SCRIPTS:
            raise ValueError(f"unknown script {script!r}: {sorted(opponents.SCRIPTS)}")
        self.script, self.policy, self.side = script, opponents.SCRIPTS[script], side
        self.params = params or load_params()
        self.table = sim_abilities.table(self.params, "cpu")
        self.ability_keys = sim_abilities.keys(self.params)
        self.batch = None

    # --- a new battle: the simulator's state from the first state document ---
    def start(self, doc):
        layout = {u["n"]: u for u in doc.get("layout") or ()}
        factions = doc.get("factions") or {}
        army = {"attacker": int(doc.get("attacker", 2)),
                "sides": {1: {"faction": factions.get("own"), "units": []},
                          2: {"faction": factions.get("enemy"), "units": []}}}
        for u in doc["units"]:
            spec = layout.get(u["n"])
            lord = spec.get("slot") == "lord" if spec else u["n"].endswith("_lord")
            x, z = _num(u.get("x"), 0.0), _num(u.get("z"), 0.0)
            side = int(u["side"])
            entry = {"key": u["key"], "x": x, "z": z, "b": _num(u.get("b"), 90.0 if side == 1 else 270.0),
                     "width": (spec or {}).get("width"), "general": lord, "name": u["n"]}
            if _num(u.get("men"), 0) > 0:
                entry["men"] = float(u["men"])
            army["sides"][side]["units"].append(entry)
        self.st = sim_scenario.build([army], self.params, "cpu")
        sim_abilities.set_rule(self.st.u, torch.tensor([[self.side == 1, self.side == 2]]))
        slot = sim_scenario.slots(army, self.st.N // 2)
        self.names = [u["n"] for u in doc["units"]]
        self.slot = np.array([slot[n] for n in self.names])
        self.sides = np.array([int(u["side"]) for u in doc["units"]])
        self.name_of = {int(s): n for n, s in zip(self.names, self.slot)}
        # each unit's ability keys by the simulator's slots (the bridge's abilities_used are by key)
        self.ab = [[int(self.st.u[f"ab{k}"][0, s]) for k in range(sim_abilities.SLOTS)] for s in self.slot]
        self.ab_keys = [[self.ability_keys[a] if a >= 0 else "" for a in row] for row in self.ab]
        self.given, self.points, self.other_points, self.prev = {}, {}, {}, None
        self.batch = doc["batch"]

    # --- the game's state laid over the simulator's ---
    def lay(self, doc, others):
        """The simulator state for this decision (in place); returns the state dict in the names' order."""
        s = exchange.arrays(doc, self.names)
        for own in (1, 2):
            exchange.engaged_targets(s, self.sides, own=own)
        self.points = exchange.order_points(s, self.names, self.sides, self.given, self.points, own=self.side)
        self.other_points = exchange.order_points(s, self.names, self.sides, others or {}, self.other_points,
                                                  own=3 - self.side)
        u, sl = self.st.u, torch.as_tensor(self.slot)
        t = float(s["t"][0])
        dt = t - self.prev[2] if self.prev is not None else 0.0
        old_x, old_z = u["x"][0, sl].clone(), u["z"][0, sl].clone()
        for k in FLOATS + ("ox", "oz"):
            v = torch.as_tensor(s[k][0], dtype=torch.float32)
            u[k][0, sl] = torch.where(torch.isfinite(v), v, u[k][0, sl])
        men = torch.as_tensor(s["men"][0], dtype=torch.float32)
        u["men"][0, sl] = men
        seen = torch.as_tensor(np.isfinite(s["x"][0]) & np.isfinite(s["z"][0]))
        u["gone"][0, sl] = (men > 0) & ~seen
        u["s"][0, sl] = torch.as_tensor(s["s"][0])
        u["r"][0, sl] = torch.as_tensor(s["r"][0] | s["s"][0])
        for k in BOOLS:
            u[k][0, sl] = torch.as_tensor(s[k][0])
        u["fat"][0, sl] = torch.where(torch.isfinite(torch.as_tensor(s["fat"][0], dtype=torch.float32)),
                                      torch.as_tensor(s["fat"][0], dtype=torch.float32), u["fat"][0, sl])
        to_slot = np.append(self.slot, -1)                    # names index -> slot; -1 stays -1
        u["target"][0, sl] = torch.as_tensor(to_slot[s["target"][0]])
        u["order_kind"][0, sl] = torch.as_tensor(s["order_kind"][0])
        u["order_target"][0, sl] = torch.as_tensor(to_slot[s["order_target"][0]])
        if dt > 0:
            u["vx"][0, sl] = torch.where(seen, (u["x"][0, sl] - old_x) / dt, torch.zeros_like(old_x))
            u["vz"][0, sl] = torch.where(seen, (u["z"][0, sl] - old_z) / dt, torch.zeros_like(old_z))
            # the melee clock and the rout's (battle.py, morale.py) by the game's flags
            reset_s = float(self.params.sim["contact"].get("reset_s", 0.0))
            m, r = u["m"][0, sl], u["r"][0, sl]
            out_s = torch.where(m, torch.zeros_like(u["out_s"][0, sl]), u["out_s"][0, sl] + dt)
            u["out_s"][0, sl] = out_s
            u["contact_s"][0, sl] = torch.where(m, u["contact_s"][0, sl] + dt, torch.where(
                out_s < reset_s, u["contact_s"][0, sl], torch.zeros_like(out_s)))
            u["rout_s"][0, sl] = torch.where(r, u["rout_s"][0, sl] + dt, torch.zeros_like(out_s))
        self.st.t[0] = t
        self.st.attacker[0] = int(doc.get("attacker", self.st.attacker[0]))
        self.prev = (None, None, t)
        return s

    def standing(self):
        """[names] standing: men, not routing or shattered, on the map (the script's `standing`)."""
        u, sl = self.st.u, torch.as_tensor(self.slot)
        return ((u["men"][0, sl] > 0) & ~u["r"][0, sl] & ~u["gone"][0, sl]).numpy()

    # --- the abilities: the simulator's game-AI rule ---
    def abilities(self, doc, s):
        """[{unit, key}]: the abilities the rule fires now for the script's side (module comment)."""
        timers = exchange.ability_timers(doc, self.names, self.ab_keys, own=self.side)
        T, C = self.table, sim_abilities.COL
        near_m = float(self.params.sim["abilities"]["near_m"])
        x, z, men = s["x"][0], s["z"][0], s["men"][0]
        stand = self.standing()
        shaky = (s["w"][0] | s["r"][0] | s["s"][0]) & (men > 0)
        with np.errstate(invalid="ignore"):
            d = np.hypot(x[:, None] - x[None, :], z[:, None] - z[None, :])
        d = np.where(np.isfinite(d), d, np.inf)
        same = self.sides[:, None] == self.sides[None, :]
        eye = np.eye(len(self.names), dtype=bool)
        foe_near = (~same & stand[None, :] & (d <= near_m)).any(1)
        off_now = {"out_of_melee": ~s["m"][0], "morale_is_lower_than_half_of_base_morale": s["mp"][0] < 0.5,
                   "morale_is_higher_than_wavering": ~(s["w"][0] | s["r"][0]), "health_below_50%_base": s["hp"][0] < 0.5}
        out = []
        for i, name in enumerate(self.names):
            if self.sides[i] != self.side or not stand[i]:
                continue
            for k, a in enumerate(self.ab[i]):
                if a < 0:
                    continue
                r = T[a]
                c = lambda col: float(r[C[col]])
                if c("passive") > 0 or c("auto") > 0:
                    continue
                if timers[f"ab{k}_on"][0, i] > 0 or timers[f"ab{k}_cd"][0, i] > 0:
                    continue
                if any(c(f"off_{f}") > 0 and bool(off_now[f][i]) for f in sim_abilities.OFF):
                    continue
                friends = same[i] & ~eye[i] & (men > 0) & (d[i] <= c("range_m"))
                trig = int(c("trigger"))
                want = {1: bool(s["m"][0, i]), 2: bool(foe_near[i]), 3: bool((friends & shaky).any()),
                        4: False, 5: True}.get(trig, False)
                if want and (friends & stand).sum() >= c("friends_min"):
                    out.append({"unit": name, "key": self.ab_keys[i][k]})
        return out

    # --- one decision ---
    @torch.no_grad()
    def decide(self, doc, others=None):
        """-> (orders list, think ms, abilities to use [{unit, key}]) for one state document of the
        enemy's bridge. others: the network's orders in force ({name: order}, exchange's dicts)."""
        t0 = time.perf_counter()
        if self.batch != doc["batch"]:
            self.start(doc)
        s = self.lay(doc, others)
        o = self.policy(self.st)
        sl = torch.as_tensor(self.slot)
        kind, x, z = o.kind[0, sl].numpy(), o.x[0, sl].numpy(), o.z[0, sl].numpy()
        tg, run = o.target[0, sl].numpy(), o.run[0, sl].numpy()
        tgt = np.array([self.names.index(self.name_of[int(j)]) if int(j) in self.name_of else -1 for j in tg])
        out = exchange.orders_list(self.names, self.sides, kind, x, z, tgt, run, own=self.side)
        stand = self.standing()
        ctrl = {n for n, ok, sd in zip(self.names, stand, self.sides) if ok and sd == self.side}
        for order in out:
            if order["unit"] not in ctrl:
                order["out"] = True
        exchange.remember_orders(self.given, out, ctrl)
        uses = self.abilities(doc, s)
        return out, (time.perf_counter() - t0) * 1000, uses


def _num(v, default):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else default

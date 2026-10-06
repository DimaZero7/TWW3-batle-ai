"""The simulator's state layout: the single source of truth (docs/en/training/simulator.md).

A batch of B battles, N unit slots each. A slot is one unit of either side; `side` [B, N] says
whose (1 or 2, 0 = an empty slot). Layout the simulator builds: side 1 in slots [0, H), side 2
in [H, 2H), H = N // 2; a side's lord (if any) is its first slot. Everything goes by `side`, so
other layouts work too; only own_first() assumes this one.

Three groups of per-unit tensors, all [B, N]:

* OBSERVED - the fields and names of the recorded `nn_sample` rows (tools/nn/gamedata.py,
  docs/en/training/README.md "What a recording holds"), so recorded battles and the simulator
  feed the network the same way. `target` is the recording's `t` (current target, a slot index,
  -1 none; as gamedata.Battle.target), `fat` the fatigue state as a number (FATIGUE_LEVELS)
  instead of its string; the battle time is `t` [B] (s), as gamedata.Battle.t.
* STATIC - the unit passport (config/nn/units.json) and the unit's place, fixed for the battle.
* INTERNAL - what the simulator keeps for itself (morale points, fatigue points, timers).

Names are importable without torch; the tensors need it.
"""
from dataclasses import dataclass, field

try:
    import torch
except ImportError:          # the layout (names) is importable without torch
    torch = None

# Fatigue states in order, as the game names them (`fat` in recordings; `fat` here is the index).
FATIGUE_LEVELS = ("threshold_fresh", "threshold_active", "threshold_winded", "threshold_tired",
                  "threshold_very_tired", "threshold_exhausted")
# MoraleState (`ms`) as recorded: 1 eager ... 5 wavering, 6 routing, 7 shattered.
MORALE_STATES = {1: "eager", 2: "confident", 3: "steady", 4: "shaken", 5: "wavering", 6: "routing",
                 7: "shattered"}

# name -> (dtype, meaning). dtype: "f" float32, "b" bool, "i" int64.
OBSERVED = {
    "x": ("f", "position east, m (map centre 0)"),
    "z": ("f", "position north, m"),
    "b": ("f", "bearing, degrees: facing (sin b, cos b) in (x, z); 90 = east"),
    "men": ("f", "men alive"),
    "hp": ("f", "share of health left, 0-1"),
    "mp": ("f", "MoralePercent: morale points / leadership; routs at 0; may be below 0 or above 1"),
    "ms": ("f", "MoraleState 1-7 (MORALE_STATES)"),
    "r": ("b", "routing (shattered units too)"),
    "s": ("b", "shattered"),
    "w": ("b", "wavering"),
    "m": ("b", "in melee"),
    "mv": ("b", "moving"),
    "f": ("b", "running"),
    "a": ("f", "projectiles left, whole unit"),
    "fire": ("b", "firing missiles"),
    "target": ("i", "current target: slot of the enemy fought or shot at, -1 none (recording's t)"),
    "fat": ("f", "fatigue state 0-5 (FATIGUE_LEVELS)"),
    "k": ("f", "kills: enemy men killed"),
    "ox": ("f", "the order's point x, m (own position when holding)"),
    "oz": ("f", "the order's point z, m"),
    "lf": ("b", "an enemy threatens the left flank"),
    "rf": ("b", "an enemy threatens the right flank"),
    "bf": ("b", "an enemy threatens the rear"),
    "vis": ("b", "visible to the other side"),
}

STATIC = {
    "side": ("i", "1 or 2; 0 = empty slot"),
    "lord": ("b", "the side's general"),
    "men0": ("f", "men at the start"),
    "hp_man": ("f", "health of a man"),
    "hp0": ("f", "health of the unit at the start"),
    "mass": ("f", "a man's mass"),
    "radius": ("f", "a man's radius, m"),
    "width": ("f", "frontage ordered at the start, m (files = floor(width / sp_h))"),
    "sp_h": ("f", "a man's place across the front, m (the formation template's close spacing, the database)"),
    "sp_v": ("f", "a man's place between ranks, m (the same)"),
    "walk": ("f", "walk speed, m/s"),
    "run": ("f", "run speed, m/s"),
    "charge_speed": ("f", "charge speed, m/s"),
    "charge_dist": ("f", "charge distance, m: under an attack order the unit closes the last this many metres to its "
                         "target at its charge speed (the database's battle_entities: 30 infantry, 35 lords)"),
    "charge_pose": ("f", "charge pose distance, m: under an attack order the unit takes its charge pose this far from "
                         "its target (the database's battle_entities: 25 infantry, 30 lords; the charge's morale)"),
    "accel": ("f", "acceleration, m/s2"),
    "turn": ("f", "a man's turn rate, deg/s (the database's battle_entities turn_rate: 120 infantry and the General, "
                  "180 archers and the Warlord, 240 militia): a unit on the move turns at it (movement.py), a lone man "
                  "standing too"),
    "decel": ("f", "deceleration, m/s2"),
    "attack": ("f", "melee attack"),
    "defence": ("f", "melee defence"),
    "charge_bonus": ("f", "charge bonus"),
    "damage": ("f", "melee weapon damage (base)"),
    "ap_damage": ("f", "melee weapon armour-piercing damage"),
    "bonus_v_large": ("f", "melee bonus against large"),
    "bonus_v_inf": ("f", "melee bonus against infantry"),
    "interval": ("f", "time between a man's blows, s"),
    "splash": ("f", "targets per blow (hits several)"),
    "armour": ("f", "armour"),
    "shield": ("f", "shield: chance to block a projectile from the front, 0-1"),
    "leadership": ("f", "leadership"),
    "resist_missile": ("f", "missile resistance, 0-1"),
    "resist_physical": ("f", "physical resistance, 0-1"),
    "large": ("b", "size class above small"),
    "expendable": ("b", "attribute expendable: its rout does not scare others"),
    "encourages": ("b", "attribute encourages (lords)"),
    "reflect": ("b", "attribute charge_reflection: braced, it meets a frontal infantry charge as a charge"),
    "unbreakable": ("b", "attribute unbreakable: never loses leadership, never wavers or routs"),
    "fire_move": ("b", "attribute mounted_fire_move: shoots while moving (fire whilst moving)"),
    "fatigue_immune": ("b", "attribute fatigue_immune (Perfect Vigour): gains no fatigue"),
    "direct": ("b", "direct (flat, trajectory low) fire: holds fire where friends block the line (missile.py)"),
    "ammo0": ("f", "projectiles of the whole unit at the start"),
    "range": ("f", "missile range, m (0 = no missile)"),
    "reload": ("f", "reload, s (measured where known, config/nn/sim.json)"),
    "m_damage": ("f", "projectile damage (base)"),
    "m_ap": ("f", "projectile armour-piercing damage"),
    "hit_rate": ("f", "projectile hit rate at the edge of range (config/nn/sim.json; missile.accuracy units 'rates' and "
                      "a lone man in melee)"),
    "arc": ("f", "fire arc each side of the facing, degrees: a man fires only at a target centre within it seen from "
                 "where he stands (the database's battle_entities fire arc / 2: 30, militia 35)"),
    "acc": ("f", "accuracy, 0-100: land_units accuracy + the projectile's marksmanship (the spread, missile.py)"),
    "cal_area": ("f", "the projectile's calibration area, m (the spread at its calibration distance)"),
    "cal_dist": ("f", "the projectile's calibration distance, m"),
    "muzzle_v": ("f", "the projectile's muzzle velocity, m/s (the angle it comes down at)"),
    "height": ("f", "a man's height, m (the shadow a falling projectile hits)"),
    "small_arms": ("b", "small-arms projectile (arrow, musket, sling, javelin, axe): shields block it (missile.py)"),
    "aim_s": ("f", "first shot after halting, s (config/nn/sim.json)"),
    "friendly_fire": ("f", "share of its hits aimed at a unit in melee that land on its own side (config/nn/sim.json)"),
    "morale_bonus": ("f", "morale points at the start beyond leadership (config/nn/sim.json)"),
    "rout_death_s": ("f", "a lord of this faction who routs counts as killed this long into his rout (the "
                          "faction's lords crumble: Vampire Counts, config/nn/sim.json morale.lord_fall); 0 never"),
    "cost": ("f", "multiplayer cost"),
    "cp_fixed": ("f", "combat potential that does not run out: the database's main_units melee_cp + its abilities' "
                      "additional melee and missile cp (the army-destruction strength, morale.army_collapse)"),
    "cp_missile": ("f", "combat potential of its missiles (main_units missile_cp): counts by the ammunition left"),
    "ai": ("b", "played by the game's AI: its lord fires his active abilities by the AI's rule (the network's "
                "side by order: tools/nn/sim/orders.py ability)"),
    "ab0": ("i", "the unit's first ability slot: index in the ability passports (tools/nn/sim/abilities.py keys), "
                 "-1 none"),
    "ab1": ("i", "its second ability (-1 none)"),
    "ab2": ("i", "its third ability (-1 none)"),
    "fx": ("i", "its innate effects: bitmask over config/nn/effects.json order (tools/nn/sim/effects.py)"),
    "fxt0": ("i", "its first timed effect (index in that order, -1 none)"),
    "fxt1": ("i", "its second timed effect (-1 none)"),
    "tag": ("i", "a drill's mark of the unit (tools/nn/train/drills: 1 our unit of the drill's situation, 2 the "
                 "enemy's, inserted into an otherwise normal battle; 0 every other unit); set by scenario.build from "
                 "the description's \"tag\"; the simulator and the observation never read it (only the drill's scripts)"),
}

INTERNAL = {
    "vx": ("f", "velocity x, m/s"),
    "vz": ("f", "velocity z, m/s"),
    "hp_abs": ("f", "health left, HP"),
    "morale": ("f", "morale points"),
    "fatigue": ("f", "fatigue points"),
    "recent": ("f", "HP lost recently (the last morale.casualties_s seconds: recent casualties)"),
    "extended": ("f", "HP lost over the last morale.extended_s seconds (extended casualties)"),
    "contact_s": ("f", "seconds in melee since the contact began (it goes on through gaps shorter than "
                       "contact.reset_s out of contact)"),
    "out_s": ("f", "seconds out of contact since the last contact (contact.reset_s)"),
    "charge": ("f", "the charge left, 0-1: 1 at the charge's first blow, then down by its own clock over "
                    "charge_decay_duration (13 s), in contact or not"),
    "runup": ("f", "run-up, m: run at charge.min_speed_share of the run speed or faster since the last stop, "
                   "walk or contact (charge.min_runup_m)"),
    "ran_in": ("b", "it came into its present fight moving (or charging), not standing: set when its melee clock "
                    "starts, cleared when the clock resets (battle.py: the first strike; melee.py: men gather round "
                    "a lord only when he ran in)"),
    "aim": ("f", "seconds standing still able to shoot"),
    "aim_tgt": ("i", "the enemy it aimed at last step (-1 none): a change makes it wait missile.retarget_s"),
    "turn_on": ("b", "a standing shooter turning to its target (missile.stand_fire_arc_deg starts it, "
                     "missile.turn_done_deg ends it)"),
    "leave_s": ("f", "seconds leaving melee while still touching an enemy (contact.pin_s)"),
    "exit_s": ("f", "seconds since it began to leave melee (a withdraw or a far move while in contact), in contact or "
                    "not, while it keeps leaving: at the database's melee_breakoff_secs still in contact it drops the "
                    "order (battle.py, contact.breakoff)"),
    "low_s": ("f", "seconds since its health first fell below a quarter (0 before; never back: the Wounds of a single "
                   "entity come on its initial cooldown after that and stay to the end, effects.py)"),
    "shots": ("f", "fractional shots carried to the next step"),
    "unready": ("f", "share of the men still reloading, 0-1 (0: all loaded, the start; missile.py volley)"),
    "rout_count": ("f", "times routed"),
    "rout_s": ("f", "seconds since the rout began"),
    "rally_s": ("f", "seconds since the last rally"),
    "flank_hit": ("f", "worst direction attacked from now: 0 front, 1 flank, 2 rear"),
    "under_fire_s": ("f", "seconds since last hit by a projectile"),
    "lost_worst": ("f", "the worst share of the unit lost so far (tools/nn/train/reward.py track: health, a rout's "
                        "share, out whole), kept by the reward's caller, not the simulator: a rally gives nothing back"),
    "dealt": ("f", "HP dealt in melee recently (decaying)"),
    "taken": ("f", "HP taken in melee recently (decaying)"),
    "shot_dealt": ("f", "HP dealt by its missiles to enemies recently (decaying like dealt: the morale's fight balance)"),
    "shot_taken": ("f", "HP taken from enemy missiles recently (decaying like taken: the morale's fight balance)"),
    "chm_s": ("f", "seconds left of the charge's morale block (+15; morale.charge, battle.py)"),
    "chm_n": ("f", "the charge's morale blocks so far in this charge (0 when not charging)"),
    "gone": ("b", "left the map (routed off it)"),
    "dead_s": ("f", "seconds since the unit was killed (health 0; a lord whose faction's routed lord crumbles: "
                    "since rout_death_s into his rout); 0 while not (battle.py, the lord-fall morale)"),
    "gone_s": ("f", "seconds since the unit left the map alive (routed off it); 0 while not (battle.py, the "
                    "lord-fall morale)"),
    "order_kind": ("i", "the order in force (tools/nn/sim/orders.py)"),
    "order_target": ("i", "its target slot"),
    "order_run": ("b", "its run flag"),
    "ab0_on": ("f", "seconds ability 0 stays active"),
    "ab0_cd": ("f", "seconds until ability 0 is ready"),
    "ab1_on": ("f", "seconds ability 1 stays active"),
    "ab1_cd": ("f", "seconds until ability 1 is ready"),
    "ab2_on": ("f", "seconds ability 2 stays active"),
    "ab2_cd": ("f", "seconds until ability 2 is ready"),
    "fxt0_on": ("f", "seconds timed effect 0 stays on"),
    "fxt0_cd": ("f", "seconds until timed effect 0 is ready"),
    "fxt1_on": ("f", "seconds timed effect 1 stays on"),
    "fxt1_cd": ("f", "seconds until timed effect 1 is ready"),
    "fxt0_used": ("b", "timed effect 0 has fired in this battle: its recharge runs only while its recharge context "
                       "holds (the initial one, before the first fire, always; effects.py)"),
    "fxt1_used": ("b", "timed effect 1 has fired in this battle"),
    "cmb": ("f", "the morale's fight balance points of the last step (morale.combat_points: < 0 losing, > 0 winning; 0 "
                 "out of a fight or routing): the 'losing its melee' of the effects (effects.py losing_melee)"),
    "snap_s": ("f", "seconds left of an ability laid on it at a friend's cast (passport update_targets false: Stand "
                    "Your Ground keeps on the units in range at the cast wherever they go; abilities.py)"),
    "snap_ab": ("i", "that ability (index in the passports, -1 none)"),
    "fx_on": ("i", "the innate effects on in the last step: bitmask over config/nn/effects.json order"),
}

GROUPS = {"observed": OBSERVED, "static": STATIC, "internal": INTERNAL}

# Per-unit histories, [B, K * N] (K steps of N slots, newest step first): kept 2-D so that every [B, ...]
# copy of a State (narrowing, putting back, restarting a battle from the bank) carries them unchanged.
HISTORY = {
    "lost_hist": "HP lost each step over the last K steps (morale.casualties_window \"sliding\": the recent and "
                 "extended casualty windows; K = tools/nn/sim/morale.history_steps, 0 when the windows decay)",
}


def _dtype(code):
    return {"f": torch.float32, "b": torch.bool, "i": torch.int64}[code]


@dataclass
class State:
    """A batch of battles. u: every per-unit tensor [B, N] by name (OBSERVED, STATIC, INTERNAL) and the
    per-unit histories [B, K * N] (HISTORY);
    t: battle time [B], s; attacker [B] (1 or 2); done [B]; winner [B] (0 none yet, 1 or 2);
    lord_dead_s [B, 2]: seconds since side 1's / side 2's lord died or shattered (-1: alive or none);
    bounds: the map's half-size, m (square, centre 0); keys [B][N] unit keys ("" empty slot)."""
    u: dict
    t: "torch.Tensor"
    attacker: "torch.Tensor"
    done: "torch.Tensor"
    winner: "torch.Tensor"
    lord_dead_s: "torch.Tensor"
    bounds: float = 1020.0
    keys: list = field(default_factory=list)

    @property
    def B(self):
        return self.t.shape[0]

    @property
    def N(self):
        return self.u["side"].shape[1]

    @property
    def device(self):
        return self.t.device

    def __getitem__(self, name):
        return self.u[name]

    def observation(self):
        """The contract with the network (tools/nn/model/observation.py): the OBSERVED tensors
        [B, N], `side` [B, N], `t` [B] (s), the abilities' timers ab{k}_on / ab{k}_cd [B, N] (s:
        active left, until ready; the observation shows them for own units, `on` for seen enemies),
        `fx_on` [B, N]: the innate effects on now (bitmask, tools/nn/sim/effects.py) and `gone` [B, N]
        (left the map: its gold is lost whole, the attacker's progress in the observation)."""
        out = {k: self.u[k] for k in OBSERVED}
        for k in range(3):
            for t in ("on", "cd"):
                out[f"ab{k}_{t}"] = self.u[f"ab{k}_{t}"]
        out["fx_on"] = self.u["fx_on"]
        out["gone"] = self.u["gone"]
        out["side"] = self.u["side"]
        out["t"] = self.t
        return out

    def clone(self):
        return State({k: v.clone() for k, v in self.u.items()}, self.t.clone(), self.attacker.clone(),
                     self.done.clone(), self.winner.clone(), self.lord_dead_s.clone(), self.bounds,
                     [list(r) for r in self.keys])


def empty(B, N, device="cpu", history=0):
    """B battles with N empty slots (side 0); tools/nn/sim/scenario.py fills them. history: steps K of the
    per-unit histories (HISTORY: [B, K * N])."""
    u = {}
    for group in GROUPS.values():
        for name, (code, _) in group.items():
            u[name] = torch.zeros((B, N), dtype=_dtype(code), device=device)
    for name in HISTORY:
        u[name] = torch.zeros((B, history * N), device=device)
    u["target"].fill_(-1)
    u["order_target"].fill_(-1)
    u["aim_tgt"].fill_(-1)
    u["fxt0"].fill_(-1)
    u["fxt1"].fill_(-1)
    u["snap_ab"].fill_(-1)
    u["vis"].fill_(True)
    zeros = torch.zeros(B, device=device)
    return State(u=u, t=zeros.clone(), attacker=torch.ones(B, dtype=torch.int64, device=device),
                 done=torch.zeros(B, dtype=torch.bool, device=device),
                 winner=torch.zeros(B, dtype=torch.int64, device=device),
                 lord_dead_s=torch.full((B, 2), -1.0, device=device), keys=[[""] * N for _ in range(B)])


def own_first(x, side):
    """x [B, N, ...] in the simulator's layout (side 1's half, then side 2's) seen from `side`:
    for side 2 the halves swap, so the side's own units come first."""
    if side == 1:
        return x
    H = x.shape[1] // 2
    return torch.cat([x[:, H:], x[:, :H]], dim=1)


def slot_from_own_first(index, side, N):
    """A slot index in own_first order -> the simulator's slot (-1 stays -1)."""
    if side == 1:
        return index
    H = N // 2
    return torch.where(index < 0, index, (index + H) % N)

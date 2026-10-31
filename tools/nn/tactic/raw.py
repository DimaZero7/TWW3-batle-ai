"""The tactician's raw state of every slot: the unit level's raw row (tools/nn/unit/view.py RAW - imported, not changed)
plus what the tactician's row needs beyond it (design/tactic "Состояние", "Арсенал"), by ONE formula from what the
game's bridge gives - used on a simulator state here and on the bridge's documents by tools/nn/companion/tactic_input.py
(CLAUDE.md "Вход сети один в один"; build/diag_g49/bridge_parity.md, section "тактик"). The starting point is the
draft of the common row (build/diag_g49/row_draft/row.py), widened to the page's 25 effects.

    observe_sim(st, P)        (obs, static): the effects the bridge shows (the phases on the unit, the abilities of
                              others on it, terror, hidden), the buttons' timers, the passport's flags of each slot
    extend(base, obs, static, prev, dt, P)   [B, N, len(RAW)]: view.RAW's columns (base: view.raw on the simulator,
                              unit_input.game_raw in the game) + MORE: the centre's shift speed, each effect's clock (s
                              it has been on; 0 off), the buttons' seconds active / to ready, the buttons owned, the
                              abilities received, the minimum range
    Seer                      one call a second on a simulator state, keeping the previous snapshot

The effects (EFFECTS, the page's table in its order): source
    phase     the phase on the unit (the game: ActiveEffectList's key - the bridge's fx; the simulator: fx_on)
    received  another unit's ability on it (the game: the phase of an ability it does not own; the simulator: ab_rx | fx_rx)
    game      a flag the game gives (terror: CCO IsTerrified - tr; hidden: unit:is_hidden() - hid)
    rule      worked out from the raw state by rule_effects() - the same rule in both places
    wait      the bridge does not see it yet (regeneration cut by fire, under artillery fire, hit by artillery, the
              formation broken, slowed by hits): 0 in both places until the bridge gives it (the simulator knows them,
              but the network must not learn what the game will not show - parity table)
The clock: how long the effect has been seen (the game shows no time left; the prolonging of a poison is seen in
neither); the row turns it into the share of its time left for effects with a duration.
"""
import math
from functools import lru_cache

import numpy as np
import torch

from tools.nn.unit import view as V

# name, source, duration (s; None: no clock), the enemy's row shows it
EFFECTS = (
    ("poison", "phase", 10.0, True),
    ("contaminated", "phase", 10.0, True),
    ("suppressed", "phase", 10.0, True),
    ("shieldbreaker", "phase", 10.0, True),
    ("frenzy", "phase", None, True),
    ("strength_in_numbers", "phase", None, True),
    ("scurry_away", "phase", None, True),
    ("single_entity", "phase", None, True),
    ("penitent", "phase", 20.0, True),
    ("blackpowder", "phase", None, True),
    ("regen_burned", "wait", 10.0, True),
    ("snare_net", "received", None, True),
    ("hold_the_line", "received", None, True),
    ("lord_near", "rule", None, True),
    ("fear_near", "rule", None, True),
    ("terror", "game", 14.0, True),
    ("terror_immune", "rule", 85.0, False),
    ("art_fire", "wait", 30.0, True),
    ("art_hit", "wait", 30.0, True),
    ("charge", "rule", 13.0, True),
    ("braced", "rule", None, True),
    ("broken", "wait", None, True),
    ("slowed", "wait", None, True),
    ("foreign", "received", None, True),
    ("hidden", "game", None, False),
)
E = len(EFFECTS)
EI = {e[0]: i for i, e in enumerate(EFFECTS)}
WAIT = tuple(e[0] for e in EFFECTS if e[1] == "wait")
# the catalogue's key (config/nn/effects.json order; the bridge's phase keys via phases()) of each "phase" effect
PHASE_KEY = {"poison": "wh_main_unit_contact_poison", "contaminated": "wh2_main_unit_contact_plague_monk_censer_morale",
             "suppressed": "wh2_dlc12_unit_contact_suppressive_fire", "shieldbreaker": "wh2_dlc09_unit_contact_shieldbreaker",
             "frenzy": "wh_main_unit_passive_frenzy", "strength_in_numbers": "wh2_main_unit_passive_strength_in_numbers",
             "scurry_away": "wh2_main_unit_passive_scurry_away", "single_entity": "wh3_main_unit_passive_single_entity",
             "penitent": "wh_dlc04_unit_passive_strength_of_the_penitent",
             "blackpowder": "wh3_dlc25_unit_passive_blackpowder_discipline"}
RECEIVED_KEY = {"snare_net": "wh2_main_unit_passive_snare_net", "hold_the_line": "wh_main_lord_passive_hold_the_line"}
# the pool's buttons (design/tactic "Арсенал"): actives, the General's passive aura, the spells
ARSENAL = ("wh2_main_character_abilities_verminous_valour", "wh_main_character_abilities_deadly_onslaught",
           "wh_main_character_abilities_rally", "wh_main_character_abilities_foe_seeker",
           "wh_main_character_abilities_stand_your_ground", "wh_main_lord_passive_hold_the_line",
           "wh2_main_spell_ruin_warp_lightning", "wh_main_spell_metal_searing_doom")
A = len(ARSENAL)
MORE = (("spd",) + tuple(f"e_{e[0]}" for e in EFFECTS) + tuple(f"a_on{a}" for a in range(A))
        + tuple(f"a_cd{a}" for a in range(A)) + ("a_has", "rx", "min_range"))
RAW = V.RAW + MORE
I = {k: i for i, k in enumerate(RAW)}
BASE = len(V.RAW)
EPS = 1e-3             # an effect's clock the snapshot it is first seen on
LORD_NEAR_M = 70.0     # the lord's aura (encourages): +4 leadership within 70 m (effects.json encourages)
FEAR_M = 20.0          # fear: -8 to the enemies within 20 m of its edge (fear_effect_range)
TERROR_IMMUNE_S = 85.0
STATIC_MORE = ("encourages", "fear", "brace_ok", "min_range")


@lru_cache(maxsize=1)
def catalogues():
    from tools.nn.model import abilities as MA
    from tools.nn.model import effects as ME
    return ME.load(), MA.load(), MA.load_spells()


def phases(effect):
    """The bridge's phase keys that show the "phase" effect `effect` on a unit: its ability passport's phases that carry
    stats (Blackpowder Discipline's _ii: the 10 s are over), else all its phases, else its own key."""
    key = PHASE_KEY[effect]
    _, ab, _ = catalogues()
    p = ab.get(key) or {}
    with_stats = sorted({e.get("phase") for e in p.get("effects") or () if e.get("phase")})
    return set(with_stats or p.get("phases") or (key,))


@lru_cache(maxsize=1)
def ability_keys():
    """The abilities' keys in the received bitmask's order (model/abilities.keys: sorted - the simulator's ab_rx)."""
    from tools.nn.model import abilities as MA
    return tuple(MA.keys())


@lru_cache(maxsize=1)
def spell_mask():
    """[A] bool: the buttons that are spells (a cast and its recharge; no 'active' time shown)."""
    _, _, sp = catalogues()
    return np.array([k in sp for k in ARSENAL], bool)


@lru_cache(maxsize=1)
def arsenal_times():
    """([A] active_s, [A] active_s + recharge_s) of each button (abilities.json; a passive: 0, 0; a spell: 0, its
    recharge)."""
    _, ab, sp = catalogues()
    act, full = np.zeros(A, np.float32), np.zeros(A, np.float32)
    for a, k in enumerate(ARSENAL):
        p = ab.get(k) or sp.get(k) or {}
        if p.get("passive"):
            continue
        on, rc = max(float(p.get("active_s") or 0), 0.0), max(float(p.get("recharge_s") or 0), 0.0)
        act[a], full[a] = (0.0 if k in sp else on), on + rc
    return act, full


def received_split(rx):
    """(snare_net, hold_the_line, foreign) [..] bool of a received bitmask rx [..] (int64)."""
    keys = ability_keys()
    out = []
    for name in ("snare_net", "hold_the_line"):
        k = RECEIVED_KEY[name]
        out.append(((rx >> keys.index(k)) & 1) > 0 if k in keys else torch.zeros_like(rx, dtype=torch.bool))
    other = sum((1 << r) for r, k in enumerate(keys) if k not in RECEIVED_KEY.values())
    out.append((rx & other) != 0)
    return tuple(out)


# ---------------------------------------------------------------------------------------------- the simulator's side
@lru_cache(maxsize=8)
def _sim_luts(ab_keys, sp_keys):
    """The simulator's ability / spell table index -> ARSENAL index (-1: not a button)."""
    return ([ARSENAL.index(k) if k in ARSENAL else -1 for k in ab_keys],
            [ARSENAL.index(k) if k in ARSENAL else -1 for k in sp_keys])


def observe_sim(st, P):
    """(obs, static) of a simulator state as the game's bridge gives it: obs fx [B, N, E] bool (the phase / received /
    game effects; the rule and wait ones False), rx [B, N] int64 (the received abilities' bitmask), a_on / a_cd
    [B, N, A] (s), a_has [B, N] int64; static: the passport's flags (encourages, fear, brace_ok) and min_range."""
    from tools.nn.model import effects as ME
    from tools.nn.sim import abilities as SA
    from tools.nn.sim import magic as SM
    u = st.u
    dev = u["x"].device
    fx = torch.zeros(u["x"].shape + (E,), dtype=torch.bool, device=dev)
    if "fx_on" in u:
        order = ME.keys()
        for name, src, _, _ in EFFECTS:
            if src == "phase" and PHASE_KEY[name] in order:
                fx[..., EI[name]] = ((u["fx_on"] >> order.index(PHASE_KEY[name])) & 1) > 0
    rx = torch.zeros_like(u["side"]).long()
    if "ab_rx" in u:
        rx = (u["ab_rx"] | u.get("fx_rx", torch.zeros_like(u["ab_rx"]))).long()
    sn, htl, fo = received_split(rx)
    fx[..., EI["snare_net"]], fx[..., EI["hold_the_line"]], fx[..., EI["foreign"]] = sn, htl, fo
    if "ter_s" in u:
        fx[..., EI["terror"]] = u["ter_s"] > 0
    if "hid" in u:
        fx[..., EI["hidden"]] = u["hid"]
    alive = (u["side"] > 0) & (u["men"] > 0) & ~u["gone"]
    fx = fx & alive[..., None]
    lut_a, lut_s = _sim_luts(tuple(SA.keys(P)), tuple(SM.keys(P)))
    la = torch.tensor(lut_a + [-1], device=dev)
    ls = torch.tensor(lut_s + [-1], device=dev)
    on = torch.zeros(u["x"].shape + (A,), device=dev)
    cd = torch.zeros_like(on)
    has = torch.zeros_like(u["side"]).long()
    k = 0
    while f"ab{k}" in u:
        v = la[u[f"ab{k}"].long()]                    # (-1 -> the last entry: -1)
        ok = v >= 0
        oh = torch.nn.functional.one_hot(v.clamp(min=0), A).bool() & ok[..., None]
        on = torch.where(oh, u[f"ab{k}_on"][..., None], on)
        cd = torch.where(oh, u[f"ab{k}_cd"][..., None], cd)
        has = has | torch.where(ok, 1 << v.clamp(min=0), torch.zeros_like(v))
        k += 1
    k = 0
    while f"sp{k}" in u:
        v = ls[u[f"sp{k}"].long()]
        ok = v >= 0
        oh = torch.nn.functional.one_hot(v.clamp(min=0), A).bool() & ok[..., None]
        if f"sp{k}_cd" in u:
            cd = torch.where(oh, u[f"sp{k}_cd"][..., None], cd)
        has = has | torch.where(ok, 1 << v.clamp(min=0), torch.zeros_like(v))
        k += 1
    obs = {"fx": fx, "rx": torch.where(alive, rx, torch.zeros_like(rx)), "a_on": on, "a_cd": cd, "a_has": has}
    static = {k: (u[k] if k in u else torch.zeros_like(u["x"])) for k in STATIC_MORE}
    return obs, static


# ------------------------------------------------------------------------------------------------- the shared formula
def shift_speed(base, prev, dt):
    """[B, N] the centre's shift since the previous snapshot / dt (m/s; 0 without one or for a unit not alive in
    either) - from the positions alone, the same in the simulator and the game."""
    if prev is None:
        return torch.zeros_like(base[..., V.I["x"]])
    ok = (base[..., V.I["alive"]] > 0) & (prev[..., I["alive"]] > 0)
    d = torch.sqrt((base[..., V.I["x"]] - prev[..., I["x"]]) ** 2 + (base[..., V.I["z"]] - prev[..., I["z"]]) ** 2)
    return torch.where(ok, d / dt, torch.zeros_like(d))


def rule_effects(raw, static, prev, dt, params):
    """([B, N, E] bool on, [B, N, E] bool restart): the "rule" effects worked out from the raw state now (raw [B, N,
    RAW] with the base columns, spd and the terror's clock filled) and the previous snapshot, the same rule for the
    simulator and the game:
      lord_near      a living own unit that encourages (the lord) within LORD_NEAR_M, centre to centre
      fear_near      a living enemy seen by its side that causes fear within FEAR_M of its formation's edge
      charge         came into melee (m now, not before) from a run - the centre's shift over this or the previous
                     second at least charge.min_speed_share of its run speed - and for the charge's decay after
      braced         can brace, alive, not routing, not moving now and before
      terror_immune  TERROR_IMMUNE_S after a flight from terror ended."""
    from tools.nn.sim import geometry
    B, N = raw.shape[:2]
    out = torch.zeros((B, N, E), dtype=torch.bool, device=raw.device)
    alive = raw[..., I["alive"]] > 0
    side = raw[..., I["side"]]
    x, z = raw[..., I["x"]], raw[..., I["z"]]
    eye = torch.eye(N, dtype=torch.bool, device=raw.device)[None]
    same = (side[:, :, None] == side[:, None, :]) & ~eye
    d = torch.sqrt((x[:, :, None] - x[:, None, :]) ** 2 + (z[:, :, None] - z[:, None, :]) ** 2)
    enc = static["encourages"].bool() & alive
    out[..., EI["lord_near"]] = alive & (same & enc[:, None, :] & (d <= LORD_NEAR_M)).any(2)
    gap = geometry.rect_gap(x, z, raw[..., I["b"]], raw[..., I["front"]], raw[..., I["depth"]])
    foe = (side[:, :, None] != side[:, None, :]) & (side[:, None, :] > 0)
    fear = static["fear"].bool() & alive & (raw[..., I["vis"]] > 0)
    out[..., EI["fear_near"]] = alive & (foe & fear[:, None, :] & (gap <= FEAR_M)).any(2)
    mv, m, r = raw[..., I["mv"]] > 0, raw[..., I["m"]] > 0, raw[..., I["r"]] > 0
    restart = torch.zeros_like(out)
    if prev is None:
        out[..., EI["braced"]] = static["brace_ok"].bool() & alive & ~r & ~mv
        return out, restart
    decay = float(params.battle.get("charge_decay_duration", 13.0))
    share = float(params.sim["charge"]["min_speed_share"])
    pm, pmv = prev[..., I["m"]] > 0, prev[..., I["mv"]] > 0
    pv = torch.maximum(raw[..., I["spd"]], prev[..., I["spd"]])
    start = alive & m & ~pm & (pv >= share * raw[..., I["run"]]) & (pv > 0.1)
    pc = prev[..., I["e_charge"]]
    out[..., EI["charge"]] = start | (alive & (pc > 0) & (pc + dt < decay))
    out[..., EI["braced"]] = static["brace_ok"].bool() & alive & ~r & ~mv & ~pmv
    pt, pi = prev[..., I["e_terror"]], prev[..., I["e_terror_immune"]]
    ended = (pt > 0) & (raw[..., I["e_terror"]] <= 0)
    out[..., EI["terror_immune"]] = alive & (ended | ((pi > 0) & (pi + dt < TERROR_IMMUNE_S)))
    restart[..., EI["charge"]] = start
    restart[..., EI["terror_immune"]] = ended
    return out, restart


def clocks(on, prev_clock, dt, restart=None):
    """[B, N, E] how long each effect has been on (s; 0 off): EPS when first seen, + dt while it stays; restart:
    start again at EPS (a new charge, a new immunity)."""
    if prev_clock is None:
        return torch.where(on, torch.full(on.shape, EPS, device=on.device), torch.zeros(on.shape, device=on.device))
    run = torch.where(prev_clock > 0, prev_clock + dt, torch.full_like(prev_clock, EPS))
    if restart is not None:
        run = torch.where(restart, torch.full_like(run, EPS), run)
    return torch.where(on, run, torch.zeros_like(run))


def extend(base, obs, static, prev=None, dt=1.0, params=None):
    """[B, N, len(RAW)]: base [B, N, len(view.RAW)] (view.raw / unit_input.game_raw) + MORE from the observed (obs,
    static: observe_sim / the companion's observe_game) and the previous snapshot prev [B, N, len(RAW)] (dt s before;
    None: the first)."""
    if params is None:
        from tools.nn.sim.params import load
        params = load()
    B, N = base.shape[:2]
    raw = torch.zeros((B, N, len(RAW)), device=base.device)
    raw[..., :BASE] = base
    alive = base[..., V.I["alive"]] > 0
    raw[..., I["spd"]] = shift_speed(base, prev, dt)
    e0 = I["e_" + EFFECTS[0][0]]
    pclk = None if prev is None else prev[..., e0:e0 + E]
    on = obs["fx"].bool() & alive[..., None]
    raw[..., I["e_terror"]] = clocks(on, pclk, dt)[..., EI["terror"]]
    rule, restart = rule_effects(raw, static, prev, dt, params)
    kinds = torch.tensor([e[1] == "rule" for e in EFFECTS], device=base.device)
    wait = torch.tensor([e[1] == "wait" for e in EFFECTS], device=base.device)
    on = torch.where(kinds, rule, on) & ~wait
    raw[..., e0:e0 + E] = clocks(on, pclk, dt, restart)
    spell = torch.tensor(spell_mask(), device=base.device)
    raw[..., I["a_on0"]:I["a_on0"] + A] = torch.where(spell, torch.zeros_like(obs["a_on"]), obs["a_on"]) * alive[..., None]
    raw[..., I["a_cd0"]:I["a_cd0"] + A] = obs["a_cd"] * alive[..., None]
    raw[..., I["a_has"]] = obs["a_has"].float()
    raw[..., I["rx"]] = obs["rx"].float()
    raw[..., I["min_range"]] = static["min_range"].float()
    return raw


class Seer:
    """The tactician's raw state of a simulator battle, one call a decision (the previous snapshot kept)."""

    def __init__(self, params=None):
        from tools.nn.sim.params import load
        self.P = params or load()
        self.prev = None

    def raw(self, st, dt=1.0):
        base = V.raw(st, self.P.sim["formation"]["spacing_m"], missile=self.P.sim["missile"])
        obs, static = observe_sim(st, self.P)
        r = extend(base, obs, static, self.prev, dt, self.P)
        self.prev = r
        return r


def ground_of(raw):
    """{terr, tpx, tpz, tpr} [B, 1] of each battle's map from its raw state (terrain.py's handle; terr -1: flat)."""
    from tools.nn.sim import terrain
    g = {"terr": raw[:, :1, I["terr"]].round().long()}
    for k in terrain.PLACE:
        g[k] = raw[:, :1, I[k]]
    return g


DEG = math.pi / 180

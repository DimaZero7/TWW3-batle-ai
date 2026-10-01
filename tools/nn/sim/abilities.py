"""Lord abilities (docs/en/training/simulator.md): numbers from the game's database
(config/nn/sim.json `abilities`). Only a side played by the game's AI (STATIC `ai`) uses active
abilities: the in-game bridge gives the network no ability orders. Passive ones work for every side.

When the AI fires an ability is an assumption (not measured): `melee` when its lord is in melee,
`near` when a standing enemy is within near_m of him, `waver` when a friend within the ability's
range wavers or routs; then it lasts active_s and is ready again recharge_s after it ends.
Effects: speed (walk, run, charge) and melee damage / charge bonus of the lord himself; morale
points and melee defence for his side's units within range_m (0: himself only).
"""
try:
    import torch
except ImportError:          # slots_of() is used without torch (params)
    torch = None

COLS = ("active_s", "recharge_s", "trigger", "range_m", "speed", "damage", "charge", "leadership", "defence")
TRIGGERS = {"passive": 0, "melee": 1, "near": 2, "waver": 3}
SLOTS = 3


def keys(params):
    return list(params.sim["abilities"]["table"])


def slots_of(params, unit_key):
    """Indices of the unit's abilities in the table (-1: none), SLOTS of them."""
    names = keys(params)
    own = [names.index(k) for k in params.sim["abilities"]["units"].get(unit_key, [])]
    return (own + [-1] * SLOTS)[:SLOTS]


def table(params, device):
    rows = []
    for k, a in params.sim["abilities"]["table"].items():
        rows.append([float(a.get("active_s", -1)), float(a.get("recharge_s", -1)), float(TRIGGERS[a["trigger"]]),
                     float(a.get("range_m", 0)), float(a.get("speed", 1)), float(a.get("damage", 1)),
                     float(a.get("charge", 1)), float(a.get("leadership", 0)), float(a.get("defence", 0))])
    rows.append([0.0] * len(COLS))      # index -1: no ability
    return torch.tensor(rows, device=device)


def apply(u, params, dt, standing, engaged, dist, same_side):
    """Advance the timers, fire the AI's abilities, and lay their effects on u (in place).
    Returns the fields it changed with their old values, for restore()."""
    T = table(params, u["men"].device)
    near_m = float(params.sim["abilities"]["near_m"])
    eye = torch.eye(u["men"].shape[1], dtype=torch.bool, device=u["men"].device)[None]
    foe_near = (~same_side & standing[:, None, :] & (dist <= near_m)).any(2)
    shaky = (u["w"] | u["r"]) & (u["men"] > 0)
    speed = torch.ones_like(u["men"]); dmg = torch.ones_like(speed); chg = torch.ones_like(speed)
    lead = torch.zeros_like(speed); dfn = torch.zeros_like(speed)
    for k in range(SLOTS):
        row = T[u[f"ab{k}"]]                        # [B, N, COLS]; -1 picks the empty last row
        has = u[f"ab{k}"] >= 0
        trig, rng = row[..., 2], row[..., 3]
        on = (u[f"ab{k}_on"] - dt).clamp(min=0)
        cd = (u[f"ab{k}_cd"] - dt).clamp(min=0)
        reach = same_side & ((dist <= rng[:, :, None]) | eye)          # owner i -> unit j
        friend_shaky = (reach & ~eye & shaky[:, None, :]).any(2)
        want = torch.where(trig == 1, engaged, torch.where(trig == 2, foe_near, torch.where(trig == 3, friend_shaky,
                                                                                               torch.zeros_like(engaged))))
        fire = u["ai"] & standing & has & (trig > 0) & (on <= 0) & (cd <= 0) & want
        on = torch.where(fire, row[..., 0], on)
        cd = torch.where(fire, row[..., 0] + row[..., 1], cd)
        u[f"ab{k}_on"], u[f"ab{k}_cd"] = on, cd
        active = has & standing & ((on > 0) | (trig == 0))
        speed = speed * torch.where(active, row[..., 4], 1.0)
        dmg = dmg * torch.where(active, row[..., 5], 1.0)
        chg = chg * torch.where(active, row[..., 6], 1.0)
        give = (reach & active[:, :, None]).float()
        lead = lead + torch.einsum("bij,bi->bj", give, row[..., 7])
        dfn = dfn + torch.einsum("bij,bi->bj", give, row[..., 8])
    old = {k: u[k] for k in ("walk", "run", "charge_speed", "damage", "ap_damage", "charge_bonus", "defence",
                              "morale_bonus")}
    for k in ("walk", "run", "charge_speed"):
        u[k] = u[k] * speed
    u["damage"], u["ap_damage"] = u["damage"] * dmg, u["ap_damage"] * dmg
    u["charge_bonus"] = u["charge_bonus"] * chg
    u["defence"] = u["defence"] + dfn
    u["morale_bonus"] = u["morale_bonus"] + lead
    return old


def restore(u, old):
    u.update(old)

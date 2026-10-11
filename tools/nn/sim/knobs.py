"""Per-battle values of the simulator's uncertain numbers (domain randomisation, tools/nn/train/dr.py; config/nn/dr.json).

A number of config/nn/sim.json that was measured in a few probes, fitted or estimated may get, per battle, a value drawn
inside its range (tools/nn/train/dr.py, training only, off by default). The drawn value of a SCALAR knob lives in the
state as u["k_<name>"] [B, N] (the same on every slot, like the battle id: it travels with the battle through every
narrowing, compaction and restart of the batch); the simulator reads it by get(). Without that field - the default,
and always outside training - get() gives the sim.json number itself, so the code runs exactly as before.

Per-unit knobs (a passport-like field of every unit: aim_s, quiet_s, ...) need nothing here: tools/nn/train/dr.py
changes the field itself at the battle's start, as tools/nn/train/randomise.py does the damage.

SCALARS: knob name -> what it is (the sim.json path of its centre; a "*_scale" knob multiplies a table, centre 1).
"""
PREFIX = "k_"

SCALARS = {
    "front_fill_ranks": "melee.front_fill_ranks",
    "wave_amp": "melee.wave.amp",
    "hold_rate": "contact.hold_rate",
    "pursuit_rate": "contact.pursuit_rate",
    "leave_taken_melee": "contact.leave_taken.melee",
    "pin_s": "contact.pin_s",
    "pin_melee_s": "contact.pin_melee_s",
    "lord_max_attackers": "contact.lord_max_attackers",
    "lord_splash_struck": "contact.lord_splash_struck",
    "min_runup_m": "charge.min_runup_m",
    "strong_ratio": "morale.strong_ratio",
    "combat_ratio_scale": None,          # x morale.combat_ratio slightly / yes / significantly (winning)
    "rally_hazard_scale": None,          # x morale.rally_hazard.p_per_s (every bin)
    "multi_combat": "fatigue.calibration.multi_combat",
    "sk_cone_deg": "skirmish.cone_deg",
    "sk_other_end_per_s": "skirmish.other_end_per_s",
    "sk_new_point_m": "skirmish.new_point_m",
}


def centre(sim, name):
    """The sim.json number a scalar knob spreads around (1.0 for a *_scale knob)."""
    path = SCALARS[name]
    if path is None:
        return 1.0
    x = sim
    for k in path.split("."):
        x = x[k]
    return float(x)


def get(u, name, default, dims=2):
    """The battle's value of knob `name`: u["k_<name>"] as [B, 1] (dims 2: against [B, N]) or [B, 1, 1] (dims 3:
    against [B, N, N]); `default` (the sim.json number, a float) when the battle has none (domain randomisation off)."""
    t = u.get(PREFIX + name)
    if t is None:
        return default
    t = t[:, :1]
    return t[:, :, None] if dims == 3 else t


def full_like(x, v):
    """torch.full_like(x, v) for a float v (the code before), v broadcast to x's shape for a per-battle tensor."""
    import torch
    if isinstance(v, (int, float)):
        return torch.full_like(x, float(v))
    return v.to(x.dtype).expand_as(x)

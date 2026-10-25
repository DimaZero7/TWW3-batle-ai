"""Hiding and spotting (docs/ru/training/simulator.md "Скрытность"; config/nn/sim.json stealth; line g43, 25.10.2026).

The game's rules (the game's database and the community's breakdown of the engine, high confidence unless said):
* hiding is a flag of the whole unit: one man seen, the unit is seen;
* a unit hides while more than share_hide (85 %) of its men stand on ground that hides them and stays hidden while more
  than share_keep (75 %); for a unit with the attribute stalk every ground hides (our maps have no woods, so only stalk
  units ever hide: the Gutter Runners, the Huntsmen; hide_forest units - the rest - never);
* it becomes hidden to the enemy hide_s (5 s) after it began to meet that, unseen; when the share falls below share_keep
  while it is hidden and nobody sees it, it stays hidden grace_s (1.5 s) more;
* an enemy unit sees it within its spotting distance (the database's land_units spot_dist_scrub: 80 m, the Jezzails
  160 m; our open ground is not woods) x the hidden unit's hiding_scalar (0.8 the Gutter Runners, 1.0 the Huntsmen),
  + fresh_bonus_m (20 m) to the spotting distance for the first fresh_s (5 s) after it hid; measured edge to edge of the
  formations (geometry.pairwise gap at the step's start): the Gutter Runners are seen within 64 m (80 m just after they
  hid), by the Jezzails within 128 m;
* its own acts reveal it: firing (at will and on the move; the attribute snipe would keep it hidden - none of our hiding
  units has it) and melee; an attack order does not (reveal.attack_order 0: the probe of 25.10.2026), running does not.
The probe 'скрытность' (25.10.2026, 3 battles) checks the rules: seen at 66.6 / 64.8 / 62.9 m standing / walking /
running (64), the Huntsmen by the Jezzails at 163.5 m (160), firing reveals 0.6 s after it begins, hidden again 5.0 s
after it stops (config/nn/sim.json stealth.why).
What being hidden does (battle.py, the step after): the enemy's shooters do not take it as a target, an attack order on
it is no order (HOLD, as a dead target), the enemy's blows do not fall on it, and the enemy's morale does not count it
(the threats, secure flanks, strong enemies, routers, the rally's nearest enemy); the network's side does not see it
(the state's `vis`: the observation shows its last seen place, tools/nn/model/observation.py), its own side sees it
with the token's `hidden` column. A dead unit is seen (the game shows a destroyed unit).


State: STATIC stalk (the attribute's rule flag, effects.py), spot_m (its own spotting distance), hide_k (hiding_scalar);
INTERNAL hid (hidden now), hid_s (seconds it has met the conditions unseen: the hide clock), hid_t (seconds since it hid:
the fresh bonus), hid_g (seconds of the grace used). Only json without torch here at import.
"""
try:
    import torch
except ImportError:          # the config reader is importable without torch
    torch = None


def cfg(params):
    """sim.json stealth ({} : off)."""
    return params.sim.get("stealth") or {}


def on(params):
    """Stealth acts: sim.json stealth.on and some unit of the passports can hide (the attribute stalk)."""
    c = cfg(params)
    return bool(c.get("on")) and bool(getattr(params, "can_hide", False))      # (Params.can_hide: once, at the load)


def start(u, params):
    """At the battle's start (scenario.build), in place: a unit that can hide starts hidden (stealth.start_hidden; an
    assumption: the deployment's units are hidden from the first second, as the game shows them) - hid on, its clock
    full, the fresh bonus over; vis = not hidden."""
    if not on(params) or "hid" not in u:
        return
    present = u["side"] > 0
    can = present & u["stalk"] & (u["men"] > 0)
    c = cfg(params)
    if c.get("start_hidden"):
        u["hid"] = can.clone()
        u["hid_s"] = torch.where(can, torch.full_like(u["hid_s"], float(c["hide_s"])), torch.zeros_like(u["hid_s"]))
        u["hid_t"] = torch.where(can, torch.full_like(u["hid_t"], float(c["fresh_s"])), torch.zeros_like(u["hid_t"]))
    u["vis"] = present & ~u["hid"]


def seen(u):
    """[B, N] the units the other side may see now (not hidden by stealth): all when the state has no `hid`."""
    return ~u["hid"] if "hid" in u else torch.ones_like(u["side"] > 0)


def cover(u):
    """[B, N] the share of the unit's men on ground that hides it: 1 for stalk (any ground), 0 otherwise (no woods)."""
    return u["stalk"].float()


def update(u, pw, params, dt, fired, engaged, attack):
    """The step's spotting, in place (module doc): fired [B, N] it shot this step, engaged [B, N] in melee, attack [B, N]
    an attack order in force. pw: the step's pairwise geometry (its gap, edge to edge). Sets hid, hid_s, hid_t, hid_g
    and vis (= present and not hidden) for the next step and the observation."""
    if not on(params) or "hid" not in u:
        return
    c = cfg(params)
    rv = c.get("reveal") or {}
    present = u["side"] > 0
    alive = present & (u["men"] > 0) & ~u["gone"]
    can = alive & u["stalk"]
    hid = u["hid"] & can
    # who sees whom: an alive enemy j sees i within j's spotting distance (+ the fresh bonus while i hid lately) x i's
    # hiding_scalar, edge to edge
    fresh = hid & (u["hid_t"] < float(c["fresh_s"]))
    reach = (u["spot_m"][:, None, :] + torch.where(fresh, float(c["fresh_bonus_m"]), 0.0)[:, :, None]) * u["hide_k"][
        :, :, None]                                                     # [B, i hidden, j observer]
    spotted = (pw["enemy"] & alive[:, None, :] & (pw["gap"] <= reach)).any(2)
    acts = torch.zeros_like(can)
    if rv.get("fire", 1):
        acts = acts | fired
    if rv.get("melee", 1):
        acts = acts | engaged
    if rv.get("attack_order", 1):
        acts = acts | attack
    reveal = spotted | acts
    share = cover(u)
    begin = share > float(c["share_hide"])
    keep = share > float(c["share_keep"])
    # not hidden: the clock runs while it meets the conditions unseen; at hide_s it hides
    clock = torch.where(can & ~hid & begin & ~reveal, u["hid_s"] + dt, torch.zeros_like(u["hid_s"]))
    clock = torch.where(hid, u["hid_s"], clock)
    hides = can & ~hid & (clock >= float(c["hide_s"]) - 1e-6)
    # hidden: revealed at once by an act or an enemy's eyes; out of cover it stays hidden grace_s more
    grace = torch.where(hid & ~keep, u["hid_g"] + dt, torch.zeros_like(u["hid_g"]))
    lost = hid & (reveal | (grace > float(c["grace_s"]) + 1e-6))
    now = (hid & ~lost) | hides
    u["hid"] = now
    u["hid_s"] = torch.where(now, clock, torch.where(lost, torch.zeros_like(clock), clock))
    u["hid_t"] = torch.where(hides, torch.zeros_like(u["hid_t"]), torch.where(now, u["hid_t"] + dt,
                                                                               torch.zeros_like(u["hid_t"])))
    u["hid_g"] = torch.where(now, grace, torch.zeros_like(grace))
    u["vis"] = present & ~now

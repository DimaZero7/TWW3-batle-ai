"""Hiding and spotting (docs/ru/training/simulator.md "Скрытность"; config/nn/sim.json stealth; line g43, 25.10.2026).

The game's rules (the game's database and the community's breakdown of the engine, high confidence unless said):
* hiding is a flag of the whole unit: one man seen, the unit is seen;
* a unit hides while more than share_hide (85 %) of its men stand on ground that hides them and stays hidden while more
  than share_keep (75 %); for a unit with the attribute stalk every ground hides; for hide_forest (31 of our 36 units)
  forest ground - each man's own cell of the map's ground layer (ground.py; a map without one, or none: never);
* it becomes hidden to the enemy hide_s (5 s) after it began to meet that, unseen; when the share falls below share_keep
  while it is hidden and nobody sees it, it stays hidden grace_s (1.5 s) more; revealed for less than quick_s (1.5 s) it
  hides again at once (build/diag_g49/terrain/stealth.md);
* an enemy unit, not routing (routers_blind), sees it within its spotting distance - the database's land_units
  spot_dist_tree (60 m) when most of the hidden unit's men stand on forest, else spot_dist_scrub (80 m; the Jezzails
  160 m) - x the hidden unit's hiding_scalar (0.8 the Gutter Runners, the clanrats), + fresh_bonus_m (20 m) while the
  unit is not yet hidden (its hide clock running: fresh_before; 0 - the old reading, the first fresh_s after it hid);
  edge to edge of the formations (geometry.pairwise gap at the step's start): the Gutter Runners are seen within 64 m,
  by the Jezzails within 128 m, clanrats in the woods within 48 m;
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


State: STATIC stalk (the attribute's rule flag, effects.py), hidef (hide_forest), spot_m / spot_tree (its spotting
distances), hide_k (hiding_scalar); INTERNAL hid (hidden now), hid_s (seconds it has met the conditions unseen: the hide
clock), hid_t (seconds since it hid), hid_g (seconds of the grace used), hid_r (the quick re-hide's window left), hid_d
(deployed, not stepped: a hide_forest unit in the woods at the first step starts hidden). Only json without torch here at
import.
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
        if "hid_d" in u:
            # a hide_forest unit deployed in the woods starts hidden too - decided at the first step, once the battle's
            # map is set (ground.py; the same assumption as stalk's)
            u["hid_d"] = present & ~can
    u["vis"] = present & ~u["hid"]


def seen(u):
    """[B, N] the units the other side may see now (not hidden by stealth): all when the state has no `hid`."""
    return ~u["hid"] if "hid" in u else torch.ones_like(u["side"] > 0)


def cover(u, forest=None):
    """[B, N] the share of the unit's men on ground that hides it: 1 for stalk (any ground); for hide_forest the share
    of its men on forest ground (forest [B, N]: ground.shares; None - no ground layer - 0); 0 otherwise."""
    share = u["stalk"].float()
    if forest is not None and "hidef" in u:
        share = torch.maximum(share, torch.where(u["hidef"], forest, torch.zeros_like(forest)))
    return share


def update(u, pw, params, dt, fired, engaged, attack):
    """The step's spotting, in place (module doc): fired [B, N] it shot this step, engaged [B, N] in melee, attack [B, N]
    an attack order in force. pw: the step's pairwise geometry (its gap, edge to edge; its front and depth for the men's
    ground). Sets hid, hid_s, hid_t, hid_g, hid_r and vis (= present and not hidden) for the next step and the
    observation."""
    if not on(params) or "hid" not in u:
        return
    c = cfg(params)
    rv = c.get("reveal") or {}
    present = u["side"] > 0
    alive = present & (u["men"] > 0) & ~u["gone"]
    # the men on forest ground (ground.py: each man's own cell; build/diag_g49/terrain/stealth.md 1) and whether the
    # forest is the ground most of them stand on (the spotting's 'tree' distance)
    forest = mostly = None
    from tools.nn.sim import ground
    if ground.on(u):
        sh = ground.shares(u, pw["front"], pw["depth"], params)
        forest = sh[..., 1]
        mostly = (forest > 0) & (forest >= sh.amax(-1) - 1e-6)
    share = cover(u, forest)
    can = alive & u["stalk"]
    if forest is not None and "hidef" in u:
        can = can | (alive & u["hidef"])
    hid = u["hid"] & can
    # who sees whom: an alive enemy j, not routing (a router observes nothing: stealth.md 1), sees i within j's spotting
    # distance - its spot_dist_tree when most of i's men stand on forest, else spot_dist_scrub - (+ the fresh bonus while
    # i is not yet hidden: its hide clock running, stealth.md 1) x i's hiding_scalar, edge to edge
    fresh = ~hid
    if not c.get("fresh_before"):
        fresh = hid & (u["hid_t"] < float(c["fresh_s"]))           # (the old reading: the first fresh_s after it hid)
    spot = u["spot_m"][:, None, :]
    if mostly is not None and "spot_tree" in u:
        spot = torch.where(mostly[:, :, None], u["spot_tree"][:, None, :], spot)
    reach = (spot + torch.where(fresh, float(c["fresh_bonus_m"]), 0.0)[:, :, None]) * u["hide_k"][
        :, :, None]                                                     # [B, i hidden, j observer]
    eyes = alive & ~u["r"] if c.get("routers_blind") else alive
    spotted = (pw["enemy"] & eyes[:, None, :] & (pw["gap"] <= reach)).any(2)
    acts = torch.zeros_like(can)
    if rv.get("fire", 1):
        acts = acts | fired
    if rv.get("melee", 1):
        acts = acts | engaged
    if rv.get("attack_order", 1):
        acts = acts | attack
    reveal = spotted | acts
    begin = share > float(c["share_hide"])
    keep = share > float(c["share_keep"])
    # not hidden: the clock runs while it meets the conditions unseen; at hide_s it hides
    clock = torch.where(can & ~hid & begin & ~reveal, u["hid_s"] + dt, torch.zeros_like(u["hid_s"]))
    clock = torch.where(hid, u["hid_s"], clock)
    hides = can & ~hid & (clock >= float(c["hide_s"]) - 1e-6)
    if "hid_d" in u:
        # (start_hidden) deployed in the woods: hidden from the first step, its clock full, the fresh window over
        first = u["hid_d"] & can & ~hid & begin & ~reveal
        clock = torch.where(first, torch.full_like(clock, float(c["hide_s"])), clock)
        hides = hides | first
        u["hid_d"] = torch.zeros_like(u["hid_d"])
    quick = float(c.get("quick_s") or 0.0)
    if quick > 0 and "hid_r" in u:
        # revealed for less than quick_s (1.5 s), it hides again at once when it meets the conditions unseen (the
        # program: stealth.md 1); hid_r - the seconds of that window left since it was revealed
        again = can & ~hid & begin & ~reveal & (u["hid_r"] > 1e-6)
        clock = torch.where(again, torch.full_like(clock, float(c["hide_s"])), clock)
        hides = hides | again
    # hidden: revealed at once by an act or an enemy's eyes; out of cover it stays hidden grace_s more
    grace = torch.where(hid & ~keep, u["hid_g"] + dt, torch.zeros_like(u["hid_g"]))
    lost = hid & (reveal | (grace > float(c["grace_s"]) + 1e-6))
    now = (hid & ~lost) | hides
    u["hid"] = now
    u["hid_s"] = torch.where(now, clock, torch.where(lost, torch.zeros_like(clock), clock))
    u["hid_t"] = torch.where(hides, torch.zeros_like(u["hid_t"]), torch.where(now, u["hid_t"] + dt,
                                                                               torch.zeros_like(u["hid_t"])))
    u["hid_g"] = torch.where(now, grace, torch.zeros_like(grace))
    if quick > 0 and "hid_r" in u:
        u["hid_r"] = torch.where(lost, torch.full_like(u["hid_r"], quick),
                                 torch.where(now, torch.zeros_like(u["hid_r"]), (u["hid_r"] - dt).clamp(min=0)))
    u["vis"] = present & ~now

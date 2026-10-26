"""What carrying out a mission is (S28; docs/en/training/general.md "Missions"): every decision, for each own unit of a
group of a plan, a number 0..1 of how well it does its group's mission kind (config.MISSION_KINDS) towards its group's
target, from the simulator's state after the decision's steps (runtime.Runtime.execute; the executor's bonus is that
number against its kind's running mean, signal.exec_bonus).

The state's truth, not the side's view (a judge of the training: no unit is hidden in the simulator). A target group
is its units standing now (alive, on the map, not routing); its point - their centre; a tile's - its centre. The
measures (d: metres; near(d, R) = 1 within R, down to 0 at 3 R):

    attack    1 in melee with a unit of the target group (the state's m and target), else 0.5 x (1 - d / NEAR_M) to its
              centre: "close on it and fight it" (the centre, not its nearest unit: a straggler running off is not the
              group - the nearest unit paid a chase of one crossbow 700 m off, build/diag_skv)
    flank     as attack, counted only off the front of the target: in melee (flank_by "contact", run.py
              --general-flank-measure, the default) by the side its contact with a unit of the target group BEGAN on
              (the state's hit_sec, the simulator's morale.attacked_first 'attacked in the flank / rear': flank or rear
              -> 1, front -> 0; the struck unit turns to face its attacker, so the angle now says 'front' after a good
              flank charge); where no contact side is held (hit_sec absent or 0 - the game's state, the rule off) and
              under "angle", and before melee: where the unit stands seen from the target's unit it fights (else the
              nearest one) - more than FLANK_DEG off that unit's facing, the sides of a struck unit the simulator uses
              (config/nn/sim.json contact.front_deg 45: CA's front +-45)
    hold      near(d to the target point, the reach): a tile's half side (the tile itself), GROUP_M around a group
    screen    on the segment from the target group's centre to the standing enemy unit nearest to it (its projection
              0..1 of the way) and near(the distance off the segment, SCREEN_M). With run.py --general-screen-clear 1:
              0 while the unit stands in a line of fire of a standing own direct-fire shooter (fireline.screen_lines:
              its aim, else the nearest enemy in its range) - a screen stands beside or ahead off the shooters' lines,
              not in them (in the sim g45_2 vs S27 screens were 18 % of the friends in the lines, build/diag_fireline)
    shoot     a shooter (range and ammunition): 0 in melee, 1 shooting at a unit of the target group, 0.5 within its
              range of one, else 0; without missiles: 0 in melee, 1 at KEEP_M of the target group's nearest unit.
              The line of fire (run.py --general-exec-fireline F > 0; 0: off, as before): a shooter's part x (1 - F x
              the share of its fire over the decision's steps that friends took away - the simulator's fc_skip / fc_cap,
              missile.friend_los: held for a closed line, or its men's paths blocked; fireline_share). In the sim
              g44_3 vs S27 (build/diag_fireline/sim) 95 % of the shooters' direct fire is under "shoot", 28 % of it
              lost, ~80 % of the loss to our other groups' melee units in the line: the measure said 1 for a unit that
              fired with half its men blocked.
              Strict (run.py --general-exec-shoot-strict S > 0; 0: off): while the target group's nearest unit is in
              the shooter's range and seen by our side (the state's vis), not firing at the target group (firing at
              another or silent) gives 0.5 x (1 - S) instead of 0.5; out of range or unseen - as before. The shooters
              did not take the mission's target: a changed target left the fire by groups as it was (26.4 % -> 26.3 %,
              build/diag_intel/targets)
    withdraw  0 in melee, else 0.5 x near(d to the target point, the reach) + 0.5 x (distance to the nearest standing
              enemy / AWAY_M, at most 1)

The blocker (run.py --general-exec-blocker B > 0; 0: off): every measure of a unit not in melee x (1 - B x the share of
the decision's steps it stood in the line of fire of another own direct-fire shooter that lost fire to friends -
fireline.blocking, Runtime.note_fire): the friends in the lines were ~80 % of the lost fire, and none of them paid.

Every measure x the group's cohesion: 1 within COHESION_M[0] of its own group's centre (its standing units), down to 0
at COHESION_M[1] - a unit that leaves its group does not carry out the group's mission.

A unit counts (ok) when it decided this decision (acting), its group has a mission and the target is there (a group
with standing units; a tile inside the map; screen: some standing enemy). These are the measure's definitions (what
"done" means), not the game's mechanics.
"""
import math

import torch

from tools.nn.general import config as gconfig

NEAR_M = 300.0           # attack / flank: the closing part (0 from this far)
FLANK_DEG = 45.0         # flank: off the front of the target's unit (config/nn/sim.json contact.front_deg)
GROUP_M = 100.0          # hold / withdraw to a group: reach around its centre (a line of 3-4 units ~150 m wide)
SCREEN_M = 50.0          # screen: within this of the segment (down to 0 at 3x)
KEEP_M = (50.0, 250.0)   # shoot without missiles: the distance kept to the target group (beyond a charge's 30-35 m)
AWAY_M = 300.0           # withdraw: the enemy's distance counted up to this
COHESION_M = (150.0, 300.0)  # every measure x 1 within the first of its own group's centre, down to 0 at the second


def near(d, reach):
    """1 within reach, down to 0 at 3 x reach."""
    return (1.0 - (d - reach) / (2.0 * reach)).clamp(0.0, 1.0)


def _gather(x, idx):
    """x [R, N] at idx [R, N, ...] along the units."""
    R, N = x.shape
    return x.gather(1, idx.reshape(R, -1)).reshape(idx.shape)


FLANK_BY = ("contact", "angle")   # run.py --general-flank-measure


def fireline_share(cap, skip):
    """The share 0..1 of a unit's fire friends took away (the simulator's bookkeeping summed over steps: cap the fire
    it had, skip of it what friends took; missile.friend_los): skip / cap, 0 where it had no fire."""
    return torch.where(cap > 0, (skip / cap.clamp(min=1e-9)).clamp(0.0, 1.0), torch.zeros_like(cap))


def measure(u, b, s, main, kind, target, tgt, acting, G, C, tile_half, cfg, flank_by="contact", blocked=None,
            fireline=0.0, blocking=None, blocker=0.0, screen_clear=False, shoot_strict=0.0):
    """-> (m [R, N] 0..1, ok [R, N] the units counted, k [R, N] each unit's mission kind) of rows [R] (b their battles,
    s their sides 1 / 2) on the simulator's state u ([B, N] fields): main [R, N] the plan's group numbers (ours
    0..G-1, theirs G..2G-1, -1 none), kind / target [R, G] our groups' missions, tgt [R, G, 2] a tile target's world
    point (NaN else), acting [R, N] (the plan in force and the unit deciding), C coarse tiles, tile_half (m) their
    half side, cfg the GeneralConfig (its kinds); flank_by: the flank's melee part by the contact's side ("contact")
    or the angle now ("angle"); blocked [R, N] (fireline_share over the decision) and fireline F: the shoot measure of a
    shooter x (1 - F x blocked) (module doc; None or F 0: as before); blocking [R, N] (the share of the decision a unit
    stood in a line of fire that lost fire) and blocker B: every measure of a unit not in melee x (1 - B x blocking);
    screen_clear: the screen measure 0 in our shooters' lines of fire; shoot_strict S: the shoot measure of a shooter whose
    target is in range and seen but not fired at 0.5 x (1 - S) (module doc)."""
    f = (lambda k: u[k][b])
    x, z = f("x").float(), f("z").float()
    side = f("side")
    alive = (side > 0) & (f("men") > 0) & ~f("gone")
    standing = alive & ~f("r")
    own = side == s[:, None]
    foe = (side > 0) & ~own
    mi = main.clamp(0, G - 1)
    in_g = own & (main >= 0) & (main < G) & acting
    k_u = kind.gather(1, mi)
    t_u = target.gather(1, mi)
    p = torch.stack([x, z], -1)                                                     # [R, N, 2]
    D = (p[:, :, None, :] - p[:, None, :, :]).norm(dim=-1)                          # [R, N, N]
    big = torch.full_like(D, 1e9)
    is_grp = t_u < 2 * G
    member = (main[:, None, :] == t_u[:, :, None]) & standing[:, None, :] & is_grp[..., None]   # [R, N (i), N (j)]
    n_t = member.sum(-1)
    grp_ok = n_t > 0
    dT, jT = torch.where(member, D, big).min(-1)                                    # the target's nearest unit
    cg = torch.einsum("rij,rjc->ric", member.float(), p) / n_t.clamp(min=1)[..., None]
    tp = tgt.gather(1, mi[..., None].expand(-1, -1, 2))
    P = torch.where(is_grp[..., None], cg, torch.nan_to_num(tp))
    P_ok = torch.where(is_grp, grp_ok, torch.isfinite(tp).all(-1) & (t_u < 2 * G + C))
    dP = (p - P).norm(dim=-1)
    reach = torch.where(is_grp, torch.full_like(dP, GROUP_M), torch.full_like(dP, tile_half))
    melee = f("m")
    slot = f("target")
    hit_t = _gather_rows(member, slot) & (slot >= 0)                                 # its target is in the group
    engaged = melee & hit_t
    dC = (p - cg).norm(dim=-1)                                                      # to the target group's centre
    close = torch.where(grp_ok, (1.0 - dC / NEAR_M).clamp(0.0, 1.0), torch.zeros_like(dC))
    attack = torch.where(engaged, torch.ones_like(dT), 0.5 * close)
    # flank: where the unit stands seen from the target's unit (the one it fights, else the nearest one)
    j = torch.where(engaged, slot.clamp(min=0), jT)
    vx, vz = x - _gather(x, j), z - _gather(z, j)
    bj = _gather(f("b").float(), j) * (math.pi / 180.0)
    cos = (vx * torch.sin(bj) + vz * torch.cos(bj)) / (vx * vx + vz * vz).sqrt().clamp(min=1e-3)
    off = cos < math.cos(FLANK_DEG * math.pi / 180.0)
    if flank_by == "contact" and "hit_sec" in u:
        # the side of the target's unit our unit's contact began on (u hit_sec [B, N * N]: row the attacker, column the
        # struck unit; 1 front, 2 flank, 3 rear, 0 none), the worst over the target group's units it touches
        R, N = x.shape
        hs = u["hit_sec"][b].reshape(R, N, N)
        began = torch.where(member, hs, torch.zeros_like(hs)).amax(-1)              # [R, N (i)]
        off = torch.where(engaged & (began > 0), began >= 2, off)
    elif flank_by not in FLANK_BY:
        raise ValueError(f"flank_by {flank_by!r}: one of {FLANK_BY}")
    flank = torch.where(engaged, off.float(), 0.5 * close * off.float())
    hold = near(dP, reach)
    # screen: between the target group's centre A and the standing enemy nearest to it
    foe_st = foe & standing
    dA = (P[:, :, None, :] - p[:, None, :, :]).norm(dim=-1)                          # [R, N (i), N (j)]
    dA = torch.where(foe_st[:, None, :], dA, big)
    jE = dA.argmin(-1)
    foe_any = foe_st.any(-1, keepdim=True).expand_as(dP)
    E = torch.stack([_gather(x, jE), _gather(z, jE)], -1)
    v = E - P
    w = p - P
    vv = (v * v).sum(-1).clamp(min=1e-6)
    along = (w * v).sum(-1) / vv
    lat = (w[..., 0] * v[..., 1] - w[..., 1] * v[..., 0]).abs() / vv.sqrt()
    screen = ((along >= 0.0) & (along <= 1.0)).float() * near(lat, SCREEN_M)
    if screen_clear:
        from tools.nn.general import fireline as gfire
        screen = screen * (~gfire.screen_lines(u, b, s, foe_st)).float()
    # shoot
    shooter = (f("range") > 0) & (f("a") > 0)
    firing = f("fire") & hit_t
    rng = f("range").float()
    in_rng = grp_ok & (dT <= rng)
    half = torch.full_like(dT, 0.5)
    if shoot_strict > 0:
        seen = _gather(f("vis"), jT) if "vis" in u else torch.ones_like(in_rng)
        half = torch.where(seen, half * (1.0 - shoot_strict), half)        # in range and seen: fire at it or less
    s_sh = torch.where(firing, torch.ones_like(dT), torch.where(in_rng, half, torch.zeros_like(dT)))
    s_no = (grp_ok & (dT >= KEEP_M[0]) & (dT <= KEEP_M[1])).float()
    if blocked is not None and fireline > 0:
        s_sh = s_sh * (1.0 - fireline * blocked.to(s_sh.dtype)).clamp(min=0.0)     # the line of fire (module doc)
    shoot = torch.where(melee, torch.zeros_like(dT), torch.where(shooter, s_sh, s_no))
    # withdraw
    dE = torch.where(foe_st[:, None, :], D, big).min(-1).values
    away = torch.where(foe_any, (dE / AWAY_M).clamp(0.0, 1.0), torch.ones_like(dE))
    withdraw = torch.where(melee, torch.zeros_like(dT), 0.5 * hold + 0.5 * away)
    by = {"attack": (attack, grp_ok), "hold": (hold, P_ok), "flank": (flank, grp_ok), "screen": (screen, P_ok & foe_any),
          "shoot": (shoot, grp_ok), "withdraw": (withdraw, P_ok)}
    names = [k.name for k in gconfig.mission_kinds(cfg)]
    m_all = torch.stack([by[n][0] for n in names], -1)                              # [R, N, K]
    v_all = torch.stack([by[n][1] for n in names], -1)
    kk = k_u.clamp(0, len(names) - 1)[..., None]
    m = m_all.gather(-1, kk)[..., 0]
    # the cohesion: the distance to its own group's centre (the group's standing units; the unit itself counts)
    mate = (main[:, None, :] == main[:, :, None]) & (standing & own)[:, None, :]
    mate = mate | torch.eye(main.shape[1], dtype=torch.bool, device=main.device)[None]
    co = torch.einsum("rij,rjc->ric", mate.float(), p) / mate.sum(-1).clamp(min=1)[..., None]
    lo, hi = COHESION_M
    m = m * (1.0 - ((p - co).norm(dim=-1) - lo) / (hi - lo)).clamp(0.0, 1.0)
    if blocking is not None and blocker > 0:
        m = torch.where(melee, m, m * (1.0 - blocker * blocking.to(m.dtype)).clamp(min=0.0))   # the blocker
    ok = in_g & v_all.gather(-1, kk)[..., 0]
    return torch.where(ok, m, torch.zeros_like(m)), ok, k_u


def _gather_rows(member, slot):
    """member [R, N, N] at the column slot [R, N] of each row i (-1: False)."""
    return member.gather(-1, slot.clamp(min=0)[..., None])[..., 0]

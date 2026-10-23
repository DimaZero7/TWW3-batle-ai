"""The teacher general (run.py --general-teacher P; docs/en/training/general.md "Teacher"): in a share P of the
learner's battles a scripted rule gives our groups their missions instead of the network's choices (the groups are the
network's), so the executor learns to carry out meaningful missions while the general's own choices are still near
random. Those plans train the executor only: not the general's PPO, the plan value, the classifier or the branches
(runtime: the row's "teach").

The rule (on the groups' standing units, the state's truth), later rules over earlier ones:
  every group     attack the nearest enemy group (by the groups' centres);
  shooters        a group whose units are mostly shooters (range and ammunition): shoot the enemy group its shots hurt
                  most - the mean HP a shot of its shooters takes from that group's units (shot_hp: the simulator's
                  damage rule melee.per_hit - armour roll, overkill, resistances - less a shield's block of small arms
                  from the front) x the share of those pairs in reach (range + APPROACH_M); the nearest group on a tie
                  and where no group is in reach (without `shot`: the nearest). The human's battle against g31: by the
                  nearest group 96 % of the slingers' fire (armour-piercing 1) went at greatswords of armour 95;
  one screen      the non-shooting group without the lord nearest to our largest shooting group screens it;
  the lord        his group, with a standing enemy unit within THREAT_M of him: withdraw to our largest other group -
                  unless the group fights in melee and holds (some unit in melee, none wavering: taskmask.fighting, the
                  general's mask GeneralConfig.melee_stay): then it holds the tile it stands on (the lord rides with
                  the infantry by the group rule; the human's battle against g31: "withdraw" to the whole infantry in
                  melee, 46 orders, those units killed 0.03 men/s against 0.19). No rule gives "withdraw" to a group
                  that fights and holds;
  wait            the groups left to attack hold the coarse tile each stands on while the farthest of them has its
                  nearest standing enemy unit farther than HOLD_M from its centre and our shooters have not been
                  firing for FIRE_S yet (the row's `engage`); then all attack together - S27 waited and met the enemy
                  (80 hold orders in the first 100 s against the general's 8, build/diag_skv); by each group alone
                  (the human's battle against g31) the attack went piecemeal, contact 81 s apart (the runtime's re-plan
                  on "the enemy near", GeneralConfig.replan_near_m, makes the switch at once);
  the flank       (flank=True, run.py --general-teacher-flank 1; build/diag_flank: the teacher never gave flank, the
                  general picked it for 1-2 % of groups and the measure stayed 0):
                  - two or more of our non-fast groups attack one enemy group X within NEAR_FLANK_M of them: the one
                    nearest to X's front point (FRONT_M ahead of its centre along its units' mean facing; without
                    facings - its centre) attacks and pins it, the others flank X;
                  - a fast group (most units of the role "fast", groups.roles: cavalry, chariots, a run of 7 m/s; not a
                    shooting one) told attack or withdraw flanks the enemy group X nearest to it that one of our
                    non-fast, non-shooting groups pins: attacks X, or holds / screens with its centre within PIN_M of
                    X's centre (the pinning group's nearest enemy group).
Where no enemy group stands the network's choice stays. Every pair is one the general may choose (its kind takes the
target's type; shoot only a group with a shooter - GeneralConfig.task_mask's rule; hold a tile there is), so the
general's imitation of the teacher (learn.py) never meets a masked pair.
"""
import torch

from tools.nn.general import config as gconfig
from tools.nn.general import taskmask as gtaskmask

THREAT_M = 150.0         # the lord is threatened by a standing enemy unit this close
HOLD_M = 250.0           # a group holds while the nearest standing enemy unit is farther than this (run.py
#                          --general-teacher-hold-m) ...
FIRE_S = 30.0            # ... and until our shooters have been firing this long (--general-teacher-fire-s)
FRONT_M = 100.0          # the flank rule: the target's front point this far ahead of its centre (the pinning group is
#                          the nearest to it; my rule - a line of 3-4 units is ~150 m wide)
NEAR_FLANK_M = 300.0     # ... a second attacker flanks an enemy group whose centre is within this of its own
PIN_M = 150.0            # ... a holding / screening group pins its nearest enemy group within this (centre to centre)
APPROACH_M = 100.0       # shoot: an enemy unit within the shooter's range + this is in reach (a short step in: ~20-25 s
#                          at a walk, inside one re-plan of 30 s; my rule)
TIE = 1e-3               # shoot: groups whose value is within this share of the best are equal (the nearest wins)
INDEX = {k.name: k.index for k in gconfig.MISSION_KINDS}


def shot_hp(x, z, rng, m_damage, m_ap, small_arms, armour, hp_man, shield, resist, men0, approach=APPROACH_M):
    """[M, N (shooter i), N (target j)] the HP a projectile of unit i takes from unit j when it hits, the simulator's
    rule (missile.volley: melee.per_hit of the projectile's damage and armour-piercing against j's armour, health of a
    man and resistance; a lone man loses the mean; a shield blocks small arms from the front - the teacher assumes the
    front), 0 where j is beyond i's range + approach (centre to centre) or i has no range. The fields [M, N] are the
    state's (sim/state.py); resist: the resistance against i's projectiles (missile + physical, at most 0.9)."""
    from tools.nn.sim import melee as smelee
    hit = smelee.per_hit(m_damage.float()[:, :, None], m_ap.float()[:, :, None], armour.float()[:, None, :],
                         hp_man.float()[:, None, :], resist.float()[:, None, :], single=(men0 <= 1)[:, None, :])
    hit = hit * (1 - shield.float()[:, None, :] * small_arms.float()[:, :, None])
    d = torch.hypot(x.float()[:, :, None] - x.float()[:, None, :], z.float()[:, :, None] - z.float()[:, None, :])
    reach = (rng.float()[:, :, None] > 0) & (d <= rng.float()[:, :, None] + approach)
    return hit * reach.float()


def shoot_target(mem, shooter, shot, d, G, near_e):
    """[M, G] each of our groups' shoot target (module doc, "shooters"): the enemy group of the largest mean shot_hp
    of the group's shooters at its units (pairs out of reach count 0); within TIE of the best - the nearest (d [M, G,
    G] our groups' centres to theirs, inf where no group); nothing in reach - near_e."""
    sh = (mem[:, :, :G] & shooter[..., None]).float()                              # [M, N, G] our shooters
    foe = mem[:, :, G:].float()                                                     # [M, N, G] their units
    tot = torch.einsum("mig,mij,mje->mge", sh, shot.float(), foe)
    val = tot / (sh.sum(1)[:, :, None] * foe.sum(1)[:, None, :]).clamp(min=1)       # [M, G, G (enemy)]
    val = val.masked_fill(~torch.isfinite(d), 0.0)
    best = val.max(-1, keepdim=True).values
    top = (val >= best * (1 - TIE)) & (best > 0) & torch.isfinite(d)
    pick = d.masked_fill(~top, float("inf")).argmin(-1)
    return torch.where(best[..., 0] > 0, G + pick, near_e)


def own_tile(cent, tiles, tiles_ok):
    """[M, G] the coarse tile (of those that may be targets) nearest to each group's centre cent [M, G, 2]."""
    dt = (cent[:, :, None, :] - tiles[:, None, :, :]).norm(dim=-1).masked_fill(~tiles_ok[:, None, :], float("inf"))
    return dt.argmin(-1)


def choose(main, active, x, z, own, standing, shooter, lord, G, kinds, tiles=None, tiles_ok=None, engage=None,
           hold_m=HOLD_M, facing=None, flank=True, fast=None, melee=None, waver=None, shot=None):
    """-> (kind [M, G], target [M, G], use [M, G]: our groups the rule gives a mission) of plans [M]: main [M, N] the
    plan's group numbers, active [M, 2G], the units' x, z, own, standing, shooter, lord [M, N]; kinds: the network's
    number of kinds (a rule whose kind is beyond it is not used); tiles [M, C, 2] the coarse tiles' world centres,
    tiles_ok [M, C] those that may be targets, engage [M] our shooters have fired FIRE_S (None: no waiting - every
    group attacks); facing [M, N] the units' facings (degrees, the state's b; None: the flank rule by the centres),
    flank: the flank rule on, fast [M, N] the units of the role "fast" (groups.roles; None: no fast groups); melee,
    waver [M, N] the units in melee and those wavering or worse (None: nobody - the lord's group withdraws always);
    shot [M, N, N] the HP a shot of unit i takes from unit j in reach (shot_hp; None: shooters shoot the nearest)."""
    M, N = main.shape
    dev = main.device
    ar = torch.arange(2 * G, device=dev)
    mem = (main[..., None] == ar) & standing[..., None]                             # [M, N, 2G]
    n = mem.sum(1).float()
    p = torch.stack([x.float(), z.float()], -1)
    cent = torch.einsum("mng,mnc->mgc", mem.float(), p) / n.clamp(min=1)[..., None]
    act = active & (n > 0)
    own_g, foe_g = act[:, :G], act[:, G:]
    d = (cent[:, :G, None, :] - cent[:, None, G:, :]).norm(dim=-1).masked_fill(~foe_g[:, None, :], float("inf"))
    near_e = G + d.argmin(-1)                                                       # [M, G]
    has_e = foe_g.any(-1, keepdim=True).expand(M, G)
    k_of = INDEX.get
    on = (lambda name: INDEX[name] < kinds)              # a rule whose kind the network has not: not used
    kind = torch.full((M, G), k_of("attack"), dtype=torch.long, device=dev)
    target = near_e.clone()
    use = own_g & has_e
    foe_st = ~own & standing
    # shooters
    sh = torch.einsum("mng,mn->mg", mem[:, :, :G].float(), shooter.float()) / n[:, :G].clamp(min=1)
    shooters = own_g & (sh > 0.5) & on("shoot")
    kind = torch.where(shooters & has_e, torch.full_like(kind, k_of("shoot")), kind)
    target = torch.where(shooters & has_e, near_e if shot is None else shoot_target(mem, shooter, shot, d, G, near_e),
                         target)
    # one screen of the largest shooting group
    lord_g = (mem[:, :, :G] & lord[..., None]).any(1)
    size = n[:, :G]
    big_s = size.masked_fill(~shooters, -1.0)
    s_idx = big_s.argmax(-1)                                                        # [M]
    has_s = big_s.max(-1).values > 0
    cand = own_g & ~shooters & ~lord_g & has_s[:, None] & on("screen")
    ds = (cent[:, :G] - cent[:, :G].gather(1, s_idx[:, None, None].expand(-1, 1, 2))).norm(dim=-1)
    ds = ds.masked_fill(~cand, float("inf"))
    x_idx = ds.argmin(-1)
    has_x = cand.any(-1)
    pick = (torch.arange(G, device=dev) == x_idx[:, None]) & has_x[:, None]
    kind = torch.where(pick, torch.full_like(kind, k_of("screen")), kind)
    target = torch.where(pick, s_idx[:, None].expand(M, G), target)
    use = use | pick
    # the lord threatened: withdraw to the largest other group
    lord_st = lord & standing & own
    lp = (p * lord_st[..., None].float()).sum(1) / lord_st.float().sum(1).clamp(min=1)[:, None]
    dl = (p - lp[:, None, :]).norm(dim=-1).masked_fill(~foe_st, float("inf")).min(-1).values
    threat = lord_st.any(-1) & (dl < THREAT_M)
    other = size[:, None, :].expand(M, G, G).clone().masked_fill(~own_g[:, None, :], -1.0)
    other[:, torch.arange(G, device=dev), torch.arange(G, device=dev)] = -1.0
    o_idx = other.argmax(-1)
    has_o = other.max(-1).values > 0
    fights = torch.zeros_like(own_g)
    if melee is not None:
        fights = gtaskmask.fighting(mem[:, :, :G], melee, torch.zeros_like(melee) if waver is None else waver)
    wd = own_g & lord_g & threat[:, None] & has_o & on("withdraw") & ~fights
    kind = torch.where(wd, torch.full_like(kind, k_of("withdraw")), kind)
    target = torch.where(wd, o_idx, target)
    use = use | wd
    # ... the lord's group fighting in melee: holds the tile it stands on (no withdraw, no chase)
    if tiles is not None and on("hold"):
        lh = own_g & lord_g & threat[:, None] & fights & tiles_ok.any(-1, keepdim=True)
        kind = torch.where(lh, torch.full_like(kind, k_of("hold")), kind)
        target = torch.where(lh, 2 * G + own_tile(cent[:, :G], tiles, tiles_ok), target)
        use = use | lh
    # wait: the attacking groups hold the tiles they stand on while the farthest of them is far from the enemy and our
    # shooters have not fired long; then all attack together
    if engage is not None and tiles is not None and on("hold"):
        de = (cent[:, :G, None, :] - p[:, None, :, :]).norm(dim=-1).masked_fill(~foe_st[:, None, :], float("inf"))
        att = use & (kind == k_of("attack"))
        far = ((de.min(-1).values > hold_m) & att).any(-1, keepdim=True)          # [M, 1] the farthest attacker
        wait = att & far & ~engage[:, None] & tiles_ok.any(-1, keepdim=True)
        kind = torch.where(wait, torch.full_like(kind, k_of("hold")), kind)
        target = torch.where(wait, 2 * G + own_tile(cent[:, :G], tiles, tiles_ok), target)
    # the flank
    if flank and on("flank"):
        fast_g = torch.zeros_like(own_g)
        if fast is not None:
            fs = torch.einsum("mng,mn->mg", mem[:, :, :G].float(), fast.float()) / n[:, :G].clamp(min=1)
            fast_g = own_g & (fs > 0.5) & ~shooters
        tix = (target - G).clamp(0, G - 1)
        d_t = d.gather(-1, tix[..., None])[..., 0]                                  # to its target's centre
        # (b) two or more non-fast groups attack one enemy group near them: the nearest to its front pins, others flank
        att = use & (kind == k_of("attack")) & ~fast_g                              # [M, G]
        front = cent[:, G:]                                                         # [M, G (enemy), 2]
        if facing is not None:
            rad = torch.deg2rad(facing.float())
            dirs = torch.stack([torch.sin(rad), torch.cos(rad)], -1)                # (x, z) of the facing b
            md = torch.einsum("mng,mnc->mgc", mem[:, :, G:].float(), dirs)
            md = md / md.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            front = front + FRONT_M * md
        fp = front.gather(1, tix[..., None].expand(-1, -1, 2))                      # [M, G, 2] of each group's target
        dfp = (cent[:, :G] - fp).norm(dim=-1).masked_fill(~att, float("inf"))
        same = (tix[:, :, None] == tix[:, None, :]) & att[:, :, None] & att[:, None, :]          # [M, G, G]
        ar_g = torch.arange(G, device=dev)
        # another attacker of the same target is nearer to its front (ties: the lower number pins)
        before = same & ((dfp[:, None, :] < dfp[:, :, None])
                         | ((dfp[:, None, :] == dfp[:, :, None]) & (ar_g[None, None, :] < ar_g[None, :, None])))
        fl = att & before.any(-1) & (d_t < NEAR_FLANK_M)
        kind = torch.where(fl, torch.full_like(kind, k_of("flank")), kind)
        # (a) a fast group flanks the nearest enemy group one of our non-fast, non-shooting groups pins
        pinner = own_g & use & ~fast_g & ~shooters
        pin_att = pinner & (kind == k_of("attack"))
        d_near = d.min(-1).values
        pin_hold = pinner & ((kind == k_of("hold")) | (kind == k_of("screen"))) & (d_near < PIN_M)
        x_att = torch.where(pin_att, tix, torch.full_like(tix, -1))
        x_hold = torch.where(pin_hold, near_e - G, torch.full_like(tix, -1))
        ar_e = torch.arange(G, device=dev)
        pinned = ((x_att[..., None] == ar_e) | (x_hold[..., None] == ar_e)).any(1) & foe_g        # [M, G (enemy)]
        dp = d.masked_fill(~pinned[:, None, :], float("inf"))                       # [M, G (ours), G (enemy)]
        px = dp.argmin(-1)
        fa = fast_g & use & ((kind == k_of("attack")) | (kind == k_of("withdraw"))) & pinned.any(-1, keepdim=True)
        kind = torch.where(fa, torch.full_like(kind, k_of("flank")), kind)
        target = torch.where(fa, G + px, target)
    return kind, target, use

"""The teacher general (run.py --general-teacher P; docs/en/training/general.md "Teacher"): in a share P of the
learner's battles a scripted rule gives our groups their missions instead of the network's choices (the groups are the
network's), so the executor learns to carry out meaningful missions while the general's own choices are still near
random. Those plans train the executor only: not the general's PPO, the plan value, the classifier or the branches
(runtime: the row's "teach").

The rule (on the groups' standing units, the state's truth), later rules over earlier ones:
  every group     wait, then attack: hold the coarse tile it stands on while the nearest standing enemy unit is farther
                  than HOLD_M from its centre and our shooters have not been firing for FIRE_S yet (the row's `engage`);
                  then attack the nearest enemy group (by the groups' centres) - S27 waited and met the enemy (80 hold
                  orders in the first 100 s against the general's 8, build/diag_skv);
  shooters        a group whose units are mostly shooters (range and ammunition): shoot the nearest enemy group;
  one screen      the non-shooting group without the lord nearest to our largest shooting group screens it;
  the lord        his group, with a standing enemy unit within THREAT_M of him: withdraw to our largest other group;
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

THREAT_M = 150.0         # the lord is threatened by a standing enemy unit this close
HOLD_M = 250.0           # a group holds while the nearest standing enemy unit is farther than this (run.py
#                          --general-teacher-hold-m) ...
FIRE_S = 30.0            # ... and until our shooters have been firing this long (--general-teacher-fire-s)
FRONT_M = 100.0          # the flank rule: the target's front point this far ahead of its centre (the pinning group is
#                          the nearest to it; my rule - a line of 3-4 units is ~150 m wide)
NEAR_FLANK_M = 300.0     # ... a second attacker flanks an enemy group whose centre is within this of its own
PIN_M = 150.0            # ... a holding / screening group pins its nearest enemy group within this (centre to centre)
INDEX = {k.name: k.index for k in gconfig.MISSION_KINDS}


def choose(main, active, x, z, own, standing, shooter, lord, G, kinds, tiles=None, tiles_ok=None, engage=None,
           hold_m=HOLD_M, facing=None, flank=True, fast=None):
    """-> (kind [M, G], target [M, G], use [M, G]: our groups the rule gives a mission) of plans [M]: main [M, N] the
    plan's group numbers, active [M, 2G], the units' x, z, own, standing, shooter, lord [M, N]; kinds: the network's
    number of kinds (a rule whose kind is beyond it is not used); tiles [M, C, 2] the coarse tiles' world centres,
    tiles_ok [M, C] those that may be targets, engage [M] our shooters have fired FIRE_S (None: no waiting - every
    group attacks); facing [M, N] the units' facings (degrees, the state's b; None: the flank rule by the centres),
    flank: the flank rule on, fast [M, N] the units of the role "fast" (groups.roles; None: no fast groups)."""
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
    # wait: hold the tile it stands on while the enemy is far and our shooters have not fired long
    if engage is not None and tiles is not None and on("hold"):
        de = (cent[:, :G, None, :] - p[:, None, :, :]).norm(dim=-1).masked_fill(~foe_st[:, None, :], float("inf"))
        far = de.min(-1).values > hold_m                                            # [M, G]
        dt = (cent[:, :G, None, :] - tiles[:, None, :, :]).norm(dim=-1).masked_fill(~tiles_ok[:, None, :],
                                                                                   float("inf"))
        t_idx = dt.argmin(-1)
        wait = far & ~engage[:, None] & tiles_ok.any(-1, keepdim=True)
        kind = torch.where(wait, torch.full_like(kind, k_of("hold")), kind)
        target = torch.where(wait, 2 * G + t_idx, target)
    # shooters
    sh = torch.einsum("mng,mn->mg", mem[:, :, :G].float(), shooter.float()) / n[:, :G].clamp(min=1)
    shooters = own_g & (sh > 0.5) & on("shoot")
    kind = torch.where(shooters & has_e, torch.full_like(kind, k_of("shoot")), kind)
    target = torch.where(shooters & has_e, near_e, target)
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
    wd = own_g & lord_g & threat[:, None] & has_o & on("withdraw")
    kind = torch.where(wd, torch.full_like(kind, k_of("withdraw")), kind)
    target = torch.where(wd, o_idx, target)
    use = use | wd
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

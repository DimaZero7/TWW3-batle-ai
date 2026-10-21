"""The teacher general (run.py --general-teacher P; docs/en/training/general.md "Teacher"): in a share P of the
learner's battles a scripted rule gives our groups their missions instead of the network's choices (the groups are the
network's), so the executor learns to carry out meaningful missions while the general's own choices are still near
random. Those plans train the executor only: not the general's PPO, the plan value, the classifier or the branches
(runtime: the row's "teach").

The rule (on the groups' standing units, the state's truth), later rules over earlier ones:
  every group     attack the nearest enemy group (by the groups' centres);
  shooters        a group whose units are mostly shooters (range and ammunition): shoot the nearest enemy group;
  one screen      the non-shooting group without the lord nearest to our largest shooting group screens it;
  the lord        his group, with a standing enemy unit within THREAT_M of him: withdraw to our largest other group.
Where no enemy group stands the network's choice stays.
"""
import torch

from tools.nn.general import config as gconfig

THREAT_M = 150.0         # the lord is threatened by a standing enemy unit this close
INDEX = {k.name: k.index for k in gconfig.MISSION_KINDS}


def choose(main, active, x, z, own, standing, shooter, lord, G, kinds):
    """-> (kind [M, G], target [M, G], use [M, G]: our groups the rule gives a mission) of plans [M]: main [M, N] the
    plan's group numbers, active [M, 2G], the units' x, z, own, standing, shooter, lord [M, N]; kinds: the network's
    number of kinds (a rule whose kind is beyond it is not used)."""
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
    # shooters
    sh = torch.einsum("mng,mn->mg", mem[:, :, :G].float(), shooter.float()) / n[:, :G].clamp(min=1)
    shooters = own_g & (sh > 0.5) & on("shoot")
    kind = torch.where(shooters & has_e, torch.full_like(kind, k_of("shoot")), kind)
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
    foe_st = ~own & standing
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
    return kind, target, use

"""The group collection by a rule (GeneralConfig.groups "rule"; docs/en/training/general.md "Group collection").

The module atlas of g17r (build/diag_atlas2) found the learned collection (net.SlotGroups' attention) degenerate: one
group for the whole army at every battle's first plan, 1.6 later, the group cards alike (cos 0.995), shooters mixed with
infantry as at random. Until the general learns its kinds by the battle, the groups are a fixed rule, the same for both
sides, on what the general's input holds (army slots, places):

  1. roles, by the unit's card: the lord and heroes (caste lord / hero); fast (a cavalry or chariot caste, or a run of
     at least FAST_RUN m/s); shooters (a missile range); melee (the rest);
  2. within a role, clusters by place: the two clusters of a side and a role whose centres are nearest join while they
     are closer than `r` (GeneralConfig.group_r) - greedily, one pair at a time;
  3. a lone lord or hero (a cluster of one) joins the nearest melee cluster of his side whose centre is closer than
     `lord_m` (GeneralConfig.lord_join_m), else stays his own group;
  4. at most `G` groups a side: while a side has more, its two nearest clusters join (those of one role first).

Only free units (net "free": present, alive, not routing) take part; an enemy never seen stands at its zeros (the
input's). -> raw [P, N] each unit's place (ours 0..G-1, theirs G..2G-1, by the clusters' first slot; -1 not free); the
groups' numbers then follow net.match_groups as the learned collection's do.
"""
import torch

from tools.nn.general import army_input as ai
from tools.nn.model import passport as pp

ROLES = ("shoot", "melee", "lord", "fast")
FAST_RUN = 7.0            # m/s: a unit that runs at least this fast is "fast" (cavalry runs 8-10, infantry 4-5)
FAST_CASTES = ("melee_cavalry", "missile_cavalry", "monstrous_cavalry", "chariot")
POS_M = 500.0             # net.POS_M (the input's places are / this)
_BIG = 1e9


def roles(num, lab):
    """[P, N] long: each unit's role (ROLES' index) of its slot numbers num [P, N, 45] and labels lab [P, N, 3]."""
    caste = lab[..., ai.LABELS.index("caste")]
    lord = (caste == pp.CASTES.index("lord")) | (caste == pp.CASTES.index("hero"))
    fast_c = torch.zeros_like(lord)
    for c in FAST_CASTES:
        fast_c = fast_c | (caste == pp.CASTES.index(c))
    fast = fast_c | (num[..., ai.INDEX["run"]] * 10.0 >= FAST_RUN)
    shoot = num[..., ai.INDEX["range"]] > 0
    out = torch.full_like(caste, ROLES.index("melee"))
    out = torch.where(shoot, torch.full_like(out, ROLES.index("shoot")), out)
    out = torch.where(fast, torch.full_like(out, ROLES.index("fast")), out)
    return torch.where(lord, torch.full_like(out, ROLES.index("lord")), out)


def _centres(label, xy, free):
    """(centre [P, N, 2] of every cluster id, count [P, N]) of the units' cluster ids label [P, N]."""
    P, N = label.shape
    w = free.to(xy.dtype)
    idx = label.clamp(min=0)
    tot = xy.new_zeros(P, N, 2).scatter_add_(1, idx[..., None].expand(-1, -1, 2), xy * w[..., None])
    cnt = xy.new_zeros(P, N).scatter_add_(1, idx, w)
    return tot / cnt.clamp(min=1.0)[..., None], cnt


def _merge(label, cost):
    """One merge a row: the pair (i < j) of the least finite cost [P, N, N] - the units of cluster j go to i.
    -> (label, merged [P])."""
    P, N = label.shape
    flat = cost.reshape(P, -1)
    best, at = flat.min(-1)
    ok = best < _BIG / 2
    i, j = at // N, at % N
    lo, hi = torch.minimum(i, j), torch.maximum(i, j)
    move = ok[:, None] & (label == hi[:, None])
    return torch.where(move, lo[:, None].expand_as(label), label), ok


def collect(num, lab, pos, own, free, G, r=100.0, lord_m=60.0):
    """raw [P, N]: each free unit's place by the rule (module doc); -1 for the others."""
    P, N = free.shape
    dev = free.device
    xy = pos[..., :2].float() * POS_M
    role = roles(num, lab)
    ar = torch.arange(N, device=dev)
    label = torch.where(free, ar.expand(P, N), torch.full_like(role, -1))
    side_u = own.long()
    upper = ar[:, None] < ar[None, :]                                               # i < j

    def state():
        cent, cnt = _centres(label, xy, free)
        root = (label == ar) & free                                                 # cluster ids alive: their root
        # a cluster's role and side: its root's (the clusters of steps 2-3 are of one role and side)
        pair = root[:, :, None] & root[:, None, :] & upper[None]
        pair = pair & (side_u[:, :, None] == side_u[:, None, :])
        d = (cent[:, :, None, :] - cent[:, None, :, :]).norm(dim=-1)
        return cent, cnt, root, pair, d

    melee, lordr = ROLES.index("melee"), ROLES.index("lord")
    # 2. within a role, by place
    for _ in range(N):
        cent, cnt, root, pair, d = state()
        ok = pair & (role[:, :, None] == role[:, None, :]) & (d < r)
        label, merged = _merge(label, torch.where(ok, d, torch.full_like(d, _BIG)))
        if not bool(merged.any()):
            break
    # 3. a lone lord or hero to the nearest melee cluster within lord_m
    for _ in range(N):
        cent, cnt, root, pair, d = state()
        lone = root & (role == lordr) & (cnt == 1)
        both = pair | pair.transpose(1, 2)
        ok = both & lone[:, :, None] & (role == melee)[:, None, :] & root[:, None, :] & (d < lord_m)
        label, merged = _merge(label, torch.where(ok, d, torch.full_like(d, _BIG)))
        if not bool(merged.any()):
            break
    # 4. at most G groups a side
    for _ in range(N):
        cent, cnt, root, pair, d = state()
        n_side = torch.stack([(root & ~own).sum(1), (root & own).sum(1)], -1)       # [P, 2] (side_u 0, 1)
        over = n_side.gather(1, side_u) > G                                         # [P, N] the unit's side has too many
        ok = pair & over[:, :, None]
        cost = d + 1e5 * (role[:, :, None] != role[:, None, :]).to(d.dtype)
        label, merged = _merge(label, torch.where(ok, cost, torch.full_like(d, _BIG)))
        if not bool(merged.any()):
            break
    # the places: each side's clusters in the order of their first slot
    root = (label == ar) & free
    rank_own = torch.cumsum((root & own).long(), 1) - 1
    rank_foe = torch.cumsum((root & ~own).long(), 1) - 1
    place = torch.where(own, rank_own, G + rank_foe)                               # [P, N] by cluster id
    raw = place.gather(1, label.clamp(min=0))
    return torch.where(free, raw, torch.full_like(raw, -1))

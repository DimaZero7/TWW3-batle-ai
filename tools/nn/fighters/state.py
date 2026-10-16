"""The lab's soldiers: every man of every unit a point with his own health, beside the simulator's unit state.

Layout: B battles, M places a battle (the most men of any battle of the batch; the rest are padding). A unit's men are
contiguous, in its slot order: unit n of battle b owns [ustart[b, n], ustart[b, n] + men0[b, n]). Per man [B, M]:
  uid        his unit's slot (0 for padding: `valid` False)
  file, rank0  his file and starting rank in the formation (rank-major; the last rank's men centred)
  dx, dz     his place as an offset from his unit's centre, m, along the map's axes (world-aligned: the unit's move
             carries him, its turn does not - he walks to his new place)
  hp         his health (the passport's hp_per_man whole), alive = hp > 0 and a man of the unit
  cd         his attack clock, s (0: ready)
  tgt        the man he fights (an index into M, -1 none)
Per unit [B, N]: flip (the formation turned about: ranks and files swapped, the game's about-face in place), b_last.
Static: the order that groups each file of a unit (file compaction: the man behind steps up).

Needs torch."""
import math

import torch

from tools.nn.sim import geometry


class Fighters:
    """The lab's men (module docstring). Build with build(st, params)."""

    def __init__(self, **t):
        self.__dict__.update(t)

    @property
    def B(self):
        return self.hp.shape[0]

    @property
    def M(self):
        return self.hp.shape[1]

    def clone(self):
        return Fighters(**{k: (v.clone() if torch.is_tensor(v) else v) for k, v in self.__dict__.items()})


def _files0(u):
    h = torch.where(u["sp_h"] > 0, u["sp_h"], torch.full_like(u["sp_h"], 1.5))
    return torch.minimum(torch.clamp(torch.floor(u["width"] / h + 1e-4), min=1), u["men0"].clamp(min=1))


def build(st, params, seed=0):
    """Fighters for every man of st (tools/nn/sim/state.State): at their places in their unit's formation, each with
    his passport's health; a unit that starts with fewer men than its men0 has the rest dead, and its missing health
    (men x HP a man > its HP) comes off its men as wounds spread evenly (conform)."""
    u = st.u
    dev = u["men"].device
    B, N = u["men"].shape
    present = (u["side"] > 0)
    men0 = torch.where(present, u["men0"].round(), torch.zeros_like(u["men0"])).long()
    M = max(1, int(men0.sum(1).max()))
    files0 = _files0(u).long()
    m0, f0 = men0.cpu().tolist(), files0.cpu().tolist()
    uid = torch.zeros((B, M), dtype=torch.long)
    k_in = torch.zeros((B, M), dtype=torch.long)
    valid = torch.zeros((B, M), dtype=torch.bool)
    file = torch.zeros((B, M), dtype=torch.long)
    rank0 = torch.zeros((B, M), dtype=torch.long)
    ustart = torch.full((B, N), M, dtype=torch.long)
    for b in range(B):
        pos = 0
        for n in range(N):
            c = m0[b][n]
            if c <= 0:
                continue
            ustart[b, n] = pos
            F = max(1, min(f0[b][n], c))
            k = torch.arange(c)
            R = (c + F - 1) // F
            rem = c - (R - 1) * F                       # men in the last rank
            r = k // F
            f = k % F
            last = r == R - 1
            f = torch.where(last, f + (F - rem) // 2, f)
            uid[b, pos:pos + c] = n
            k_in[b, pos:pos + c] = k
            valid[b, pos:pos + c] = True
            file[b, pos:pos + c] = f
            rank0[b, pos:pos + c] = r
            pos += c
    # file compaction's static order: by (unit, file, rank)
    key = torch.where(valid, (uid * 4096 + file) * 4096 + rank0, torch.full_like(uid, 2 ** 40))
    order = torch.argsort(key, dim=1)
    seg = torch.where(valid, uid * 4096 + file, torch.full_like(uid, -1)).gather(1, order)
    start = torch.ones_like(seg, dtype=torch.bool)
    start[:, 1:] = seg[:, 1:] != seg[:, :-1]
    idx = torch.arange(M).expand(B, M)
    seg_first = torch.cummax(torch.where(start, idx, torch.zeros_like(idx)), dim=1).values
    # the slot's scatter (the database's unit_spacings close_formation_spacing_variation: a man's place varies by up to
    # that many metres; uniform within +-half of it)
    scat = torch.tensor([[float(((params.units.get(st.keys[b][n]) or {}).get("spacing") or {}).get("scatter") or 0.0)
                          for n in range(N)] for b in range(B)])
    g = torch.Generator().manual_seed(seed)
    sc = scat.gather(1, uid)
    jit_a = (torch.rand((B, M), generator=g) - 0.5) * sc
    jit_f = (torch.rand((B, M), generator=g) - 0.5) * sc
    to = lambda t: t.to(dev)
    F = Fighters(uid=to(uid), k=to(k_in), valid=to(valid), file=to(file), rank0=to(rank0), ustart=to(ustart),
                 umen0=to(men0), files0=to(files0), order=to(order), seg_first=to(seg_first), jit_a=to(jit_a),
                 jit_f=to(jit_f), flip=torch.zeros((B, N), dtype=torch.bool, device=dev),
                 b_last=u["b"].clone(), cd=torch.zeros((B, M), device=dev),
                 tgt=torch.full((B, M), -1, dtype=torch.long, device=dev),
                 hp=torch.zeros((B, M), device=dev), dx=torch.zeros((B, M), device=dev),
                 dz=torch.zeros((B, M), device=dev), pend=torch.zeros((B, M), device=dev),
                 sh=torch.zeros((B, N), device=dev), down=torch.zeros((B, M), device=dev), aim=torch.full((B, M), -1, dtype=torch.long, device=dev),
                 gen=torch.Generator(device=dev).manual_seed(seed + 1), u_ref=u)
    hp_man = u["hp_man"].gather(1, F.uid)
    men_now = u["men"].round().long().gather(1, F.uid)
    F.hp = torch.where(F.valid & (F.k < men_now), hp_man, torch.zeros_like(hp_man))
    conform(F, u)
    sa, sf = slots(F, u)
    F.dx, F.dz = to_world(u, F, sa, sf)
    return F


def ugather(u, F, name):
    """A unit field [B, N] at each man [B, M]."""
    return u[name].gather(1, F.uid)


def alive(F):
    return F.valid & (F.hp > 0)


def unit_count(F, mask, N):
    """[B, N] the men of each unit for which mask [B, M] holds."""
    return torch.zeros((F.B, N), device=F.hp.device).scatter_add(1, F.uid, mask.float())


def unit_sum(F, v, N):
    return torch.zeros((F.B, N), device=F.hp.device, dtype=v.dtype).scatter_add(1, F.uid, v)


def rank_now(F, live):
    """Each man's rank now: the living men ahead of him in his file (the man behind steps up)."""
    a = live.gather(1, F.order).float()
    cs = torch.cumsum(a, 1) - a
    r = cs - cs.gather(1, F.seg_first)
    return torch.zeros_like(r).scatter(1, F.order, r)


def slots(F, u):
    """(across, along) [B, M]: each man's place in his unit's frame, m (along + the facing). Files keep the ordered
    width (the files0 of the passport's spacing h); ranks v apart, the front rank at the box's front (the simulator's
    geometry.dims: ranks = ceil(men / files)); a formation turned about (flip) mirrors its places; a lone man at the
    centre."""
    live = alive(F)
    N = u["men"].shape[1]
    h = torch.where(u["sp_h"] > 0, u["sp_h"], torch.full_like(u["sp_h"], 1.5))
    v = torch.where(u["sp_v"] > 0, u["sp_v"], torch.full_like(u["sp_v"], 1.5))
    n_live = unit_count(F, live, N)
    ranks = torch.ceil(n_live.clamp(min=1) / F.files0.float())
    r = rank_now(F, live)
    hg, vg = h.gather(1, F.uid), v.gather(1, F.uid)
    fg = F.files0.float().gather(1, F.uid)
    sa = (F.file.float() - (fg - 1) / 2) * hg + F.jit_a
    sf = (ranks.gather(1, F.uid) - 1) / 2 * vg - r * vg + F.jit_f
    flip = F.flip.gather(1, F.uid)
    sa = torch.where(flip, -sa, sa)
    sf = torch.where(flip, -sf, sf)
    lone = (u["men0"] <= 1).gather(1, F.uid)
    return torch.where(lone, torch.zeros_like(sa), sa), torch.where(lone, torch.zeros_like(sf), sf)


def to_world(u, F, sa, sf):
    """(dx, dz) along the map's axes from (across, along) in the unit's frame: along = facing (sin b, cos b), across =
    (cos b, -sin b) (geometry: the front runs along it)."""
    b = ugather(u, F, "b") * geometry.DEG
    s, c = torch.sin(b), torch.cos(b)
    return sa * c + sf * s, -sa * s + sf * c


def positions(F, u):
    """(x, z) [B, M] of every man."""
    return ugather(u, F, "x") + F.dx, ugather(u, F, "z") + F.dz


def unit_rank(F, score, mask):
    """[B, M] each man's rank within his unit by score ascending, among the men of mask (others rank after them)."""
    s = torch.where(mask, score, torch.full_like(score, 1e9))
    s = (s - s.amin(1, keepdim=True).clamp(max=0)).clamp(max=1e9)
    key = F.uid.double() * 2e9 + s.double()
    order = torch.argsort(key, dim=1)
    pos = torch.empty_like(order).scatter_(1, order, torch.arange(F.M, device=order.device).expand_as(order))
    first = torch.searchsorted(key.gather(1, order).contiguous(), (F.uid.double() * 2e9).contiguous())
    return pos - first


def conform(F, u):
    """Put the men to the unit state's men and health where they differ (a start with losses, the step replay's reset
    to the recording): the most wounded die first, the dead come back whole at the back; then the unit's missing
    health comes off (or goes back to) the living as wounds, by the wounds they have (evenly where none) - the lab's
    own spread of wounds carries over."""
    N = u["men"].shape[1]
    hp_man = ugather(u, F, "hp_man")
    live = alive(F)
    want = torch.where(u["side"] > 0, u["men"].round(), torch.zeros_like(u["men"])).clamp(min=0)
    want = torch.minimum(want, F.umen0.float())
    have = unit_count(F, live, N)
    extra = (have - want).clamp(min=0).gather(1, F.uid)
    lack = (want - have).clamp(min=0).gather(1, F.uid)
    # the most wounded of the living die
    r_live = unit_rank(F, F.hp, live)
    die = live & (r_live.float() < extra)
    # the dead come back (the highest-ranked dead first: the back of the formation)
    dead = F.valid & ~live
    r_dead = unit_rank(F, -F.rank0.float(), dead)
    back = dead & (r_dead.float() < lack)
    hp = torch.where(die, torch.zeros_like(F.hp), F.hp)
    hp = torch.where(back, hp_man, hp)
    live = F.valid & (hp > 0)
    # the health
    lone = (u["men0"] <= 1).gather(1, F.uid)
    tot = unit_sum(F, torch.where(live, hp, torch.zeros_like(hp)), N)
    goal = torch.where(u["side"] > 0, u["hp_abs"], torch.zeros_like(u["hp_abs"]))
    wound = torch.where(live, hp_man - hp, torch.zeros_like(hp))
    w_sum = unit_sum(F, wound, N)
    n_live = unit_count(F, live, N)
    cap = n_live * u["hp_man"] - goal                      # the wounds the unit should have
    cap = torch.maximum(cap, torch.zeros_like(cap))
    differs = (tot - goal).abs() > 0.5
    # new wound = cap x (w + eps) / sum(w + eps): the lab's spread kept, even where none
    eps = 1e-3 * u["hp_man"]
    share = (wound + eps.gather(1, F.uid)) / (w_sum + eps * n_live).clamp(min=1e-9).gather(1, F.uid)
    new_w = (cap.gather(1, F.uid) * share).clamp(max=hp_man - 1e-3)
    hp_w = torch.where(live & differs.gather(1, F.uid), hp_man - new_w, hp)
    hp_w = torch.where(lone & F.valid, torch.where(live, goal.gather(1, F.uid), hp_w), hp_w)
    F.hp = torch.where(F.valid, hp_w.clamp(min=0), torch.zeros_like(hp_w))
    # those who came back stand at their places (set by the caller's next slots); their clock is fresh
    F.cd = torch.where(back, torch.zeros_like(F.cd), F.cd)
    F.tgt = torch.where(back | die, torch.full_like(F.tgt, -1), F.tgt)
    return back


def write_back(F, u):
    """The unit state's men and health from the men (after a step)."""
    N = u["men"].shape[1]
    live = alive(F)
    present = u["side"] > 0
    men = unit_count(F, live, N)
    hp = unit_sum(F, torch.where(live, F.hp, torch.zeros_like(F.hp)), N)
    u["men"] = torch.where(present, men, u["men"])
    u["hp_abs"] = torch.where(present, hp, u["hp_abs"])
    u["hp"] = torch.where(present, hp / u["hp0"].clamp(min=1e-6), torch.zeros_like(hp))


def differs(F, u):
    """[B, N] True where the unit state's men or health are not the men's (an outside reset)."""
    N = u["men"].shape[1]
    live = alive(F)
    present = u["side"] > 0
    men = unit_count(F, live, N)
    hp = unit_sum(F, torch.where(live, F.hp, torch.zeros_like(F.hp)), N)
    return present & (((men - u["men"].round()).abs() > 0.5) | ((hp - u["hp_abs"]).abs() > 0.5))


def rotate_offsets(F, u, db_deg, mask):
    """Turn the men's offsets with their unit by db_deg [B, N] where mask [B, N] (a reset that turned the unit)."""
    a = (db_deg * geometry.DEG).gather(1, F.uid)
    m = mask.gather(1, F.uid)
    s, c = torch.sin(a), torch.cos(a)
    # a bearing grows clockwise (facing (sin b, cos b)): turning by a maps (dx, dz) -> (dx c + dz s, -dx s + dz c)
    nx = F.dx * c + F.dz * s
    nz = -F.dx * s + F.dz * c
    F.dx = torch.where(m, nx, F.dx)
    F.dz = torch.where(m, nz, F.dz)


def wrap_deg(a):
    return torch.remainder(a + 180.0, 360.0) - 180.0


__all__ = ["Fighters", "build", "slots", "to_world", "positions", "conform", "write_back", "differs", "alive",
           "unit_count", "unit_sum", "ugather", "rotate_offsets", "wrap_deg", "rank_now", "unit_rank", "math"]

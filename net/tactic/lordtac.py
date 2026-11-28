"""Our lord in the tactic level (tmp/drill_lord_t1/DRILL_v2.md sections 3-5, BUILD_NOTES.md): the shared stage code
every lord drill builds on - his facts at a tick, the code's masks over a lord player's output, the script players.
Off unless the level is given a lord player (lower.Tacticians(lord_player=), tactic/side.Side(lord_player=)): then
the infantry's path is bit for bit the same (pass 1 still gets the raw with the lord hidden; nothing here draws a
random number or writes an infantry store).

    facts(fx, S, ls, order, gate, busy, now)   View.lord {raw [r, RAW] his unhidden row, slot [r] (-1 none), has,
                 standing, engaged (net/unit/lord.engaged_of: his melee flag OR an edge gap <= scripts.TOUCH_M to a seen
                 standing enemy), gate, busy (the executor steps out), order {skill, tgt, born, age} (his ISSUED order,
                 after CANCEL), legal [r, K] (fx.valid - no enemy lord, not routing - with a row), bound [r, K]
                 (pin.pinned at orders.teach touch_m & fx.mt >= 2: the T0-1 teacher's 'bound'), dist / gap [r, K] (his
                 centre distance / edge gap to each row, m), eligible [r] (has & standing & gate & not engaged & not
                 busy & legal.any())}
    cancel(fx, S, ls, order, engaged)   [r] the CANCEL hygiene (BUILD_NOTES decision 1): an ATTACK on t whose t is
                 gone / unseen / routing, or with NO own non-routing unit of the group within orders.teach touch_m edge
                 gap of t, while he is not engaged - cleared at once (one tick: geometry does not flicker)
    masks(lv, k)   (delivered [r], target slot [r], counts): a player's request k [r] (a row, -1 KEEP) under the code's
                 masks - engaged: refused (hygiene; the issued order kept); not eligible or not legal: rejected
    players      a lord player = a callable view -> {"k": [r] long, the row he attacks, -1 KEEP} (the tactic's players'
                 shape; view.lord is facts' dict): IDLE never orders (he stays at the post); LT - DRILL_v2 section 5
                 core: eligible, no issued target, a bound legal enemy within orders.reach_m -> ATTACK the nearest bound
                 one (distance, then slot); LD1 - the same opportunities, the nearest legal one (loose included)
    PLAYERS / player(name)   the registry. THE SEAM for a network: a lord player of the same shape registered here
                 (PLAYERS["N"] = lambda: <its player>), its inputs read off view.lord and view.fx / view.inp
"""
import torch

from net.tactic import cfg
from net.tactic import raw as TR

INF = float("inf")


def c_lord():
    """config tactic.json orders.lord (absent: {})."""
    return cfg()["orders"].get("lord") or {}


def touch_m():
    return float(cfg()["orders"]["teach"]["touch_m"])


def _world_pair(S, ls, slot):
    """(dist, gap) [r, K] his centre distance and edge gap (sim/geometry.rect_gap) to the battle's slots slot [r, K]
    (-1: inf), from the unhidden raw S [r, N, RAW]."""
    from sim import geometry
    I = TR.I
    r = S.shape[0]
    k = torch.arange(r, device=S.device)
    c = lambda n: S[..., I[n]].float()                                                # noqa: E731
    gp = geometry.rect_gap(c("x"), c("z"), c("b"), c("front"), c("depth"))[k, ls.clamp(min=0)]       # [r, N]
    x, z = c("x"), c("z")
    lx, lz = x[k, ls.clamp(min=0)], z[k, ls.clamp(min=0)]
    d = torch.sqrt((x - lx[:, None]) ** 2 + (z - lz[:, None]) ** 2)                   # [r, N]
    sc = slot.clamp(min=0)
    ok = (slot >= 0) & (ls >= 0)[:, None]
    return (torch.where(ok, d.gather(1, sc), torch.full_like(sc, INF, dtype=d.dtype)),
            torch.where(ok, gp.gather(1, sc), torch.full_like(sc, INF, dtype=gp.dtype)))


def cancel(fx, S, ls, order, engaged):
    """(module doc) [r] bool."""
    att = (ls >= 0) & (order["skill"] == 2) & (order["tgt"] >= 0)
    t_row = fx.row_of_slot(order["tgt"][:, None])[:, 0]
    tc = t_row.clamp(min=0)
    ok_t = (t_row >= 0) & fx.valid.gather(1, tc[:, None])[:, 0]
    holder = fx.mine & ~fx.r                                                            # [r, K] own, not routing
    g = fx.gap.gather(2, tc[:, None, None].expand(-1, fx.K, 1))[..., 0]                  # [r, K] gap(row a, t)
    near = (holder & (g <= touch_m())).any(1)
    return att & (~ok_t | ~near) & ~engaged


def facts(fx, S, ls, order, gate, busy, now):
    """(module doc) the View.lord dict."""
    from net.tactic import pin as PN
    from net.unit import lord as LD
    I = TR.I
    r = S.shape[0]
    k = torch.arange(r, device=S.device)
    has = ls >= 0
    L = torch.where(has[:, None], S[k, ls.clamp(min=0)], torch.zeros_like(S[:, 0]))
    standing = has & (L[:, I["alive"]] > 0) & ~(L[:, I["r"]] > 0) & ~(L[:, I["s"]] > 0)
    own = torch.where(has, L[:, I["side"]].round().long(), torch.ones_like(ls))
    engaged = LD.engaged_of(S[..., :TR.BASE], ls.clamp(min=0), own) & has
    dist, gap = _world_pair(S, ls, fx.slot)
    legal = fx.valid & (fx.slot >= 0) & has[:, None]
    bound = PN.pinned(fx, touch=touch_m()) & (fx.mt >= 2)
    eligible = standing & gate & ~engaged & ~busy & legal.any(1)
    age = torch.where(order["tgt"] >= 0, (now - order["born"]).clamp(min=0.0), torch.zeros_like(order["born"]))
    return {"raw": L, "slot": ls, "has": has, "standing": standing, "engaged": engaged, "gate": gate, "busy": busy,
            "order": {**order, "age": age}, "legal": legal, "bound": bound, "dist": dist, "gap": gap,
            "eligible": eligible, "slot_of_rows": fx.slot}


def masks(lv, k, fx):
    """(module doc) -> (delivered [r], slot [r] of the delivered target (-1 none), counts {name: [r]})."""
    o = lv["order"]
    kc = k.clamp(min=0)
    slot = torch.where(k >= 0, fx.slot.gather(1, kc[:, None])[:, 0], torch.full_like(k, -1))
    req = lv["has"] & (k >= 0) & ((slot != o["tgt"]) | (o["skill"] != 2))
    leg = (k >= 0) & lv["legal"].gather(1, kc[:, None])[:, 0]
    refused = req & lv["engaged"]
    ok = req & ~refused & lv["eligible"] & leg
    rejected = req & ~refused & ~ok
    return ok, torch.where(ok, slot, torch.full_like(slot, -1)), {
        "lord_requests": req.float(), "lord_refused": refused.float(), "lord_rejected": rejected.float(),
        "lord_orders": ok.float()}


def _nearest(lv, cand):
    """[r] the row of cand [r, K] nearest him (distance, then the battle's slot), -1 none."""
    d = torch.where(cand, lv["dist"], torch.full_like(lv["dist"], INF))
    mn = d.amin(1, keepdim=True)
    tie = cand & (d == mn)
    slot = torch.where(tie, lv["slot_of_rows"], torch.full_like(lv["slot_of_rows"], 10 ** 6))
    row = slot.argmin(1)
    return torch.where(cand.any(1), row, torch.full_like(row, -1))


class Idle:
    """IDLE: never orders - he stays at the post (the parent's lord, the 'post' control)."""
    name = "IDLE"

    def __call__(self, v):
        return {"k": torch.full(v.lord["slot"].shape, -1, dtype=torch.long, device=v.lord["slot"].device)}


class LT:
    """LT (DRILL_v2 section 5, core): no memory, reads the issued order (after CANCEL)."""
    name = "LT"

    def options(self, v):
        """[r, K] the rows he may be sent at and [r] the opportunity (eligible, no issued target, a bound one)."""
        lv = v.lord
        reach = float(cfg()["orders"]["reach_m"])
        near = lv["legal"] & (lv["dist"] <= reach)
        bound = near & lv["bound"]
        opp = lv["eligible"] & (lv["order"]["tgt"] < 0) & bound.any(1)
        return near, bound, opp

    def __call__(self, v):
        near, bound, opp = self.options(v)
        k = _nearest(v.lord, bound)
        return {"k": torch.where(opp, k, torch.full_like(k, -1))}


class LD1(LT):
    """LD1 (DRILL_v2 section 5's price arm): LT's opportunities, the nearest legal enemy (loose included)."""
    name = "LD1"

    def __call__(self, v):
        near, bound, opp = self.options(v)
        k = _nearest(v.lord, near)
        return {"k": torch.where(opp, k, torch.full_like(k, -1))}


PLAYERS = {"IDLE": Idle, "LT": LT, "LD1": LD1}


def player(name):
    """A fresh lord player of PLAYERS by name (None / 'off': None - no lord in the level)."""
    if name in (None, "", "off"):
        return None
    p = PLAYERS[name]()
    p.name = name
    return p

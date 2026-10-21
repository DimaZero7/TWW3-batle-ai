"""The task's masks (S28; docs/en/training/general.md "Task masks"): the mission of a unit's group cuts the executor's
choices - the executor is not talked into its mission by a bonus, it is limited to it (GeneralConfig.task_mask, run.py
--general-task-mask; 0: no mask, the executor computes what it did to the bit).

Masks are logits set to the heads' NEG before the softmax (as the general's kind x target), per own unit with a task,
on the chained heads' parts (tools/nn/model/chain.py: kind -> target -> sector -> fine cell):

    withdraw  no attack (the kind closed; keep closed too while the order in force is an attack); the sectors (move,
              withdraw) only those whose centre is no nearer the nearest seen enemy than the unit is now, or nearer the
              mission's point than the unit while AWAY_M from that enemy
    shoot     a unit without missiles (the token's ammo 0): no attack (and no keep of an attack in force) - the
              simulator's attack order of a shooter is to fire from its range (sim battle.py close_in), of the others
              a melee; a shooter's attack only at the target group's units; moving free
    attack,   the attack only at the target group's units (the visible ones; none visible: no target mask); the
    flank     sectors not farther from the target group's centre than the unit is now (at least ATTACK_M), so a
              charge is never cut
    hold      the sectors within the reach of the mission's point (mission.py: a tile's half side, GROUP_M of a group);
              the attack only at enemies within HOLD_FIGHT x the reach of the point (where the measure is 0 beyond)
    screen    the sectors near the segment from the target group's centre to the seen enemy nearest to it (within
              mission.SCREEN_M of it, between its ends)
Each piece is a part (PARTS; GeneralConfig.task_mask_parts, run.py --general-task-mask-parts); by default only those
that cost nothing at minute 0 (config.MASK_PARTS: the movement ones and the general's shoot_group - the ones that forbid
fighting, withdraw_attack, attack_target, shoot_*, cost g13 0.2-1.2 of the rating; docs "Task masks").
A sector counts by its centre, with half its diagonal of slack (a 100 m sector: 71 m): the fine cell is free inside it.
The target pointer is shared with the spells (chain.py): a target mask limits a unit's spells the same way.

Fall-back: a part whose mask closes every choice the unit had is not masked (mask_fallback in the log); a target mask
that leaves the attack no target closes the attack (hold: nothing near the point to fight) - except attack / flank /
shoot, whose target group not seen leaves the targets free (counted as a fall-back).

The task's numbers per unit come with the observation (Runtime.task: "task_mk" the mission kind or -1, "task_pt" the
mission's point in the side's frame (m), "task_reach" its reach (0: no point), "task_tg" the target index, "task_main"
every slot's group number), so the rollout and the update compute the same masks from the stored decision (PPO's old
log-probability is of the masked choice). The game (tools/nn/companion) gets them the same way.
"""
import math

import torch

from tools.nn.general import config as gconfig
from tools.nn.general import mission as gmission
from tools.nn.model import chain
from tools.nn.model import observation as ob
from tools.nn.model import sectors as sc
from tools.nn.model.heads import NEG
from tools.nn.sim.orders import ATTACK, KEEP, KINDS, MOVE, WITHDRAW

KIND = {k.name: k.index for k in gconfig.MISSION_KINDS}
ATTACK_M = 150.0         # attack / flank: the sectors may come up to this far from the target group's centre at least
HOLD_FIGHT = 3.0         # hold: the attack only at enemies within this x the reach of the point (the measure's 0)
AWAY_M = gmission.AWAY_M  # withdraw: towards the point only this far from the nearest enemy
KEYS = ("task_mk", "task_pt", "task_reach", "task_tg", "task_main")
# the masks' parts (GeneralConfig.task_mask_parts, run.py --general-task-mask-parts): each one alone switchable
PARTS = ("withdraw_attack",   # withdraw: no attack (nor keep of an attack in force)
         "withdraw_move",     # withdraw: the sectors away from the nearest enemy (or towards the point)
         "shoot_melee",       # shoot: a unit without missiles no attack (nor keep of one)
         "shoot_target",      # shoot: a shooter's attack only at the target group
         "attack_target",     # attack / flank: the attack only at the target group
         "attack_move",       # attack / flank: the sectors not farther from the target group
         "hold_target",       # hold: the attack only at enemies near the point
         "hold_move",         # hold: the sectors within the point's reach
         "screen_move",       # screen: the sectors near the segment
         "shoot_group")       # the general: "shoot" only for a group of shooters (net.General.shoot_ok)


def parse_parts(text):
    """frozenset of PARTS of a comma list ("all": every part; "" or "none": none); an unknown name: ValueError."""
    text = (text or "").strip()
    if text == "all":
        return frozenset(PARTS)
    names = [t.strip() for t in text.split(",") if t.strip() and t.strip() != "none"]
    bad = [n for n in names if n not in PARTS]
    if bad:
        raise ValueError(f"task mask parts {bad}: not in {', '.join(PARTS)}")
    return frozenset(names)


def parts_of(gcfg):
    """The parts in force of a GeneralConfig (none without task_mask)."""
    if not getattr(gcfg, "task_mask", False):
        return frozenset()
    return parse_parts(getattr(gcfg, "task_mask_parts", "all"))


def fields(kind_u, point, reach, target_u, main, on):
    """The task's numbers of the observation (module doc): kind_u [R, N] the unit's group's mission kind, point [R, N, 2]
    its point in the side's frame (m), reach [R, N] (0: no point), target_u [R, N] the target index, main [R, N] the
    plan's group numbers of every slot, on [R, N] the units with a task."""
    small = torch.int8
    return {"task_mk": torch.where(on, kind_u, torch.full_like(kind_u, -1)).to(small),
            "task_pt": torch.where(on[..., None], torch.nan_to_num(point), torch.zeros_like(point)).float(),
            "task_reach": torch.where(on, reach, torch.zeros_like(reach)).float(),
            "task_tg": torch.where(on, target_u, torch.full_like(target_u, -1)).to(small),
            "task_main": main.to(small)}


def has(obs):
    return obs.get("task_mk") is not None


def _d(ax, az, bx, bz):
    """Element-wise distance (no matmul: the rollout's and the update's batches give the same bits)."""
    return ((ax - bx) ** 2 + (az - bz) ** 2).sqrt()


@torch.no_grad()
def masks(cfg, obs, sec_ok, parts=None):
    """{"kind" [B, N, 5], "target" [B, N, N], "sector" [B, N, S], "on" [B, N]} the task's allowed choices (True: allowed;
    to AND with the heads' own masks) and the units under a mask, of the observation obs (with KEYS) and the sectors
    inside the map sec_ok [B, S]; "fb_target" [B, N] the units whose target group is not seen (targets left free).
    parts: the PARTS in force (None: every one); a unit counts as under a mask ("on") when a part of its kind is."""
    P = frozenset(PARTS) if parts is None else parts
    mk = obs["task_mk"].long()
    ctrl = obs["ctrl"]
    free = obs.get("free", ctrl)
    on = (mk >= 0) & ctrl
    pos = obs["pos"].float() * ob.POS
    px, pz = pos[..., 0], pos[..., 1]
    foe = obs["target_ok"]                                                    # [B, N] the enemies one may attack
    tok = obs["tokens"]
    shooter = tok[..., ob.INDEX["ammo"]].float() > 0
    att_now = tok[..., ob.INDEX["order_attack"]].float() > 0.5
    pt = obs["task_pt"].float()
    tx, tz = pt[..., 0], pt[..., 1]
    reach = obs["task_reach"].float()
    has_pt = reach > 0
    tg, main = obs["task_tg"].long(), obs["task_main"].long()
    is_k = (lambda n: on & (mk == KIND[n]))
    att, hold, scr, sh, wd = (is_k("attack") | is_k("flank"), is_k("hold"), is_k("screen"), is_k("shoot"),
                              is_k("withdraw"))
    use = (lambda *names: any(n in P for n in names))
    att, hold, scr = att & use("attack_target", "attack_move"), hold & use("hold_target", "hold_move"), scr & use("screen_move")
    sh, wd = sh & use("shoot_melee", "shoot_target"), wd & use("withdraw_attack", "withdraw_move")
    on = att | hold | scr | sh | wd                                           # the units under a part in force
    big = torch.full(px.shape + px.shape[-1:], 1e9, device=px.device)                          # [B, N, N]
    # targets: the target group's units (attack, flank, a shooter's shoot); hold: the enemies near the point
    member = (main[:, None, :] == tg[:, :, None]) & (tg[:, :, None] >= 0) & foe[:, None, :]       # [B, N (i), N (j)]
    by_group = (att if "attack_target" in P else torch.zeros_like(att)) | (sh & shooter & ("shoot_target" in P))
    seen_t = member.any(-1)
    dPj = _d(px[:, None, :], pz[:, None, :], tx[:, :, None], tz[:, :, None])                     # enemy j - i's point
    near_p = (dPj <= HOLD_FIGHT * reach[..., None]) & foe[:, None, :]
    tgt = torch.ones_like(member)
    tgt = torch.where((by_group & seen_t)[..., None], member, tgt)
    tgt = torch.where((hold & has_pt & ("hold_target" in P))[..., None], near_p, tgt)
    # kinds: no attack under withdraw and a shoot without missiles (nor keep of an attack in force)
    no_attack = (wd & ("withdraw_attack" in P)) | (sh & ~shooter & ("shoot_melee" in P))
    K = len(KINDS)
    ar = torch.arange(K, device=mk.device)
    kind = ~(((ar == ATTACK) & no_attack[..., None]) | ((ar == KEEP) & (no_attack & att_now & free)[..., None]))
    # sectors (their centres in the side's frame, with half a diagonal of slack)
    c = sc.centres(cfg, mk.device)
    cx, cz = c[:, 0], c[:, 1]
    h = sc.step(cfg) * math.sqrt(0.5)
    dcP = _d(cx, cz, tx[..., None], tz[..., None])                                                 # [B, N, S]
    duP = _d(px, pz, tx, tz)
    D = _d(px[:, :, None], pz[:, :, None], px[:, None, :], pz[:, None, :])                         # [B, N, N]
    dE, jE = torch.where(foe[:, None, :], D, big).min(-1)                                         # the nearest enemy
    any_foe = foe.any(-1, keepdim=True).expand_as(dE)
    ex, ez = px.gather(1, jE), pz.gather(1, jE)
    dcE = _d(cx, cz, ex[..., None], ez[..., None])
    sec = torch.ones_like(dcP, dtype=torch.bool)
    s_att = dcP <= torch.maximum(duP, torch.full_like(duP, ATTACK_M))[..., None] + h
    s_hold = dcP <= reach[..., None] + h
    # screen: the segment from the point A (the target group's centre) to the seen enemy nearest to A
    _, jA = torch.where(foe[:, None, :], dPj, big).min(-1)
    vx, vz = px.gather(1, jA) - tx, pz.gather(1, jA) - tz
    vn = (vx * vx + vz * vz).sqrt().clamp(min=1.0)
    wx, wz = cx - tx[..., None], cz - tz[..., None]
    along = (wx * vx[..., None] + wz * vz[..., None]) / (vn * vn)[..., None]
    lat = (wx * vz[..., None] - wz * vx[..., None]).abs() / vn[..., None]
    tol = (h / vn)[..., None]
    s_scr = (lat <= gmission.SCREEN_M + h) & (along >= -tol) & (along <= 1.0 + tol)
    away = dcE >= dE[..., None] - h
    toward = (dcP <= duP[..., None]) & (dcE >= AWAY_M) & has_pt[..., None]
    s_wd = away | toward
    sec = torch.where((att & has_pt & ("attack_move" in P))[..., None], s_att, sec)
    sec = torch.where((hold & has_pt & ("hold_move" in P))[..., None], s_hold, sec)
    sec = torch.where((scr & has_pt & any_foe)[..., None], s_scr, sec)
    sec = torch.where((wd & any_foe & ("withdraw_move" in P))[..., None], s_wd, sec)
    del sec_ok   # (the heads AND it in; the fall-back is theirs)
    return {"kind": kind, "target": tgt, "sector": sec, "on": on, "fb_target": by_group & ~seen_t}


def apply(m, ctrl, free, ok, sec_ok):
    """The heads' masks with the task's m (masks()): -> (allowed kinds [B, N, 5], targets [B, N, N], sectors [B, N, S],
    fall-back [B, N] where a part's mask closed every choice and was dropped). ok [B, N] the heads' targets (target_ok),
    sec_ok [B, S] the sectors inside the map. The kinds' order: tools/nn/sim/orders.py KINDS."""
    base = torch.stack([~ctrl | free, free, free & ok.any(-1, keepdim=True), free, ctrl], -1)
    t_ok = ok[:, None, :] & m["target"]
    task_k = torch.stack([~ctrl | free, free, free & t_ok.any(-1), free, ctrl], -1) & m["kind"]
    k_fb = ~task_k.any(-1)
    kind = torch.where(k_fb[..., None], base, task_k)
    s_ok = sec_ok[:, None, :] & m["sector"]
    s_fb = ~s_ok.any(-1) & sec_ok.any(-1, keepdim=True)
    sec = torch.where(s_fb[..., None], sec_ok[:, None, :].expand_as(s_ok), s_ok)
    fb = m["on"] & (k_fb | s_fb | m["fb_target"])
    return kind, t_ok, sec, fb


class MaskedChainHeads(chain.ChainHeads):
    """chain.ChainHeads with the task's masks (module doc): the same weights and the same computation, the kinds, the
    targets and the sectors also cut per unit where the observation has the task's numbers (KEYS); without them
    every output is chain.ChainHeads's to the bit (test_general_taskmask_torch.py). Its forward is ChainHeads.forward
    with three lines more (tools/nn/model is the simulator's version: not edited for the general; a test holds the
    copy to the original's source)."""

    def forward(self, x, s, cells_in, obs_t, action=None, greedy=False, temperature=1.0, abilities=True, pairs=None):
        cfg = self.cfg
        u = self.norm(x[:, 1:])
        ctrl, ok = obs_t["ctrl"], obs_t["target_ok"]
        free = obs_t.get("free", ctrl)
        sec_ok = cells_in.any(-1)                                                 # [B, S]
        tm = masks(cfg, obs_t, sec_ok, getattr(self, "parts", None)) if has(obs_t) else None
        # hold, move, attack, withdraw, keep (tools/nn/sim/orders.py KINDS order); a held unit only keeps
        allowed = torch.stack([~ctrl | free, free, free & ok.any(-1, keepdim=True), free, ctrl], -1)
        t_ok = ok[:, None, :]
        sec_m = sec_ok[:, None, :]
        if tm is not None:
            allowed, t_ok, sec_m, _ = apply(tm, ctrl, free, ok, sec_ok)
        bad = torch.zeros(u.shape[0], device=u.device)
        kind, bad = chain._cap(self.kind(u), bad)
        kind = kind.masked_fill(~allowed, NEG)
        target = self.q(u) @ self.k(u).transpose(1, 2) / math.sqrt(cfg.pointer)
        if pairs is not None and getattr(self, "target_pair", None) is not None:
            target = target + self.target_pair(pairs[:, 1:, 1:])[..., 0].to(target.dtype)
        target, bad = chain._cap(target, bad)
        target = target.masked_fill(~t_ok, NEG)
        a_kind = chain._pick(kind, greedy, temperature) if action is None else action.kind
        has_target = target.max(-1).values > NEG / 2
        if action is None:
            a_target = torch.where((a_kind == ATTACK) & has_target, chain._pick(target, greedy, temperature),
                                   torch.full_like(a_kind, -1))
        else:
            a_target = action.target
        attack = (a_kind == ATTACK) & (a_target >= 0)
        tvec = u.gather(1, a_target.clamp(min=0)[..., None].expand(-1, -1, u.shape[-1])) * attack[..., None].to(u.dtype)
        c = self.cond(u + self.kind_emb(a_kind) + self.target_emb(tvec))
        # the place: a sector, then a fine cell of it
        sector = self.place_q(c) @ self.place_k(s).transpose(1, 2) / math.sqrt(cfg.sector_d)
        pos = obs_t["pos"].to(u.dtype) * sc.ob.POS
        dist = (pos[..., None, :] - sc.centres(cfg, u.device).to(u.dtype)).square().sum(-1).clamp(min=1e-6).sqrt()
        sector, bad = chain._cap(sector - torch.nn.functional.softplus(self.place_near(c)) * dist / sc.step(cfg), bad)
        sector = sector.masked_fill(~sec_m, NEG)
        if action is None:
            a_sector = chain._pick(sector, greedy, temperature)
        else:
            a_sector, a_fine_given = sc.split(cfg, action.point)
        idx = a_sector[..., None].expand(-1, -1, s.shape[-1])
        s_at = s.gather(1, idx)                                                   # [B, N, ds]
        fine, bad = chain._cap(self.fine(torch.cat([c, s_at], -1)), bad)
        fine_ok = cells_in.gather(1, a_sector[..., None].expand(-1, -1, cells_in.shape[-1]))
        fine = fine.masked_fill(~fine_ok, NEG)
        a_fine = chain._pick(fine, greedy, temperature) if action is None else a_fine_given
        commit, bad = chain._cap(self.commit(c), bad)
        run, bad = chain._cap(self.run(c)[..., 0], bad)
        out = {"kind": kind, "target": target, "sector": sector, "fine": fine, "commit": commit, "run": run}
        if action is None:
            a_commit = chain._pick(commit, greedy, temperature)
            p_run = torch.sigmoid(chain._clean(run, temperature))
            a_run = (run > 0) if greedy else torch.rand_like(p_run) < p_run
        else:
            a_commit, a_run = action.commit, action.run
        a_ability = None
        if obs_t.get("abil") is not None:
            usable = obs_t["abil_ok"] & ctrl[..., None]                                   # [B, N, ACTIVE]
            score = self._ability_scores(u, obs_t, usable)
            score, bad = chain._cap(score, bad)
            slot = torch.where(usable, score, torch.full_like(score, NEG))
            none, bad = chain._cap(self.ability_none(u), bad)
            out["ability"] = torch.cat([none, slot], -1)
            if action is None and abilities:
                a_ability = chain._pick(out["ability"], greedy, temperature) - 1
                cast = (a_ability >= chain.ab.SPELL_AT) & (a_target < 0) & has_target
                a_target = torch.where(cast, chain._pick(target, greedy, temperature), a_target)
            elif action is not None:
                a_ability = action.ability
        point = sc.point_index(cfg, a_sector, a_fine)
        out["nan_fixed"] = bad                     # [B] non-finite logits of the row (taken as masked)
        return out, chain.Action(a_kind, point, a_target, a_run, a_ability, a_commit)


# ChainHeads.forward's source this copy follows (source_sha(); test_general_taskmask_torch.py: a change there is to be
# copied here, then this updated)
SOURCE_SHA = "09b0aa0586717d69"


def source_sha():
    """The first 16 hex of sha256 of chain.ChainHeads.forward's source."""
    import ast
    import hashlib
    src = open(chain.__file__, encoding="utf-8").read()
    for n in ast.parse(src).body:
        if isinstance(n, ast.ClassDef) and n.name == "ChainHeads":
            for f in n.body:
                if isinstance(f, ast.FunctionDef) and f.name == "forward":
                    seg = ast.get_source_segment(src, f, padded=True)      # (read as text: one line end)
                    return hashlib.sha256(seg.encode()).hexdigest()[:16]
    return None

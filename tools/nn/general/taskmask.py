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
    shoot     the volley lock (shoot_volley; volley_lock(), Runtime.task keeps its clock): a shooter that fires or aims
              (the token's firing: the simulator's flag and the game's IsFiringMissiles are on from the moment a
              standing shooter has a target in range and aims at it) only keeps its order - no move, no other target,
              no hold - from the first decision it fires until a volley is seen (the token's volley_ready: the ammo
              fell at this decision), or at most aim + VOLLEY_SLACK x the passport's reload (the game's database) +
              one decision; the decision of the volley is free (an order there costs least: the men reload on), then
              the next cycle locks again. Not in melee, nor with a seen enemy within VOLLEY_FREE_M (the unit may step
              back or turn): every order is a new aim in the game and in the simulator (missile.aim_reset_on_order)
    any       melee_stay (GeneralConfig.melee_stay; the general's rule net.melee_ok and the teacher's): a unit in melee
              (the token's flag) and not wavering or worse, of a group that holds (fighting(): under waver_share of
              its units wavering or worse) - or of any group, its mission not "withdraw" - gets no move, no withdraw
              and no keep of a move / withdraw in force, nor of an attack in force on an enemy it does not fight
              ("task_otgt": the order's target slot, Runtime.task; not among the enemies below, or not known: closed
              - in the simulator, battle.py contact.attack_leave, a unit in melee attacking an enemy it does not
              touch walks off and strikes nobody up to the 24 s breakoff); its attack only at the seen enemies in
              melee within STAY_M of it (those it fights; none: no attack - hold or keep). The human's battle against g34: a hold led a
              fighting unit back to its point (22 orders), a withdraw of a group with one wavering unit of 8-14 led
              its steady ones out of melee (63 of 72 withdraw orders)
    shoot     shoot_line (GeneralConfig.shoot_line_m > 0): while our infantry groups hold (a group under "hold", under
              half of it shooters - the teacher's wait for the farthest group, or the general's own hold), a unit
              under "shoot" goes no farther forward (the side's frame) than the most forward centre of those groups
              + shoot_line_m (Runtime.task: "task_line", LINE_NONE where none): the sectors that reach behind that
              line, the fine cells behind it (MaskedChainHeads), the attack of a shooter only at enemies in its range
              now or from the line (their forward within the line + its range: it stops behind the line; in range
              now alone cut the general's fire by ~70 % in the check) - of a unit without missiles: at enemies behind
              the line, no keep of an attack in force while the unit
              walks and does not fire, nor of a move beyond the line (a standing shooter keeps its attack: a new
              order resets its aim; closing that keep cut the general's fire by ~70 % in the check). The human's battle against g34: the infantry
              waited 66 s, the shooting groups went 107-140 m ahead of it alone and took the blow first
Each piece is a part (PARTS; GeneralConfig.task_mask_parts, run.py --general-task-mask-parts); by default only those
that cost nothing at minute 0 (config.MASK_PARTS: the movement ones, the general's shoot_group and the volley lock -
the ones that forbid fighting, withdraw_attack, attack_target, shoot_melee/target, cost g13 0.2-1.2 of the rating;
docs "Task masks").
A sector counts by its centre, with half its diagonal of slack (a 100 m sector: 71 m): the fine cell is free inside it.
The target pointer is shared with the spells (chain.py): a target mask limits a unit's spells the same way.

Fall-back: a part whose mask closes every choice the unit had is not masked (mask_fallback in the log); a target mask
that leaves the attack no target closes the attack (hold: nothing near the point to fight) - except attack / flank /
shoot, whose target group not seen leaves the targets free (counted as a fall-back).

The task's numbers per unit come with the observation (Runtime.task: "task_mk" the mission kind or -1, "task_pt" the
mission's point in the side's frame (m), "task_reach" its reach (0: no point), "task_tg" the target index, "task_main"
every slot's group number, "task_lock" the volley lock), so the rollout and the update compute the same masks from the
stored decision (PPO's old log-probability is of the masked choice). The game (tools/nn/companion) gets them the same way.
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
KEYS = ("task_mk", "task_pt", "task_reach", "task_tg", "task_main", "task_lock", "task_line", "task_otgt")
LINE_NONE = 1e4          # "task_line" where no line holds the unit (m: beyond any map)
# the volley lock (shoot_volley, volley_lock()): a volley seen at this decision (volley_ready x reload under this, s)
# frees the decision; the simulator's men load fully only after ~1.1-1.3x the passport's reload (observation.py VOLLEY):
# the lock's longest cycle aim + VOLLEY_SLACK x reload + one decision (a shooter that aims and never fires is let go);
# a seen enemy within VOLLEY_FREE_M of the unit (centre to centre) frees it (a mask's rule, not a game number)
VOLLEY_OPEN_S = 0.5
VOLLEY_SLACK = 1.3
VOLLEY_FREE_M = 50.0
VOLLEY_STEP_S = 1.0
STAY_M = 60.0            # melee_stay: the enemies in melee this close (centre to centre) are those the unit fights (a
#                          line's unit is 30-40 m wide, a contact from a flank ~40 m; a mask's rule, not a game number)
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
         "shoot_group",       # the general: "shoot" only for a group of shooters (net.General.shoot_ok)
         "shoot_volley",      # shoot: a firing shooter only keeps its order until its volley (volley_lock)
         "melee_stay",        # any: a steady unit in melee does not move off (on with GeneralConfig.melee_stay)
         "shoot_line")        # shoot: not ahead of our holding infantry (on with GeneralConfig.shoot_line_m > 0)


def fighting(mem, melee, waver, share=0.5, alive=None):
    """[P, G] the groups that fight in melee and hold (GeneralConfig.melee_stay: no "withdraw" for them): some member
    in melee, under `share` (GeneralConfig.waver_share) of its members wavering or worse. mem [P, N, G] the groups'
    members, melee [P, N] the units in melee (the state's melee flag), waver [P, N] wavering or worse (the morale state
    >= 5: units.WAVERING_MS); alive [P, N, G] the members the share counts (with the routing ones; None: mem).
    Net.melee_ok, the teacher and the executor's melee_stay read the same rule."""
    alive = mem if alive is None else alive
    n = alive.sum(1).float()
    n_w = (alive & waver.bool()[..., None]).sum(1).float()
    return (mem & melee.bool()[..., None]).any(1) & (n_w < share * n)


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
    """The parts in force of a GeneralConfig (none without task_mask): task_mask_parts, and melee_stay with
    GeneralConfig.melee_stay and shoot_line with GeneralConfig.shoot_line_m > 0 (rule fields: an older checkpoint gets
    them too, whatever its saved parts)."""
    if not getattr(gcfg, "task_mask", False):
        return frozenset()
    p = parse_parts(getattr(gcfg, "task_mask_parts", "all"))
    if getattr(gcfg, "melee_stay", False):
        p = p | {"melee_stay"}
    if getattr(gcfg, "shoot_line_m", 0.0) > 0:
        p = p | {"shoot_line"}
    return p


def fields(kind_u, point, reach, target_u, main, on, lock=None, line=None, otgt=None):
    """The task's numbers of the observation (module doc): kind_u [R, N] the unit's group's mission kind, point [R, N, 2]
    its point in the side's frame (m), reach [R, N] (0: no point), target_u [R, N] the target index, main [R, N] the
    plan's group numbers of every slot, on [R, N] the units with a task; lock [R, N] the volley lock (volley_lock();
    None: nobody); line [R, N] the forward limit of shoot_line (m in the side's frame; None or LINE_NONE: none); otgt
    [R, N] the slot of the attack target of each unit's order in force (the state's order_target; -1 none; None: not
    known - melee_stay then closes the keep of an attack in force)."""
    small = torch.int8
    lock = torch.zeros_like(on) if lock is None else lock & on
    line = torch.full_like(reach, LINE_NONE) if line is None else torch.where(on, line, torch.full_like(line, LINE_NONE))
    return {"task_mk": torch.where(on, kind_u, torch.full_like(kind_u, -1)).to(small),
            "task_pt": torch.where(on[..., None], torch.nan_to_num(point), torch.zeros_like(point)).float(),
            "task_reach": torch.where(on, reach, torch.zeros_like(reach)).float(),
            "task_tg": torch.where(on, target_u, torch.full_like(target_u, -1)).to(small),
            "task_main": main.to(small), "task_lock": lock.bool(), "task_line": line.float(),
            **({} if otgt is None else {"task_otgt": torch.where(on, otgt.long(), torch.full_like(otgt.long(), -1))
                                        .to(torch.int16)})}


def _missile_column(name, scale):
    """The token's column of the passport's missile `name` / scale (passport.features: found by its value, not its
    place)."""
    from tools.nn.model import passport
    base = {"men": 1, "hp_total": 1, "hp_per_man": 1, "mass": 1, "armour": 0, "leadership": 0, "multiplayer_cost": 0,
            "caste": "", "size": "", "speed": {"walk": 0, "run": 0, "charge": 0},
            "melee": {"attack": 0, "defence": 0, "charge_bonus": 0, "damage": 0, "ap_damage": 0, "bonus_v_large": 0,
                      "bonus_v_infantry": 0, "attack_interval_s": 0, "splash_max_attacks": 1},
            "shield": {"missile_block_chance": 0}}
    a = passport.features(dict(base, missile={"range_m": 100, "reload_s": 1}))
    b = passport.features(dict(base, missile=dict({"range_m": 100, "reload_s": 1}, **{name: scale})))
    (i,) = [k for k in range(len(a)) if a[k] != b[k]]
    assert abs(b[i] - 1.0) < 1e-6, f"passport.features: the scale of {name} changed"
    return ob.INDEX[f"passport_{i}"]


RELOAD_SCALE = 20.0
RELOAD_COL = _missile_column("reload_s", RELOAD_SCALE)
RANGE_SCALE = 500.0
RANGE_COL = _missile_column("range_m", RANGE_SCALE)


def cycle_s(tok):
    """([.., N] the passport's reload_s, [.., N] the volley lock's longest cycle, s) of the tokens [.., N, TOKEN]: aim
    (the larger of the first aim and the silence after a new target, passport.aim) + VOLLEY_SLACK x reload + one
    decision."""
    from tools.nn.model import passport
    reload = tok[..., RELOAD_COL].float() * RELOAD_SCALE
    aim = torch.maximum(tok[..., ob.INDEX["aim_first"]].float(), tok[..., ob.INDEX["aim_retarget"]].float())
    return reload, aim * passport.AIM_S + VOLLEY_SLACK * reload + VOLLEY_STEP_S


@torch.no_grad()
def volley_lock(obs, shoot, t, t0):
    """(lock [R, N], t0 [R, N]) the volley lock (module doc) at the battle time t [R] (s) of the observation obs (pos,
    tokens, ctrl, target_ok) for the units under "shoot" (shoot [R, N]), from the clock t0 [R, N] (when the unit's
    lock cycle began, -1: none; Runtime keeps it). Free at a volley seen now (volley_ready x reload <= VOLLEY_OPEN_S,
    after the cycle began) or after cycle_s(); a new cycle begins with that decision."""
    tok = obs["tokens"]
    t = t.float()[:, None].expand_as(t0)
    reload, longest = cycle_s(tok)
    shooter = (tok[..., ob.INDEX["ammo"]].float() > 0) & (reload > 0)
    firing = tok[..., ob.INDEX["firing"]].float() > 0.5
    melee = tok[..., ob.INDEX["melee"]].float() > 0.5
    pos = obs["pos"].float() * ob.POS
    px, pz = pos[..., 0], pos[..., 1]
    D = _d(px[:, :, None], pz[:, :, None], px[:, None, :], pz[:, None, :])
    near = (obs["target_ok"][:, None, :] & (D <= VOLLEY_FREE_M)).any(-1)
    cond = shoot & obs["ctrl"] & shooter & firing & ~melee & ~near
    begun = t0 >= 0
    since = tok[..., ob.INDEX["volley_ready"]].float() * reload
    volley = begun & (t > t0) & (since <= VOLLEY_OPEN_S)
    free = volley | (begun & (t - t0 >= longest))
    lock = cond & ~free
    t0 = torch.where(~cond, torch.full_like(t0, -1.0), torch.where(free | ~begun, t, t0))
    return lock, t0


def has(obs):
    return obs.get("task_mk") is not None


def _d(ax, az, bx, bz):
    """Element-wise distance (no matmul: the rollout's and the update's batches give the same bits)."""
    return ((ax - bx) ** 2 + (az - bz) ** 2).sqrt()


@torch.no_grad()
def stay(obs, mk, on, share=0.5):
    """[B, N] the units melee_stay holds in their fight (module doc): own units with a task (on), in melee, not wavering
    or worse, whose mission is not "withdraw" of a group that has wavered (fighting(): the group's alive units by
    task_main, at least `share` of them wavering or worse)."""
    tok = obs["tokens"]
    waver = ((tok[..., ob.INDEX["state_wavering"]] + tok[..., ob.INDEX["state_routing"]]
              + tok[..., ob.INDEX["state_shattered"]]).float() > 0.5)
    melee = tok[..., ob.INDEX["melee"]].float() > 0.5
    own = obs.get("own", obs["ctrl"])
    alive = own & obs.get("attend", own)
    main = obs["task_main"].long()
    G = int(main.max().clamp(min=0)) + 1
    ar = torch.arange(G, device=main.device)
    mem = (main[..., None] == ar) & alive[..., None]                                   # [B, N, G]
    held = fighting(mem, melee, waver, share)                                         # [B, G] (any member in melee)
    held_u = held.gather(1, main.clamp(0, G - 1)) & (main >= 0)
    wd = mk == KIND["withdraw"]
    return on & melee & ~waver & (~wd | held_u)


def masks(cfg, obs, sec_ok, parts=None, waver_share=0.5):
    """{"kind" [B, N, 5], "target" [B, N, N], "sector" [B, N, S], "on" [B, N]} the task's allowed choices (True: allowed;
    to AND with the heads' own masks) and the units under a mask, of the observation obs (with KEYS) and the sectors
    inside the map sec_ok [B, S]; "fb_target" [B, N] the units whose target group is not seen (targets left free).
    parts: the PARTS in force (None: every one); a unit counts as under a mask ("on") when a part of its kind is;
    waver_share: GeneralConfig.waver_share (melee_stay's group rule)."""
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
    lk = obs.get("task_lock")
    vl = sh & ("shoot_volley" in P) & (lk.bool() if lk is not None else torch.zeros_like(sh))   # the volley lock
    sh, wd = sh & use("shoot_melee", "shoot_target"), wd & use("withdraw_attack", "withdraw_move")
    st = stay(obs, mk, on, waver_share) if "melee_stay" in P else torch.zeros_like(on)   # melee_stay
    ln_v = obs.get("task_line")
    line = (is_k("shoot") & ("shoot_line" in P) & (ln_v.float() < LINE_NONE / 2) if ln_v is not None
            else torch.zeros_like(on))
    limit = ln_v.float() if ln_v is not None else torch.full_like(px, LINE_NONE)
    on = att | hold | scr | sh | wd | vl | st | line                          # the units under a part in force
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
    kind = torch.where(vl[..., None], ar == KEEP, kind)                        # the volley lock: only keep
    # melee_stay: no move, no withdraw, no keep of a move / withdraw in force; the attack only at the enemies it fights
    off_now = ((tok[..., ob.INDEX["order_move"]].float() > 0.5) | (tok[..., ob.INDEX["order_withdraw"]].float() > 0.5))
    k_st = ~((ar == MOVE) | (ar == WITHDRAW) | ((ar == KEEP) & (off_now & free)[..., None]))
    kind = torch.where(st[..., None], k_st, kind)
    # shoot_line: no keep of an attack while walking to it, not firing (a shooter walks to its target's range; a
    # standing one aiming keeps - a new order resets its aim) or of a move beyond the line in force
    firing = tok[..., ob.INDEX["firing"]].float() > 0.5
    o_fwd = px + tok[..., ob.INDEX["order_fwd"]].float() * ob.POS
    moving = tok[..., ob.INDEX["moving"]].float() > 0.5
    beyond = ((att_now & ~firing & moving) | (off_now & (o_fwd > limit))) & line & ~st
    kind = kind & ~((ar == KEEP) & (beyond & free)[..., None])
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
    melee_j = tok[..., ob.INDEX["melee"]].float() > 0.5
    fights = foe[:, None, :] & melee_j[:, None, :] & (D <= STAY_M)                # [B, N, N] the enemies i fights
    tgt = torch.where(st[..., None], fights, tgt)
    # ... and no keep of an attack in force on an enemy it does not fight (its target slot task_otgt; not known: closed)
    ot = obs.get("task_otgt")
    if ot is not None:
        ot = ot.long()
        keep_att = fights.gather(2, ot.clamp(min=0)[..., None])[..., 0] & (ot >= 0)
    else:
        keep_att = torch.zeros_like(st)
    kind = kind & ~((ar == KEEP) & (st & att_now & ~keep_att & free)[..., None])
    # shoot_line: a shooter's attack only at enemies in its range now or in range from the line (their forward within
    # the line + its range: walking to its range it stops behind the line), of a unit without missiles at enemies behind
    rng = tok[..., RANGE_COL].float() * RANGE_SCALE
    in_rng = (D <= rng[..., None]) | (px[:, None, :] <= (limit + rng)[..., None])
    behind = px[:, None, :] <= limit[..., None]
    t_line = foe[:, None, :] & torch.where((shooter & (rng > 0))[..., None], in_rng, behind)
    tgt = torch.where((line & ~st)[..., None], tgt & t_line, tgt)
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
    s_line = (cx - sc.step(cfg) / 2) <= limit[..., None]                     # the sector reaches behind the line
    sec = torch.where((line & ~st)[..., None], sec & s_line, sec)
    del sec_ok   # (the heads AND it in; the fall-back is theirs)
    return {"kind": kind, "target": tgt, "sector": sec, "on": on, "fb_target": by_group & ~seen_t,
            "line": torch.where(line & ~st, limit, torch.full_like(limit, LINE_NONE))}


def fine_line(cfg, m, sector, fine_ok):
    """[B, N, fine^2] the fine cells of the chosen sector [B, N] that shoot_line leaves (their forward centre behind the
    line m["line"]; every one where no line, or where it would close all of fine_ok)."""
    if m is None or "line" not in m:
        return fine_ok
    cells = sc.cells(cfg, sector.device)[..., 0]                                  # [S, fine^2] forward
    behind = cells[sector] <= m["line"][..., None]
    ok = fine_ok & behind
    return torch.where(ok.any(-1, keepdim=True), ok, fine_ok)


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
        tm = (masks(cfg, obs_t, sec_ok, getattr(self, "parts", None), getattr(self, "waver_share", 0.5))
              if has(obs_t) else None)
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
        fine_ok = fine_line(cfg, tm, a_sector, fine_ok)
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
        # (after the ability: a sampled decision draws the other parts as without the formation's heads)
        a_face, a_width, bad = self._form(c, obs_t, a_kind, ctrl, free, out, bad, action, greedy, temperature)
        point = sc.point_index(cfg, a_sector, a_fine)
        out["nan_fixed"] = bad                     # [B] non-finite logits of the row (taken as masked)
        return out, chain.Action(a_kind, point, a_target, a_run, a_ability, a_commit, a_face, a_width)


# ChainHeads.forward's source this copy follows (source_sha(); test_general_taskmask_torch.py: a change there is to be
# copied here, then this updated)
SOURCE_SHA = "939e14e292d8c71e"


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

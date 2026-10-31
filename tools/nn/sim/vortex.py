"""The vortex at every man's death (docs/ru/training/simulator.md "Гноевики"; the game's program:
build/diag_roster/wh3_dlc29_skv_inf_pusbags/mechanic.md P1-P9; config/nn/sim.json vortex). The class of the database's
battle_vortexs fired by an ability of behaviour vortex_on_entity_death (the ability passport's `vortex`,
config/nn/abilities.json; the catalogue's, config/nn/effects.json): a unit whose effect has one (STATIC vx_fx: the Pusbags'
Pestilent Perfection) leaves at each of its men's deaths - any cause: an enemy's blow, a projectile, its own direct damage
(direct.py) - a circle on the spot where he stood, the effects table's numbers (tools/nn/sim/effects.py VORTEX):

  delay    it waits vx_delay (1 s) and then lives 1 s (P2): it strikes round(vx_delay / dt) steps after the death (the
           queue vx_q1, vx_q2: at most QUEUE steps, 1 s at the step's 0.5 s)
  who      the enemies only (affects_enemies; affects_allies 0: its own side never, P5), large bodies too; each man within
           its goal radius vx_r (2 m; d^2 - r^2 < R^2 with his body's r, the program's test P3) once (a man struck by
           a vortex is not struck by it again within 1 s: once in its life, P4)
  where    a man dies next to an enemy j when j's blows killed him (they are in contact), or - any other death (his own
           direct damage, a projectile) - when he stood in the contact with j: the share of the unit's men in the
           contact (its men striking j or struck by it, melee.strikes F_men) of its men. The vortex strikes j if the
           pair is still in contact then. The men of j it reaches: j's ranks behind its front, the front at the bodies'
           radii + vortex.contact_gap_m (MEASURED, the pusbags probe's soldier places: a dead Pusbag stood 1.27-2.55 m,
           p10-p90, from the nearest swordsman, centre to centre - 0.07-1.35 m between the bodies; averaged over the
           range) from the dead man, the next ranks sp_v further; a rank's men within the circle
           2 sqrt(R^2 + r^2 - y^2) / sp_h (j's close spacing) - 1.03 swordsmen (the probe: 1.0); a lone body 1 where
           its body reaches
  damage   vx_d (5) a man struck, vx_ap (0) whole: the base through the armour's roll U[0.5 A, A] (no shield), rounded
           as the program does round(max(0, (1 - cut) x base) + ap) - armour 0: 5, 35: 3.71, 100: ~1; the resistances
           by the vortex's flags (P6): magical (vx_magic) - no physical resistance; a spell (vx_spell) - the spell
           resistance; the ward (resist_all) always: the Pusbags' meets the ward alone; the kills Poisson on the living
           men (melee.shot_kills)
  knock    dv = (1 + collision_restitution_coefficient 0.3) F S / (F + m) (F vx_f 25, S vx_v 30, m the man's mass:
           infantry 7.8 m/s, cavalry and monsters below 1): a man whose closeness 1 - (d - r) / R is at least 0.5 is
           knocked down when dv > shockwave_speed_delta_threshold_knocked_down 3.5 (the flight past 5.25 too; P7) - the
           men within R / 2 + r of the dead man (0.29 swordsmen; the probe's places: 0.46), less his
           knock_interrupts_ignore_chance; they lie the
           mean knockdown_getting_up_delay 1.85 s: the unit's front share lying vk_p (INTERNAL; in at the knocks a front
           man, out at 1 / 1.85 s), its blows x (1 - vk_p) on every enemy. The knock back (closeness 0.25, dv 2) - no
           hit reactions in the simulator
  contact  each strike lays the vortex's contact effect (vx_ct: the censer's -10 morale, 10 s, refreshed, not stacking:
           effects.lay vortex, P8) on the whole unit struck
Not modelled (the catalogue says why): a moving vortex, one striking its own side, several in a row, a subsequent one, a
vortex of another behaviour (vortex_per_entity, vortex_on_death) or of an active ability or a spell. A battle without such
a unit is unchanged bit for bit. Needs torch."""
import math

import torch

from tools.nn.sim import effects as innate
from tools.nn.sim import melee

QUEUE = 2            # steps a death waits for its vortex at most (state.PAIRS vx_q1, vx_q2)
RANKS = 4            # ranks of the struck unit a vortex may reach (R + r over its spacing between ranks)
GAPS = 8             # points of the gap's range the reach is averaged over (men_within)


def cal(params):
    return params.sim.get("vortex") or {}


def on(params):
    """sim.json vortex.on (the rule's switch)."""
    return bool(cal(params).get("on"))


def _col(T, e, name):
    return T[e, innate.COL[name]]


def owners(u, params, T):
    """[B, N] bool: the units whose vortex effect is on now (fx_on of this step) - the dead of which fire vortices."""
    has = u["vx_fx"] >= 0
    bit = ((u["fx_on"] >> u["vx_fx"].clamp(min=0)) & 1) > 0
    return has & bit & (u["side"] > 0)


def men_within(u, R, gap, inner=None):
    """[B, p, j] the men of j within R of a dead man of p whose body is `gap` (lo, hi: m, edge to edge) from the bodies
    of j's front rank - the centre distance g = r_p + r_j + the gap, averaged over GAPS points of the range: rank k at
    g + k sp_v, its men within the chord 2 sqrt(R'^2 - y^2) / sp_h, R'^2 = R^2 + r^2 (the program's body test) or,
    with inner, R' = inner (the knock's R / 2 + r); a lone body 1 where its body reaches; at most j's men."""
    sp_h = u["sp_h"].clamp(min=0.3)[:, None, :]
    sp_v = u["sp_v"].clamp(min=0.3)[:, None, :]
    r = u["radius"][:, None, :]
    g0 = u["radius"][:, :, None] + r
    R2 = R * R + r * r if inner is None else inner * inner
    lo, hi = float(gap[0]), float(gap[1])
    n = torch.zeros_like(R2 + g0)
    reach = torch.zeros_like(n)
    for i in range(GAPS):
        g = g0 + lo + (hi - lo) * (i + 0.5) / GAPS
        for k in range(RANKS):
            y = g + k * sp_v
            n = n + 2 * torch.sqrt((R2 - y * y).clamp(min=0)) / sp_h / GAPS
        reach = reach + (R2 > g * g).float() / GAPS
    lone = (u["men0"] <= 1)[:, None, :]
    n = torch.where(lone, reach, n)
    return torch.minimum(n, u["men"].clamp(min=0)[:, None, :])


def top_damage(params):
    """The largest base damage a vortex of the catalogue may round to (a weakness up to -100 % doubles it): the loop's
    bound of hit_hp, from the json (no tensor read)."""
    vs = [float(((e or {}).get("vortex") or {}).get("damage") or 0) for e in (params.effects.get("effects") or {}).values()]
    return int(2 * max(vs + [1.0])) + 1


def hit_hp(u, base, ap, magic, spell, fire, top):
    """[B, p, j] the mean HP a man of j loses to one strike of p's vortex (module doc: damage), base / ap / flags [B, p, 1];
    top: top_damage."""
    rall = u["resist_all"][:, None, :] if "resist_all" in u else torch.zeros_like(u["resist_physical"])[:, None, :]
    res = torch.where(magic > 0, rall, u["resist_physical"][:, None, :])
    res = torch.where(spell > 0, u["resist_magic"][:, None, :], res)
    if "resist_flame" in u:
        res = torch.where(fire > 0, res + u["resist_flame"][:, None, :], res)
    keep = 1 - res.clamp(-1.0, 0.9)
    b = torch.floor(base * keep + 0.5)
    a = torch.floor(ap * keep + 0.5)
    A = (u["armour"] / 100).clamp(0, 1)[:, None, :]
    lo, w = 0.5 * A, 0.5 * A
    out = torch.zeros_like(b)
    for k in range(1, top + 1):
        # P(b (1 - cut) >= k - 1/2), cut ~ U[A / 2, A]: the round of the base part reaches k
        need = 1 - (k - 0.5) / b.clamp(min=1e-6)                       # cut at most this
        p = torch.where(w > 1e-6, ((need - lo) / w.clamp(min=1e-6)).clamp(0, 1), (need >= lo).float())
        out = out + torch.where(b >= k, p, torch.zeros_like(p))
    return out + a


def strike(u, params, dt, contact):
    """The vortices striking this step (module doc): (hp [B, p, j], kills [B, p, j], hits [B, p, j]) - the HP the
    vortices of p's dead take from j, the men they kill and the men they strike; the knocks go into vk_p (in place), the
    bookkeeping vx_hits, vx_kn. contact [B, p, j] bool: the pairs in contact now (either way)."""
    R_ = params.battle
    T = innate.table(params, u["men"].device)
    E = T.shape[0] - 1
    e = u["vx_fx"].clamp(min=0, max=E)
    lag = torch.round(_col(T, e, "vx_delay") / dt).clamp(1, QUEUE)                     # [B, p]
    q = torch.where((lag == 1)[..., None], _view(u, "vx_q1"), _view(u, "vx_q2"))         # [B, p, j]
    R = _col(T, e, "vx_r")[:, :, None]
    g = cal(params).get("contact_gap_m", (0.0, 0.0))
    alive_j = ((u["men"] > 0) & ~u["gone"] & (u["side"] > 0))[:, None, :]
    fire_ = q * contact.float() * alive_j.float()
    hits = fire_ * men_within(u, R, g)
    per = hit_hp(u, _col(T, e, "vx_d")[:, :, None], _col(T, e, "vx_ap")[:, :, None], _col(T, e, "vx_magic")[:, :, None],
                 _col(T, e, "vx_spell")[:, :, None], _col(T, e, "vx_fire")[:, :, None], top_damage(params))
    hp = hits * per
    H = hits.sum(1)
    mean = torch.where(H > 0, hp.sum(1) / H.clamp(min=1e-9), torch.ones_like(H))
    k_all = melee.shot_kills(H, u["men"].clamp(min=0), u["hp_man"], u["hp_abs"], mean.clamp(min=1e-6))
    kills = torch.where((u["men0"] > 1)[:, None, :], hits / H.clamp(min=1e-9)[:, None, :] * k_all[:, None, :],
                        torch.zeros_like(hits))
    # the knock: closeness >= 0.5 (within R / 2 + r) at dv over the shockwave's knocked-down threshold
    F, S = _col(T, e, "vx_f")[:, :, None], _col(T, e, "vx_v")[:, :, None]
    dv = (1 + float(R_["collision_restitution_coefficient"])) * F * S / (F + u["mass"][:, None, :]).clamp(min=1e-6)
    down = dv > float(R_["shockwave_speed_delta_threshold_knocked_down"])
    inner = 0.5 * R + u["radius"][:, None, :]
    stay = u["knock_ign"][:, None, :] if "knock_ign" in u else torch.zeros_like(R)
    kn = torch.where(down, fire_ * men_within(u, R, g, inner=inner) * (1 - stay), torch.zeros_like(hits))
    kn = torch.where((u["men0"] > 1)[:, None, :], kn, torch.zeros_like(kn))
    u["vx_hits"] = u["vx_hits"] + hits.sum(2)
    u["vx_kn"] = u["vx_kn"] + kn.sum(2)
    return hp, kills, hits, kn.sum(1)


def lie(u, params, dt, knocked, front):
    """vk_p (in place): the front share of each unit lying from vortex knocks - knocked [B, N] men this step over its
    front men `front` [B, N] (its men in the fight), out at 1 / the mean getting-up delay."""
    R_ = params.battle
    tau = (float(R_["knockdown_getting_up_delay_min"]) + float(R_["knockdown_getting_up_delay_max"])) / 2
    add = (knocked / front.clamp(min=1.0)).clamp(0, 1)
    u["vk_p"] = (u["vk_p"] + add * (1 - u["vk_p"])) * math.exp(-dt / tau)


def up_share(u, rate):
    """rate [B, i, j] (i strikes j) x (1 - vk_p of i), in place (x 1 where nothing lies: the same bits)."""
    return rate.mul_((1 - u["vk_p"])[:, :, None])


def _view(u, name):
    B, N = u["men"].shape
    return u[name].view(B, N, N)


def queue(u, params, dead_near):
    """Push this step's dead next to each enemy (dead_near [B, p, j], the owners' rows only) into the queue, in place."""
    B, N = u["men"].shape
    u["vx_q2"] = u["vx_q1"]
    u["vx_q1"] = dead_near.reshape(B, N * N).contiguous()


def dead_near(u, params, own, drop, deaths, melee_share, F_men, men_before, enemy):
    """[B, p, j] this step's dead of p next to enemy j (module doc: where): deaths [B, i, p] the men of p killed by i (every
    cause; i = p its own direct damage), melee_share [B, i, p] the share of i's HP on p that were blows, drop [B, p] the
    men p lost, F_men [B, i, j] men of i striking j; own [B, p] the vortex owners."""
    d_mel = deaths * melee_share * enemy                                          # [B, i, p]: blows of i
    other = (drop - d_mel.sum(1)).clamp(min=0)                                    # [B, p]
    inside = torch.maximum(F_men, F_men.transpose(1, 2))                          # [B, p, j]: p's men in the contact
    share = (inside / men_before.clamp(min=1e-6)[:, :, None]).clamp(0, 1)
    near = d_mel.transpose(1, 2) + other[:, :, None] * share
    return torch.where(own[:, :, None] & enemy, near, torch.zeros_like(near))

"""Pair features of two units (v2, ModelConfig.pairs): what the network knows of a pair beyond their distance.

The attention between the units had only a learned bias per head for the distance between two units (encoder.py,
16 buckets). Who reaches whom, who closes in on whom and from which side came only through the tokens, and the
attention's q.k had to work them out itself. Now, for every pair (i looks at j), from the side's own observation (the
unit tokens and their positions: the same in the simulator and the companion, nothing new to send):

* reach: the distance / the missile range of i, and of j (0 for a unit without missiles; at most REACH_CAP), and
  whether j is in i's range and i in j's (0 / 1) - who can shoot whom;
* closing: the speed at which the two close in, m/s / VEL: minus the projection of their relative velocity on the
  line between them (> 0: they come nearer), within +-CLOSE_CAP; both must be seen now (an unseen enemy's velocity is
  not known);
* bearing: where j stands as i faces (cos, sin of the angle from i's front to j: cos < 0 behind it, sin > 0 to its
  right) and where i stands as j faces - flank and rear, both ways; both seen now.

Each comes twice: for a pair of the same side and for a pair of opposite sides (an enemy closing in is not a friend
closing in; one linear layer cannot tell them apart otherwise): FEATURES x 2 numbers per pair. The context
token and pairs with an unknown position give zeros.

Then SPELL (16.10.2026, appended last: older networks load with zero weights for them): the spell reach of a caster
i - the distance / the longest range of i's spells (its ability slots' passports, abilities.SPELL_SLOTS; 0 for a unit
without spells; at most REACH_CAP) and whether j is within it (0 / 1), the same twice (same side, opposite sides):
for each enemy the caster sees how far he is inside his spell's range or how much farther (no target is masked for
the range: the caster walks into range, tools/nn/sim/magic.py; the price of walking up is the press's own). SIZE =
(FEATURES + SPELL) x 2.

Then DAMAGE (23.10.2026, appended last: older networks load with zero weights for them and compute what they did, bit
for bit): what a hit of i takes from j, by the simulator's damage rule (sim/melee.py per_hit, the one missile.volley
and the melee use: the base damage less the armour roll, the armour-piercing part whole, the resistances - missile:
missile + physical + ward, melee: physical + ward, at most 90 % - and the damage beyond a killed man lost; a lone man
loses the mean) as a share of j's man (0..1: a hit that kills is 1), of the units' cards in the tokens (the game's
database, the same in the simulator and the companion):
* shot_i: a projectile of i (its missile damage and armour-piercing; 0 for a unit without a missile range), less j's
  shield where i stands within SHIELD_DEG of j's front (or j's facing not seen now: the front assumed);
* blow_i: a melee blow of i (its melee damage and armour-piercing; no hit chance, no bonus against large or infantry).
Only for a pair of opposite sides (a friend's damage means nothing), both known. "j in i's range" is in_range_i
already. SIZE = (FEATURES + SPELL) x 2 + DAMAGE.

The network uses them in two places, each a linear layer started at zero (so a network trained without them computes
exactly what it did: the loaded weights are kept and the new ones are 0, policy.Actor.load_state_dict):
* PairBias: + a number per attention head to the distance bias of every units' attention block (who to look at);
* the target pointer (chain.py): + a number to the logit of attacking j (who to attack).
"""
import copy
import json
import math

import numpy as np
import torch
from torch import nn

from tools import config as project
from tools.nn.model import observation as ob
from tools.nn.model import passport
from tools.nn.sim import melee as smelee

FEATURES = ("reach_i", "reach_j", "in_range_i", "in_range_j", "closing", "bearing_cos_i", "bearing_sin_i",
            "bearing_cos_j", "bearing_sin_j")
SPELL = ("spell_reach_i", "spell_in_i")
DAMAGE = ("shot_i", "blow_i")  # opposite sides only
OLD_SIZE = 2 * len(FEATURES)   # before SPELL (16.10.2026)
SPELL_SIZE = 2 * (len(FEATURES) + len(SPELL))  # before DAMAGE (23.10.2026)
SIZE = SPELL_SIZE + len(DAMAGE)                # x (same side, opposite sides) + the damage (opposite sides)
REACH_CAP = 2.0                # distance / range, at most
CLOSE_CAP = 2.0                # closing speed / VEL, within +-
RANGE_SCALE = 500.0            # the passport's range_m / 500 (passport.features)


def _range_column():
    """The passport column of the missile range (range_m / RANGE_SCALE)."""
    base = dict(men=1, hp_total=1, hp_per_man=1, mass=1, armour=0, leadership=0, multiplayer_cost=0, caste="", size="",
                speed={"walk": 0, "run": 0, "charge": 0}, shield={"missile_block_chance": 0},
                melee={"attack": 0, "defence": 0, "charge_bonus": 0, "damage": 0, "ap_damage": 0, "bonus_v_large": 0,
                       "bonus_v_infantry": 0, "attack_interval_s": 0, "splash_max_attacks": 1})
    a = np.array(passport.features(dict(base, missile={"range_m": 0})))
    b = np.array(passport.features(dict(base, missile={"range_m": RANGE_SCALE})))
    (i,) = np.nonzero(a != b)[0]
    assert abs(b[i] - 1.0) < 1e-9
    return ob.INDEX[f"passport_{i}"]


RANGE = _range_column()

_BASE = dict(men=1, hp_total=1, hp_per_man=1, mass=1, armour=0, leadership=0, multiplayer_cost=0, caste="", size="",
             speed={"walk": 0, "run": 0, "charge": 0}, shield={"missile_block_chance": 0}, damage_resist={},
             melee={"attack": 0, "defence": 0, "charge_bonus": 0, "damage": 0, "ap_damage": 0, "bonus_v_large": 0,
                    "bonus_v_infantry": 0, "attack_interval_s": 0, "splash_max_attacks": 1},
             missile={"range_m": 100})


def _column(path, value):
    """The token's column of the passport number at path (a key path of the passport dict): the one column that moves
    when that number is set to value (passport.features: found by its value, not its place)."""
    p = copy.deepcopy(_BASE)
    d = p
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = value
    a, b = np.array(passport.features(_BASE)), np.array(passport.features(p))
    (i,) = np.nonzero(a != b)[0]
    return ob.INDEX[f"passport_{i}"]


# the cards' numbers of the damage rule (DAMAGE): (column, how the token holds it - "log" passport._log(v, top), else
# v / scale)
CARD = {"m_damage": (("missile", "damage"), "log", 500), "m_ap": (("missile", "ap_damage"), "log", 500),
        "damage": (("melee", "damage"), "log", 500), "ap": (("melee", "ap_damage"), "log", 500),
        "armour": (("armour",), "lin", 200), "hp_man": (("hp_per_man",), "log", 10000), "men": (("men",), "log", 200),
        "shield": (("shield", "missile_block_chance"), "lin", 100),
        "res_all": (("damage_resist", "all"), "lin", 100), "res_physical": (("damage_resist", "physical"), "lin", 100),
        "res_missile": (("damage_resist", "missile"), "lin", 100)}
CARD_COL = {k: _column(path, 50) for k, (path, _, _) in CARD.items()}
RESIST_CAP = 0.9                # the resistances' sum, at most (CA; the simulator's rule)
# a shield blocks projectiles from within this of the unit's front (config/nn/game_rules.json, the game's database)
SHIELD_DEG = float(json.loads((project.ROOT / "config/nn/game_rules.json").read_text(encoding="utf-8"))["battle"]
                   ["shield_defence_angle_missile"])


def card(tok, name):
    """[B, N] the card's number `name` (CARD) of every token, back in the game's units (damage, armour, health of a
    man, men; shield and resistances as shares 0-1)."""
    _, kind, top = CARD[name]
    c = tok[..., CARD_COL[name]].float()
    if kind == "log":
        return torch.expm1(c * math.log1p(top)).clamp(min=0)
    return c * top / (100.0 if name in ("shield", "res_all", "res_physical", "res_missile") else 1.0)


def damage(tok, cos_j, front_known):
    """[B, N, N, 2] (DAMAGE: shot_i, blow_i) the share of j's man a hit of i takes (module doc) of the tokens [B, N, T];
    cos_j [B, N, N] i's place as j faces (bearing_cos_j), front_known [B, N, N] where that bearing is known."""
    armour, hp_man = card(tok, "armour")[:, None, :], card(tok, "hp_man").clamp(min=1.0)[:, None, :]
    single = (card(tok, "men") < 1.5)[:, None, :]
    ward = card(tok, "res_all") + card(tok, "res_physical")
    res_m = (ward + card(tok, "res_missile")).clamp(max=RESIST_CAP)[:, None, :]
    res_p = ward.clamp(max=RESIST_CAP)[:, None, :]
    col = (lambda n: card(tok, n)[:, :, None])
    shot = smelee.per_hit(col("m_damage"), col("m_ap"), armour, hp_man, res_m, single=single)
    front = ~front_known | (cos_j >= math.cos(math.radians(SHIELD_DEG)))
    shot = shot * (1 - card(tok, "shield")[:, None, :] * front.float())
    shoots = (tok[..., RANGE] > 0)[:, :, None] & ((col("m_damage") + col("m_ap")) > 0)
    shot = torch.where(shoots, shot, torch.zeros_like(shot))
    blow = smelee.per_hit(col("damage"), col("ap"), armour, hp_man, res_p, single=single)
    blow = torch.where((col("damage") + col("ap")) > 0, blow, torch.zeros_like(blow))
    return (torch.stack([shot, blow], -1) / hp_man[..., None]).clamp(0.0, 1.0)


def features(obs_t):
    """[B, 1 + N, 1 + N, SIZE] the pair features (module doc); the context token's row and column are 0."""
    tok = obs_t["tokens"]
    dt = tok.dtype
    pos = obs_t["pos"].to(dt) * ob.POS                                            # [B, N, 2] m (forward, lateral)
    known = obs_t["attend"] & (tok[..., ob.INDEX["seen"]] > 0.5)
    now = known & (tok[..., ob.INDEX["visible"]] > 0.5)
    rel = pos[:, None, :, :] - pos[:, :, None, :]                                 # [B, i, j, 2]: from i to j
    d = rel.square().sum(-1).clamp(min=1e-6).sqrt()
    u = rel / d[..., None]
    rng = tok[..., RANGE] * RANGE_SCALE                                           # [B, N] m (0: no missiles)
    shoots = rng > 0
    r_safe = torch.where(shoots, rng, torch.ones_like(rng))
    reach_i = torch.where(shoots[:, :, None], (d / r_safe[:, :, None]).clamp(max=REACH_CAP), torch.zeros_like(d))
    reach_j = torch.where(shoots[:, None, :], (d / r_safe[:, None, :]).clamp(max=REACH_CAP), torch.zeros_like(d))
    in_i = (shoots[:, :, None] & (d <= rng[:, :, None])).to(dt)
    in_j = (shoots[:, None, :] & (d <= rng[:, None, :])).to(dt)
    vel = torch.stack([tok[..., ob.INDEX["vel_fwd"]], tok[..., ob.INDEX["vel_lat"]]], -1)        # / VEL
    rel_v = vel[:, None, :, :] - vel[:, :, None, :]                               # j's velocity minus i's
    closing = (-(rel_v * u).sum(-1)).clamp(-CLOSE_CAP, CLOSE_CAP)
    face = torch.stack([tok[..., ob.INDEX["face_cos"]], tok[..., ob.INDEX["face_sin"]]], -1)     # (fwd, lat)
    fi, fj = face[:, :, None, :], face[:, None, :, :]
    cos_i = (fi * u).sum(-1)                                                      # j as i faces
    sin_i = fi[..., 0] * u[..., 1] - fi[..., 1] * u[..., 0]                       # > 0: j to i's right
    cos_j = -(fj * u).sum(-1)                                                     # i as j faces (j -> i is -u)
    sin_j = -(fj[..., 0] * u[..., 1] - fj[..., 1] * u[..., 0])
    both = known[:, :, None] & known[:, None, :]
    seen_now = now[:, :, None] & now[:, None, :]
    apart = d > 1.0                                                                # (no bearing for one place)
    zero = torch.zeros_like(d)
    feats = torch.stack([reach_i, reach_j, in_i, in_j]
                        + [torch.where(apart, c, zero) for c in (closing, cos_i, sin_i, cos_j, sin_j)], -1)
    ok = torch.stack([both] * 4 + [seen_now] * 5, -1)
    feats = torch.where(ok, feats, torch.zeros_like(feats))
    own = obs_t["own"]
    same = (own[:, :, None] == own[:, None, :])[..., None].to(dt)
    sr = spell_range(obs_t).to(dt)                                                # [B, N] m (0: no spells)
    casts = sr > 0
    s_safe = torch.where(casts, sr, torch.ones_like(sr))
    s_reach = torch.where(casts[:, :, None], (d / s_safe[:, :, None]).clamp(max=REACH_CAP), torch.zeros_like(d))
    s_in = (casts[:, :, None] & (d <= sr[:, :, None])).to(dt)
    sp = torch.where(both[..., None], torch.stack([s_reach, s_in], -1), torch.zeros_like(d)[..., None])
    dmg = damage(tok, cos_j, seen_now & apart).to(dt)
    dmg = torch.where(both[..., None], dmg, torch.zeros_like(dmg))
    out = torch.cat([feats * same, feats * (1 - same), sp * same, sp * (1 - same), dmg * (1 - same)], -1)
    return torch.nn.functional.pad(out, (0, 0, 1, 0, 1, 0))                       # the context token


def spell_range(obs_t):
    """[B, N] the longest range (m) of each unit's spells (its owned spell slots' passports; 0: none, or no ability
    input)."""
    from tools.nn.model import abilities as ab
    a = obs_t.get("abil")
    if a is None:
        return torch.zeros_like(obs_t["pos"][..., 0])
    k = list(ab.SPELL_SLOTS)
    owned = a[:, :, k, ab.STATE_INDEX["owned"]] > 0.5
    rng = ab.column("range_m", obs_t["abil_idx"][:, :, k].long()) * 100.0
    return torch.where(owned, rng.to(a.dtype), torch.zeros_like(rng, dtype=a.dtype)).max(-1).values


def zero_linear(n_out):
    """nn.Linear(SIZE, n_out, bias=False) with zero weights, drawing the random numbers a layer of SPELL_SIZE inputs
    drew (its init, then zeroed): a fresh network's other weights stay those of the code before DAMAGE, seed for seed
    (the tests' fixed seeds; the weights here are 0 either way)."""
    lin = nn.Linear(SPELL_SIZE, n_out, bias=False)
    lin.weight = nn.Parameter(torch.zeros(n_out, SIZE))
    lin.in_features = SIZE
    return lin


class PairBias(nn.Module):
    """The pair features -> + a number per attention head (started at zero: nothing changes until trained)."""

    def __init__(self, heads):
        super().__init__()
        self.lin = zero_linear(heads)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """A network saved before SPELL or DAMAGE: zero weights for them (computes what it did)."""
        from tools.nn.model.encoder import pad_inputs
        pad_inputs(state_dict, prefix + "lin.weight", SIZE)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, f):
        """f [B, L, L, SIZE] -> [B, heads, L, L]."""
        return self.lin(f).permute(0, 3, 1, 2)

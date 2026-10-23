"""tools/nn/model/pairs.py DAMAGE: what a hit of unit i takes from unit j (the simulator's damage rule, sim/melee.py
per_hit) as a share of j's man - a projectile (less j's shield from its front) and a melee blow - from the cards in
the tokens, for enemies only; appended last: a network saved before them loads with zero weights for them and computes
what it did, bit for bit. torch: the training container, on the CPU."""
import math

import pytest

torch = pytest.importorskip("torch")

from tools.nn.model import config, observation as ob, pairs, passport, policy   # noqa: E402
from tools.nn.sim import melee as smelee                                        # noqa: E402

CFG = config.preset("v2", d=32, layers=2, heads=2, pointer=16, critic_d=32, critic_layers=1, critic_heads=2,
                    sector_d=16)
BASE = dict(men=120, hp_total=7200, hp_per_man=60, mass=60, armour=25, leadership=40, multiplayer_cost=400,
            caste="melee_infantry", size="small", speed={"walk": 4, "run": 6, "charge": 8},
            shield={"missile_block_chance": 0}, damage_resist={},
            melee={"attack": 20, "defence": 20, "charge_bonus": 10, "damage": 25, "ap_damage": 5,
                   "bonus_v_large": 0, "bonus_v_infantry": 0, "attack_interval_s": 4, "splash_max_attacks": 1})
SLING = dict(BASE, missile={"range_m": 150, "ammo": 20, "damage": 7, "ap_damage": 1, "reload_s": 6, "accuracy": 30,
                            "projectile_number": 1})
GREATSWORD = dict(BASE, armour=95, hp_per_man=80)
MILITIA = dict(BASE, armour=25, hp_per_man=60)


def obs_of(cards, own, pos, face):
    """One row of units with the cards' passports in their tokens, at pos [(fwd, lat) m], facing face [(cos, sin)]."""
    n = len(cards)
    tok = torch.zeros(1, n, ob.TOKEN)
    for u, p in enumerate(cards):
        f = passport.features(p)
        for i, v in enumerate(f):
            tok[0, u, ob.INDEX[f"passport_{i}"]] = v
        tok[0, u, ob.INDEX["seen"]] = tok[0, u, ob.INDEX["visible"]] = 1.0
        tok[0, u, ob.INDEX["face_cos"]], tok[0, u, ob.INDEX["face_sin"]] = face[u]
    return {"tokens": tok, "pos": torch.tensor([pos], dtype=torch.float32) / ob.POS,
            "attend": torch.ones(1, n, dtype=torch.bool), "own": torch.tensor([own])}


def hit_share(att, tgt, missile=True, shield=0.0):
    a = att["missile"] if missile else att["melee"]
    t = torch.tensor
    h = smelee.per_hit(t(float(a["damage"])), t(float(a["ap_damage"])), t(float(tgt["armour"])),
                       t(float(tgt["hp_per_man"])))
    left = 1 - pairs.SHIELD_PIERCE * a["ap_damage"] / (a["damage"] + a["ap_damage"])   # the armour-piercing pierces
    return float(h) * (1 - shield * left) / tgt["hp_per_man"]


def test_the_cards_are_read_back_from_the_tokens():
    o = obs_of([SLING, dict(GREATSWORD, shield={"missile_block_chance": 35}, damage_resist={"missile": 20})],
               [True, False], [(0, 0), (100, 0)], [(1, 0), (-1, 0)])
    tok = o["tokens"]
    assert float(pairs.card(tok, "m_damage")[0, 0]) == pytest.approx(7, abs=1e-3)
    assert float(pairs.card(tok, "m_ap")[0, 0]) == pytest.approx(1, abs=1e-3)
    assert float(pairs.card(tok, "armour")[0, 1]) == pytest.approx(95, abs=1e-3)
    assert float(pairs.card(tok, "hp_man")[0, 1]) == pytest.approx(80, abs=1e-2)
    assert float(pairs.card(tok, "shield")[0, 1]) == pytest.approx(0.35, abs=1e-6)
    assert float(pairs.card(tok, "res_missile")[0, 1]) == pytest.approx(0.20, abs=1e-6)
    assert pairs.SHIELD_DEG == 60.0 and pairs.SHIELD_PIERCE == 0.5      # config/nn/game_rules.json, sim.json
    left = pairs.shield_left(torch.tensor([7.0, 2.0]), torch.tensor([1.0, 6.0]))
    assert left.tolist() == pytest.approx([1 - 0.5 / 8, 1 - 0.5 * 6 / 8])  # (the Ratling Gun's 6 of 8: 0.625)


def test_a_hit_is_the_simulators_damage_as_a_share_of_the_enemys_man():
    # 0 slingers (ours) at the origin facing forward; 1 greatswords, 2 militia (theirs) 100 m ahead facing them;
    # 3 a friend of the slingers
    cards = [SLING, GREATSWORD, MILITIA, MILITIA]
    o = obs_of(cards, [True, False, False, True], [(0, 0), (100, -20), (100, 20), (0, 40)],
               [(1, 0), (-1, 0), (-1, 0), (1, 0)])
    f = pairs.features(o)[0, 1:, 1:]                                     # (the context token off)
    D, k = pairs.SPELL_SIZE, {n: i for i, n in enumerate(pairs.DAMAGE)}
    assert pairs.SIZE == pairs.SPELL_SIZE + 2 == 24
    shot = f[..., D + k["shot_i"]]
    blow = f[..., D + k["blow_i"]]
    assert float(shot[0, 1]) == pytest.approx(hit_share(SLING, GREATSWORD), abs=1e-5)
    assert float(shot[0, 2]) == pytest.approx(hit_share(SLING, MILITIA), abs=1e-5)
    assert float(shot[0, 1]) < 0.5 * float(shot[0, 2])                   # the armour of 95 stops the slings
    assert float(blow[0, 2]) == pytest.approx(hit_share(SLING, MILITIA, missile=False), abs=1e-5)
    assert float(shot[1, 0]) == 0 and float(blow[1, 0]) > 0               # the greatswords shoot nothing
    assert float(shot[0, 3]) == 0 and float(blow[0, 3]) == 0              # a friend: no damage numbers
    assert bool(((f[..., D:] >= 0) & (f[..., D:] <= 1)).all())
    # a shield blocks from the front only (within 60 deg of its facing)
    shielded = dict(MILITIA, shield={"missile_block_chance": 50})
    o2 = obs_of([SLING, GREATSWORD, shielded, MILITIA], [True, False, False, True],
                [(0, 0), (100, -20), (100, 20), (0, 40)], [(1, 0), (-1, 0), (-1, 0), (1, 0)])
    s2 = pairs.features(o2)[0, 1:, 1:][..., D + k["shot_i"]]
    assert float(s2[0, 2]) == pytest.approx(hit_share(SLING, MILITIA, shield=0.5), abs=1e-5)
    o3 = obs_of([SLING, GREATSWORD, shielded, MILITIA], [True, False, False, True],
                [(0, 0), (100, -20), (100, 20), (0, 40)], [(1, 0), (-1, 0), (1, 0), (1, 0)])   # its back to them
    s3 = pairs.features(o3)[0, 1:, 1:][..., D + k["shot_i"]]
    assert float(s3[0, 2]) == pytest.approx(hit_share(SLING, MILITIA), abs=1e-5)
    # an enemy never seen: nothing
    tok = o["tokens"].clone()
    tok[:, 2, ob.INDEX["seen"]] = tok[:, 2, ob.INDEX["visible"]] = 0
    assert torch.all(pairs.features(dict(o, tokens=tok))[0, 1:, 1 + 2, D:] == 0)
    assert math.isfinite(float(f.sum()))


def test_a_network_saved_before_the_damage_numbers_computes_exactly_what_it_did():
    torch.manual_seed(0)
    net = policy.Actor(CFG)
    with torch.no_grad():                                                # (trained pair layers: not zero)
        net.pair_bias.lin.weight.normal_()
        net.heads.target_pair.weight.normal_()
    old = {k: (v[:, :pairs.SPELL_SIZE].clone() if k in ("pair_bias.lin.weight", "heads.target_pair.weight")
               else v.clone()) for k, v in net.state_dict().items()}
    loaded = policy.Actor(CFG)
    loaded.load_state_dict(old)
    for k in ("pair_bias.lin.weight", "heads.target_pair.weight"):
        w = loaded.state_dict()[k]
        assert torch.all(w[:, pairs.SPELL_SIZE:] == 0) and torch.equal(w[:, :pairs.SPELL_SIZE], old[k])
    f = torch.rand(2, 5, 5, pairs.SIZE)
    zero = f.clone()
    zero[..., pairs.SPELL_SIZE:] = 0
    # whatever the new numbers: the same attention bias, bit for bit
    assert torch.equal(loaded.pair_bias(f), loaded.pair_bias(zero))
    assert torch.equal(loaded.heads.target_pair(f), loaded.heads.target_pair(zero))

"""tools/nn/general/army_reader.py: the general's army analysis (design/general/index.html, "Проверки анализа армий",
group 1 "При сборке")."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn.general import army_input as ai                    # noqa: E402
from tools.nn.general import army_reader as ar                   # noqa: E402
from tools.nn.model import passport as pp                        # noqa: E402

F64 = torch.float64
LORD = "wh_main_emp_cha_general_0"
SPEAR = "wh_main_emp_inf_spearmen_0"
ARCHERS = "wh2_dlc13_emp_inf_archers_0"
GUNS = "wh_main_emp_inf_handgunners"
SLAVES = "wh2_main_skv_inf_skavenslaves_0"
RATLING = "wh2_dlc12_skv_inf_ratling_gun_0"
OURS = [SPEAR, LORD, ARCHERS, GUNS, "wh_dlc04_emp_inf_flagellants_0"]
THEIRS = [SLAVES, RATLING, "wh2_main_skv_cha_warlord_0", "wh2_dlc16_skv_mon_wolf_rats_0", SLAVES]


def model(seed=0, dtype=F64, **kw):
    torch.manual_seed(seed)
    m = ar.ArmyReader(**kw).to(dtype)
    with torch.no_grad():                    # biases nonzero, so an empty slot's 0 differs from a pass of zeros
        m.layer1.bias.uniform_(-0.2, 0.2)
        m.layer2.bias.uniform_(-0.2, 0.2)
    return m


def tensors(a, dtype=F64):
    n, l, e = ai.to_tensors(a)
    return n.to(dtype), l, e


def cards(m, armies, side=1):
    with torch.no_grad():
        return m(*tensors(ai.build(armies, side)))


def test_shape_and_size():
    m = ar.ArmyReader()
    assert ar.weights(m) == 49 * 256 + 256 + (4 + 12 + 4) * 256 + 256 * 256 + 256 == 83712
    assert m.variants() == {"trajectory": 4, "caste": 12, "size": 4}
    c = cards(model(), {1: OURS, 2: THEIRS})
    assert c.shape == (80, 256)
    b = ai.stack([ai.build({1: OURS, 2: THEIRS}), ai.build({1: OURS, 2: THEIRS}, 2)])
    assert model(dtype=torch.float32)(*ai.to_tensors(b)).shape == (2, 80, 256)
    assert ar.ArmyReader(hidden=64, card=32)(*ai.to_tensors(b)).shape == (2, 80, 32)


def test_slot_order_means_nothing():
    m = model()
    a = cards(m, {1: OURS, 2: THEIRS})
    perm = [3, 0, 4, 1, 2]
    b = cards(m, {1: [OURS[i] for i in perm], 2: THEIRS[::-1]})
    torch.testing.assert_close(b[:5], a[perm], rtol=0, atol=1e-12)
    torch.testing.assert_close(b[40:45], a[40:45].flip(0), rtol=0, atol=1e-12)


def test_empty_slots_change_nothing():
    """Empty slots added or taken away: the others' cards stay; an empty slot's card is 0."""
    m = model()
    a = cards(m, {1: OURS, 2: THEIRS})
    gappy = [None, OURS[0], None, OURS[1], OURS[2], None, OURS[3], OURS[4]]
    b = cards(m, {1: gappy, 2: THEIRS})
    full = [i for i, e in enumerate(gappy) if e]
    torch.testing.assert_close(b[full], a[:5], rtol=0, atol=1e-12)
    torch.testing.assert_close(b[40:], a[40:], rtol=0, atol=1e-12)
    empty = ai.build({1: gappy, 2: THEIRS}).empty
    assert not b[torch.as_tensor(empty)].any()
    # a slot marked empty is never read, whatever it holds
    x = ai.build({1: OURS, 2: THEIRS})
    n, l, e = tensors(x)
    n[7] = float("nan")
    n[60, 10:] = 1e6
    l[60] = 99                                            # not even a variant
    with torch.no_grad():
        c = m(n, l, e)
    torch.testing.assert_close(c, a, rtol=0, atol=1e-12)


def test_one_network_for_all():
    """The same unit in slot 3 and slot 17 (own) gives the same card; own slot 3 and enemy slot 57 differ only by the
    side flag's column of the first layer."""
    m = model()
    own = [SPEAR] * 3 + [ARCHERS] + [SPEAR] * 13 + [ARCHERS]
    enemy = [SLAVES] * 17 + [ARCHERS]
    x = ai.build({1: own, 2: enemy})
    n, l, e = tensors(x)
    with torch.no_grad():
        c = m(n, l, e)
        torch.testing.assert_close(c[3], c[17], rtol=0, atol=1e-12)
        h = m.first(n[[3, 57]], l[[3, 57]])
    torch.testing.assert_close(h[0] - h[1], m.layer1.weight[:, ai.SIDE].detach(), rtol=0, atol=1e-12)
    assert not torch.allclose(c[3], c[57])


def test_side_2_is_the_mirror():
    m = model()
    a = cards(m, {1: OURS, 2: THEIRS}, side=2)
    b = cards(m, {1: THEIRS, 2: OURS}, side=1)
    assert torch.equal(a, b)
    c = cards(m, {1: OURS, 2: THEIRS}, side=1)
    assert not torch.allclose(a, c)                       # the sides really swap


def test_a_new_variant_changes_nothing():
    """A new caste with zero weights: every old card the same to the bit; a unit of the new caste reads as if it
    had no caste vector."""
    m = model()
    x = tensors(ai.build({1: OURS, 2: THEIRS}))
    with torch.no_grad():
        before = m(*x)
        m.add_variant("caste")
        m.add_variant("size", 2)
        m.add_variant("trajectory")
        after = m(*x)
    assert torch.equal(before, after)
    assert m.variants() == {"trajectory": 5, "caste": 13, "size": 6}
    n, l, e = x
    l2 = l.clone()
    l2[0, 1] = 12                                         # the new caste
    with torch.no_grad():
        c = m(n, l2, e)
        h = m.layer1(n[:1]) + m.labels["trajectory"](l[:1, 0]) + m.labels["size"](l[:1, 2])
        want = m.layer2(torch.relu(h))
    torch.testing.assert_close(c[:1], want, rtol=0, atol=1e-12)
    assert torch.equal(c[1:], before[1:])


def test_no_missile_weapon_is_a_fixed_zero_vector():
    m = model()
    e = m.labels["trajectory"]
    assert e.padding_idx == 0 and not e.weight[0].any()
    x = tensors(ai.build({1: OURS, 2: THEIRS}))
    m(*x).square().sum().backward()
    assert not e.weight.grad[0].any() and e.weight.grad[1:].abs().sum() > 0
    assert m.labels["caste"].weight.grad.abs().sum() > 0


def test_effects_are_added_to_the_cards():
    m = model()
    x = tensors(ai.build({1: OURS, 2: THEIRS}))
    with torch.no_grad():
        base = m(*x)
        fx = torch.randn(80, 256, dtype=F64)
        c = m(*x, effects=fx)
    full = ~x[2]
    torch.testing.assert_close(c[full], base[full] + fx[full], rtol=0, atol=1e-12)
    assert not c[~full].any()


def test_numbers_keep_their_scale():
    """Every unit of units.json: finite cards, of about the input's scale (not flying off, not dying out); the first
    layer neither all on nor all off."""
    keys = list(pp.load())
    m = model(dtype=torch.float32)
    with torch.no_grad():
        m.layer1.bias.zero_()
        m.layer2.bias.zero_()
    x = ai.to_tensors(ai.build({1: keys[:40], 2: keys[:40][::-1]}))
    with torch.no_grad():
        c = m(*x)
        h = m.first(x[0][~x[2]], x[1][~x[2]])
    full = c[~x[2]]
    assert torch.isfinite(full).all()
    rms_in = x[0][~x[2]].square().mean().sqrt()
    rms_out = full.square().mean().sqrt()
    assert 0.2 < float(rms_out / rms_in) < 5, (float(rms_in), float(rms_out))
    on = (h > 0).double().mean()
    assert 0.2 < float(on) < 0.8
    assert full.std(0).min() > 0                          # no card number constant over all units


def test_spread_agrees_with_the_simulator():
    """army_input.spread_100 is the simulator's square_side at 100 m (tools/nn/sim/missile.py) for every shooter."""
    from tools.nn.sim import missile
    shooters = {k: p for k, p in pp.load().items() if p.get("missile")}
    rows = [p["missile"] for p in shooters.values()]
    u = {"acc": torch.tensor([[m["accuracy"] + m["marksmanship"] for m in rows]], dtype=F64),
         "cal_area": torch.tensor([[m["calibration_area_m"] for m in rows]], dtype=F64),
         "cal_dist": torch.tensor([[m["calibration_distance_m"] for m in rows]], dtype=F64)}
    v = torch.tensor([[m["muzzle_velocity"] for m in rows]], dtype=F64)[:, :, None]
    d = torch.full((1, len(rows), 1), 100.0, dtype=F64)
    theta = (0.5 * torch.asin((d * 9.81 / (v * v)).clamp(max=1.0))).clamp(min=1e-3)
    side = missile.square_side(u, d, theta)[0, :, 0]
    mine = torch.tensor([ai.spread_100(m["accuracy"] + m["marksmanship"], m["calibration_area_m"],
                                       m["calibration_distance_m"], m["muzzle_velocity"], {"spread": "square"})
                         for m in rows], dtype=F64)
    torch.testing.assert_close(mine, side, rtol=1e-9, atol=1e-9)
    assert ai._sim_missile()["accuracy"]["spread"] == "square"
    got = np.array([ai.final_stats(p)["spread_100"] for p in shooters.values()])
    assert np.allclose(got, side.numpy())

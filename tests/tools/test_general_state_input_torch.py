"""The general's slots' state group (army_input STATE, 23.10.2026): morale, wavering, fatigue and ammunition now
(units.dynamic), a shooter only with ammunition left (army_input.armed: the threat's shooters' shares, the "shoot" mask),
and an older general (45 numbers a slot) loads with zero weights for them and plans as before. torch: the container."""
import pytest

torch = pytest.importorskip("torch")

from tools.nn.general import army_input as ai                  # noqa: E402
from tools.nn.general import army_reader as ar                 # noqa: E402
from tools.nn.general import config as gconfig                 # noqa: E402
from tools.nn.general import net as gnet                       # noqa: E402
from tools.nn.general import units                             # noqa: E402

KEYS = ["wh_main_emp_inf_handgunners", "wh_main_emp_inf_swordsmen", "wh2_main_skv_inf_skavenslave_slingers_0",
        "wh2_main_skv_inf_clanrats_0", ""]
OWN = torch.tensor([[True, True, False, False, False]])
I = ai.INDEX


def table():
    t = units.UnitTable()
    return t.battle([KEYS])


def state(full, **kw):
    u = {"men": full[..., 0], "hp": torch.ones(1, 5), "hp0": full[..., 1],
         "mp": torch.tensor([[0.4, 1.2, -0.5, 4.0, 0.0]]), "ms": torch.tensor([[3.0, 5.0, 6.0, 2.0, 0.0]]),
         "fat": torch.tensor([[2.0, 0.0, 5.0, 1.0, 0.0]]), "a": full[..., 2] * torch.tensor([[0.25, 0, 0.0, 0, 0]])}
    u.update(kw)
    return u


def test_the_state_now_of_own_units_and_the_enemies_seen():
    num, lab, full = table()
    assert full.shape == (1, 5, 3) and float(full[0, 0, 2]) > 0 and float(full[0, 1, 2]) == 0
    got = units.dynamic(num, full, state(full), OWN, torch.tensor([[False, False, True, True, False]]))
    g = got[0]
    assert g[:4, I["morale"]].tolist() == pytest.approx([0.2, 0.6, -0.25, 1.5])          # mp / 2 within +-1.5
    assert g[:4, I["wavering"]].tolist() == [0, 1, 1, 0]                                 # wavering, routing
    assert g[:4, I["fatigue"]].tolist() == pytest.approx([0.4, 0, 1, 0.2])
    assert g[:4, I["ammo_come"]].tolist() == pytest.approx([0.25, 0, 0, 0])              # the slingers spent all
    assert not g[4, ai.OLD_SIZE:].any()                                                  # an empty slot: zeros
    armed = ai.armed(got)[0]
    assert armed[:4].tolist() == [True, False, False, False]                             # empty slingers: no shooter


def test_a_hidden_enemy_and_a_field_not_read_keep_the_fresh_units_numbers():
    num, lab, full = table()
    nan = torch.full((1, 5), float("nan"))
    got = units.dynamic(num, full, state(full, fat=nan), OWN, torch.zeros(1, 5, dtype=torch.bool))[0]
    fresh = num[0]
    for k in ai.STATE:
        assert torch.equal(got[2:, I[k]], fresh[2:, I[k]]), k                            # not seen
    assert torch.equal(got[:, I["fatigue"]], fresh[:, I["fatigue"]])                     # fatigue not read
    assert fresh[2, I["ammo_come"]] == 1 and fresh[0, I["morale"]] == 0.5                # full ammunition, morale 1
    bare = {k: v for k, v in state(full).items() if k in ("men", "hp", "hp0")}
    assert torch.equal(units.dynamic(num, full, bare, OWN)[..., ai.OLD_SIZE:], num[..., ai.OLD_SIZE:])


def test_the_threat_counts_a_shooter_only_with_ammunition():
    cfg = gconfig.GeneralConfig(card=16, reader_hidden=16, council_heads=2)
    gen = gnet.General(cfg, 4)
    G = cfg.places
    P, N = 1, 4
    num = torch.zeros(P, N, ai.SIZE)
    num[0, :, I["range"]] = 0.3
    num[0, :, I["ammo_come"]] = 1.0
    pos = torch.zeros(P, N, len(gnet.POS_FEATURES))
    pos[..., -1] = 1.0
    pos[0, 2:, 0] = 0.2
    inp = {"num": num, "pos": pos, "on": torch.ones(P, N, dtype=torch.bool),
           "own": torch.tensor([[True, True, False, False]])}
    c = {"main": torch.tensor([[0, 0, G, G]])}
    j = gnet.THREAT.index("own_shooters"), gnet.THREAT.index("enemy_shooters")
    th = gen.threat(inp, c)[0, 0]
    assert th[j[0]] == 1 and th[j[1]] == 1
    num[0, [0, 2, 3], I["ammo_come"]] = 0.0
    th = gen.threat(inp, c)[0, 0]
    assert th[j[0]] == pytest.approx(0.5) and th[j[1]] == 0


def test_an_older_reader_loads_with_zero_weights_and_reads_the_same_cards():
    torch.manual_seed(0)
    old = ar.ArmyReader(hidden=32, card=16, numbers=ai.OLD_SIZE)
    new = ar.ArmyReader(hidden=32, card=16)
    new.load_state_dict(old.state_dict())
    w = new.layer1.weight
    assert torch.all(w[:, ai.OLD_SIZE:] == 0) and torch.equal(w[:, :ai.OLD_SIZE], old.layer1.weight)
    a = ai.build({1: KEYS[:2], 2: KEYS[2:4]})
    x = ai.to_tensors(a)
    with torch.no_grad():
        base = new(*x)
        moved = x[0].clone()
        moved[..., ai.OLD_SIZE:] = torch.rand(moved.shape[:-1] + (len(ai.STATE),))
        assert torch.equal(new(moved, x[1], x[2]), base)                                 # whatever the state
        torch.testing.assert_close(old(x[0][..., :ai.OLD_SIZE], x[1], x[2]), base, rtol=0, atol=1e-6)


def test_an_older_general_plans_as_before():
    """A whole general saved before the state group: loaded, its cards and plan do not move with the state's values
    (only the "shoot" mask and the shooters' shares move, by the ammunition: here every shooter keeps some)."""
    from tests.tools.test_general_net_torch import CFG, SECT, inputs
    torch.manual_seed(0)
    sd = gnet.General(CFG, SECT).state_dict()
    sd = {k: (v[:, :ai.OLD_SIZE].clone() if k == "reader.layer1.weight" else v) for k, v in sd.items()}
    gen = gnet.General(CFG, SECT).double().eval()
    gen.load_state_dict(sd)
    inp = inputs()
    inp["num"][..., ai.OLD_SIZE:] = 0
    inp["num"][..., I["ammo_come"]] = 1.0
    moved = dict(inp, num=inp["num"].clone())
    moved["num"][..., [I["morale"], I["wavering"], I["fatigue"]]] = torch.rand(inp["num"].shape[:-1] + (3,),
                                                                                dtype=torch.float64)
    moved["num"][..., I["ammo_come"]] = 0.5
    with torch.no_grad():
        torch.manual_seed(5)
        ca, a = gen(inp)
        torch.manual_seed(5)
        cb, b = gen(moved)
    assert torch.equal(ca["groups"], cb["groups"]) and torch.equal(ca["shares"], cb["shares"])
    for k in ("kind", "target", "place", "lp", "value"):
        assert torch.equal(a[k], b[k]), k

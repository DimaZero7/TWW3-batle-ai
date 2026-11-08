"""Drill S-1 'snipers under cover' (tools/nn/strategist/drill_s1.py): the generator, the deployment on the game's zones,
the chain from the strategist's groups, the measures."""
import random

import pytest

torch = pytest.importorskip("torch")

from tools.nn import arena_map  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

pytestmark = pytest.mark.skipif(not (arena_map.MAPS / "ksl_plains_01" / "meta.json").exists(),
                                reason="build/maps/ksl_plains_01 (the captured map) is not here")
P = load()


def test_the_generator_gives_small_armies_with_snipers_and_fast_enemies():
    from tools.nn.strategist import drill_s1 as D
    sp = D.specs(40, random.Random(3))
    for s in sp:
        mine, foes = s["battle"][1], s["battle"][2]
        assert 9 <= len(mine) <= 12 and mine[0]["general"] and not any(u["general"] for u in mine[1:])
        assert 1 <= sum(D.role(u["key"]) == "sniper" for u in mine) <= 3
        assert foes[0]["general"] and 5 <= len(foes) <= 14
        assert sum(D.role(u["key"]) == "fast" for u in foes) == s["n_fast"] <= 4
        # the enemy stands in its zone's half of the field, my zone faces it
        mx, ex = s["mine"]["x"], s["theirs"]["x"]
        assert all((u["x"] - mx) * (ex - mx) > 0 for u in foes)
        assert abs(((s["mine"]["b"] - s["theirs"]["b"]) % 360) - 180) < 1e-6
    assert {s["src"] for s in sp} == {"m1", "m2", "m3"}
    assert {s["rot"] for s in sp} == {True, False} and {s["mirror"] for s in sp} == {True, False}
    assert {s["enemy"] for s in sp} == {"ai_game", "nearest", "hold_shoot"}
    assert {s["flank"] for s in sp if s["n_fast"]} == {"left", "right", "both"}


def test_the_teacher_deploys_every_drill_battle_inside_its_zone_and_the_exposed_puts_snipers_ahead():
    from tools.nn.strategist import drill_s1 as D
    from tools.nn.strategist import zones as ZS
    sp = D.specs(6, random.Random(5))
    b = D.Batch(sp, P, "cpu")
    for who in ("teacher", "teacher0", "exposed"):
        out = D.decide(b.inp, b.ctx, who)
        battles, g0, pl = D.placed(b, out)
        s, f = ZS.local(b.mine_zone, pl["x"], pl["z"])
        rl = D.roles_of_state(b.st0)
        for k in range(len(sp)):
            inf = [i for i, r in enumerate(rl[k]) if r in ("line", "anti") and i < len(sp[k]["battle"][1])]
            sn = [i for i, r in enumerate(rl[k]) if r == "sniper" and i < len(sp[k]["battle"][1])]
            ahead = max(float(f[k, i]) for i in sn) - float(f[k, inf].median())
            if who == "exposed":
                assert ahead > 10.0, (k, ahead)
            assert int(g0.on[k].sum()) >= 3
        assert int(out["plan"]["no_cell"].sum()) == 0


def test_a_short_battle_keeps_the_strategists_groups_and_counts_the_measures():
    from tools.nn.strategist import drill_s1 as D
    from tools.nn.unit import net as Nt
    sp = D.specs(4, random.Random(8))
    b = D.Batch(sp, P, "cpu")
    out = D.decide(b.inp, b.ctx, "teacher")
    torch.manual_seed(0)
    m, st = D.play(b, out, Nt.UnitNet().eval(), P, "cpu", limit=6)
    assert float(m["deploy_errors"].sum()) == 0
    assert st["side1"]["reforms"] == 0                    # the operative's army teacher keeps the strategist's groups
    assert torch.all((m["sniper_kept"] >= 0) & (m["sniper_kept"] <= 1))
    assert set(m) >= {"sniper_kept", "sn_first", "fast60_s", "trade", "score"}


def test_the_watch_finds_the_first_unit_to_rout():
    from types import SimpleNamespace
    from tools.nn.strategist import drill_s1 as D
    sn = torch.tensor([[True, False, False]])
    w = D.Watch(sn, torch.tensor([[False, False, True]]), at=2)
    u = {"r": torch.tensor([[False, False, False]]), "men": torch.tensor([[10, 10, 10]]),
         "gone": torch.tensor([[False, False, False]]), "x": torch.tensor([[0.0, 10.0, 100.0]]),
         "z": torch.zeros(1, 3)}
    st = SimpleNamespace(u=u, B=1, device="cpu")
    w(0, st)
    u["r"] = torch.tensor([[False, True, False]])
    u["x"] = torch.tensor([[0.0, 10.0, 50.0]])                   # the fast enemy 50 m from my sniper
    w(1, st)
    assert w.rout.tolist() == [[-1.0, 2.0, -1.0]] and w.fast60.tolist() == [2.0]
    assert w.men_at.tolist() == [[10.0, 0.0, 10.0]]              # a routed unit's men count 0

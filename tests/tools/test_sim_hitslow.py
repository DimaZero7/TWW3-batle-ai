"""The slowing from hits of gunpowder (config/nn/sim.json missile.hit_slow; docs/en/training/simulator.md "Hits of
gunpowder slow"): projectiles of the database's category musket (handguns, the Ratling Gun) slow the small formations
they hit - hit_slow.run of their speed while they run, hit_slow.walk while they walk - for hit_slow.duration_s after
the last step such hits land in; bolts (category arrow) do not (the hitslow probe). Needs torch."""
import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import battle, missile, scenario  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

P = load()
HGUN, XBOW, RATLING = "wh_main_emp_inf_handgunners", "wh_main_emp_inf_crossbowmen", "wh2_dlc12_skv_inf_ratling_gun_0"
CLANRATS, GENERAL = "wh2_main_skv_inf_clanrats_0", "wh_main_emp_cha_general_0"
HS = P.sim["missile"]["hit_slow"]
ON = P.with_cal("missile", hit_slow=dict(HS, on=True))
OFF = P.with_cal("missile", hit_slow=dict(HS, on=False))


def lane(shooter, params, d=120.0):
    """The shooter (side 1, the Empire) at (0, 0) facing north, clanrats (side 2) d m north of it facing south."""
    return scenario.build([{"attacker": 2, "sides": {
        1: {"faction": "wh_main_emp_empire", "units": [{"key": shooter, "x": 0.0, "z": 0.0, "b": 0.0}],
            "skirmish": False},
        2: {"faction": "wh2_main_skv_skaven", "units": [{"key": CLANRATS, "x": 0.0, "z": d, "b": 180.0}],
            "skirmish": False}}}], params)


def run_at(st, params, seconds, run=True):
    """The clanrats move at a run (walk) to a point far south; the shooter holds (fires at will). -> z of the
    clanrats [per step] and their hs_s [per step]."""
    H = st.N // 2
    zs, hs = [], []
    for _ in range(int(round(seconds / params.dt))):
        o = O.hold(st.B, st.N, st.device)
        o.kind[0, H] = O.MOVE
        o.x[0, H], o.z[0, H] = 0.0, -400.0
        o.run[0, H] = run
        battle.step(st, o, params)
        zs.append(float(st.u["z"][0, H]))
        hs.append(float(st.u["hs_s"][0, H]))
    return zs, hs


class TestStatic:
    def test_gunpowder_slows_bolts_do_not(self):
        st = lane(HGUN, ON)
        assert bool(st.u["hslow"][0, 0])
        assert not bool(lane(XBOW, ON).u["hslow"][0, 0])
        assert not bool(lane(HGUN, OFF).u["hslow"][0, 0])

    def test_the_ratling_gun_is_gunpowder_too(self):
        assert P.units[RATLING]["missile"]["category"] in HS["categories"]


class TestTimer:
    def test_a_hit_starts_it_and_it_runs_down(self):
        st = lane(HGUN, ON, d=100.0)
        zs, hs = run_at(st, ON, 20.0)
        first = next(k for k, h in enumerate(hs) if h > 0)
        assert hs[first] == pytest.approx(HS["duration_s"])
        assert max(hs) <= HS["duration_s"] + 1e-6

    def test_off_never_starts_it(self):
        st = lane(HGUN, OFF, d=100.0)
        _, hs = run_at(st, OFF, 20.0)
        assert max(hs) == 0.0


class TestSlowing:
    @pytest.mark.parametrize("run", [True, False])
    def test_handguns_slow_the_runners(self, run):
        """Under the handgunners' fire the clanrats cover less ground than with the rule off, by about the rule's
        factor over the seconds it is on."""
        s = 25.0 if run else 40.0
        z_on, hs = run_at(lane(HGUN, ON, d=140.0), ON, s, run)
        z_off, _ = run_at(lane(HGUN, OFF, d=140.0), OFF, s, run)
        assert any(h > 0 for h in hs)
        assert z_on[-1] > z_off[-1] + 1.0          # they come south (z falls): slowed ones are further north

    def test_bolts_do_not(self):
        z_on, hs = run_at(lane(XBOW, ON, d=140.0), ON, 25.0)
        z_off, _ = run_at(lane(XBOW, OFF, d=140.0), OFF, 25.0)
        assert max(hs) == 0.0
        assert z_on[-1] == pytest.approx(z_off[-1], abs=1e-3)


class TestFactor:
    def test_run_walk_and_who(self):
        st = lane(HGUN, ON)
        u = st.u
        u["hs_s"].fill_(5.0)
        fast = torch.ones_like(u["hs_s"], dtype=torch.bool)
        f_run = missile.hit_slow_factor(u, ON, fast)
        f_walk = missile.hit_slow_factor(u, ON, ~fast)
        H = st.N // 2
        assert float(f_run[0, H]) == pytest.approx(HS["run"]) and float(f_walk[0, H]) == pytest.approx(HS["walk"])
        u["large"][0, H] = True                    # large: not measured, not slowed
        assert float(missile.hit_slow_factor(u, ON, fast)[0, H]) == 1.0
        u["large"][0, H] = False
        u["men0"][0, H] = 1.0                      # a lone man (a lord): ignores most hit reactions
        assert float(missile.hit_slow_factor(u, ON, fast)[0, H]) == 1.0
        u["men0"][0, H] = 80.0
        u["hs_s"].fill_(0.0)
        assert float(missile.hit_slow_factor(u, ON, fast)[0, H]) == 1.0
        assert float(missile.hit_slow_factor(u, OFF, fast)[0, H]) == 1.0

"""tools/nn/train/opponents.py `human`: the scripted defence in depth modelled on the human's battles against g31 and
g34 - the line wider than the enemy's holds and steps once to meet it, the reserve waits behind it until the enemy's
first rout and goes in one at a time where the line is worst, the shooters shoot the enemy's shooters first and stand
off a flank, forward along it once the melee begins; registered at the end of the league's opponents (the others'
codes kept), not in the mix; a battle plays.
torch: the training container, on the CPU."""
import pytest

torch = pytest.importorskip("torch")

from tools.nn.model import config, critic, policy             # noqa: E402
from tools.nn.sim import battle, scenario                      # noqa: E402
from tools.nn.sim import orders as O                           # noqa: E402
from tools.nn.sim.params import load                           # noqa: E402
from tools.nn.train import drills, league, opponents, rollout  # noqa: E402

CFG = config.preset("small", d=32, layers=2, heads=2, pointer=16, critic_d=32, critic_layers=1, critic_heads=2)


def army(gap=600.0, attacker=2):
    """Side 1 (west): a general, five spearmen in a line north to south, an archer behind; side 2 (east) the same."""
    def side(sign, faces):
        x0 = sign * gap / 2
        units = [{"key": "wh_main_emp_cha_general_0", "general": True, "x": x0 + sign * 40, "z": 0, "b": faces}]
        units += [{"key": "wh_main_emp_inf_spearmen_0", "x": x0, "z": zz, "b": faces} for zz in (-80, -40, 0, 40, 80)]
        units += [{"key": "wh2_dlc13_emp_inf_archers_0", "x": x0 + sign * 30, "z": 0, "b": faces}]
        return {"faction": "wh_main_emp_empire", "units": units}
    return scenario.build([{"attacker": attacker, "sides": {1: side(-1, 90), 2: side(1, 270)}}])


def test_human_is_registered_at_the_end_and_not_in_the_mix():
    assert opponents.SCRIPTS["human"] is opponents.human
    assert league.OPPONENTS[-1] == "human" and "human" not in league.MIX
    # the older codes stay where they were
    assert [league.CODE[n] for n in ("self", "past", "nearest", "hold_shoot", "hold", "ai_like")] == [1, 2, 3, 4, 5, 6]
    assert [league.CODE[drills.opponent(n)] for n in drills.NAMES] == list(range(7, 7 + len(drills.NAMES)))


def centred(gap=600):
    """army() with side 1's line (slots 1-3) at z -40, 0, 40 and its reserve (4, 5) at -80, 80: both centres on z 0,
    the forward exactly +x."""
    st = army(gap=gap)
    st.u["z"][0, 1:6] = torch.tensor([-40.0, 0.0, 40.0, -80.0, 80.0])
    return st


def test_the_line_widens_the_reserve_waits_behind_and_the_shooters_go_to_a_flank():
    st = centred()
    o = opponents.human(st)
    O.check(o, st.N)
    # side 1: slot 0 the general, 1-5 spearmen (1-3 the line: 58 % of 5), 6 the archer; forward +x, "left" +z
    # the enemy's span 160 m (its spearmen -80..80): the line's posts 1.3 x 160 = 208 m across its centre z 0, in the
    # order the units stand, level with it (the middle one is there: holds)
    assert int(o.kind[0, 1]) == O.MOVE and int(o.kind[0, 3]) == O.MOVE and int(o.kind[0, 2]) == O.HOLD
    zs = st.u["z"][0, st.N // 2:]
    half = 1.3 * float(zs.max() - zs.min()) / 2                             # (the builder's places: ~160 m)
    assert 100 < half < 105 and o.z[0, [1, 3]].tolist() == pytest.approx([-half, half], abs=0.01)
    assert (o.x[0, [1, 3]] - (-300.0)).abs().max() < 0.01
    assert (o.kind[0, 4:6] == O.MOVE).all() and (o.x[0, 4:6] < -300 - 30).all()   # the reserve behind the line
    # the archer (at z 0: the left end): 50 m beyond the end of the widened line (its half 104)
    assert int(o.kind[0, 6]) == O.MOVE and float(o.z[0, 6]) == pytest.approx(half + 50, abs=0.5)
    assert abs(float(o.x[0, 6]) - (-300.0)) < 1.0                           # ... level with it (the forward: to the enemy)
    side1 = st.u["side"][0] == 1
    assert not bool((o.kind[0][side1] == O.ATTACK).any())                   # no attack before contact
    assert opponents.Human().line_share == 0.58 and opponents.Human().widen == 1.3
    # inside the map: a map of half-size 120 m keeps every post 30 m inside its edge
    st.bounds = 120.0
    o = opponents.human(st)
    moving = o.kind[0] == O.MOVE
    assert bool(moving.any()) and float(o.z[0][moving].abs().max()) <= 90.0 + 1e-4


def test_the_line_steps_once_to_meet_the_enemy():
    """The enemy's nearest melee unit within step_far_m (175) of the line's centre and farther than step_near_m (120):
    the free line walks step_m (57) forward of its posts; a step in force is kept; within 120 m - no step."""
    st = centred()
    u = st.u
    H = st.N // 2
    u["x"][0, H:] = u["x"][0, H:] - 450.0                     # the enemy at x -150 +- : 150 m from our line at -300
    o = opponents.human(st)
    for i in (1, 2, 3):
        assert int(o.kind[0, i]) == O.MOVE and not bool(o.run[0, i])
        assert float(o.x[0, i]) == pytest.approx(-300.0 + 57.0, abs=0.5)
    # in force (its point 57 m ahead): kept as it is, whatever the posts now
    u["order_kind"][0, 1:4] = O.MOVE
    u["ox"][0, 1:4], u["oz"][0, 1:4] = -243.0, 7.0                     # (x 57 m ahead of -300)
    o = opponents.human(st)
    assert (o.kind[0, 1:4] == O.MOVE).all() and torch.allclose(o.x[0, 1:4], torch.full((3,), -243.0))
    assert torch.allclose(o.z[0, 1:4], torch.full((3,), 7.0))
    # the enemy within 120 m: no step (posts level with the line)
    u["order_kind"][0, 1:4] = O.HOLD
    u["x"][0, H:] = u["x"][0, H:] + 50.0 - 100.0              # 100 m off
    o = opponents.human(st)
    assert float(o.x[0, 1:4].max()) < -300.0 + 10.0


def test_the_reserve_waits_for_the_enemys_first_rout_then_goes_in_one_at_a_time_where_the_line_is_worst():
    st = army(gap=600)
    u = st.u
    H = st.N // 2
    # the line's spearmen 1 and 3 fight the enemy's spearmen H+1 and H+3; 3 is worse (health x morale)
    for mine, foe, hp in ((1, H + 1, 0.9), (3, H + 3, 0.4)):
        u["x"][0, foe], u["z"][0, foe] = u["x"][0, mine] + 5.0, u["z"][0, mine]
        u["m"][0, mine], u["m"][0, foe] = True, True
        u["target"][0, mine], u["target"][0, foe] = foe, mine
        u["hp"][0, mine] = hp
    o = opponents.human(st)
    O.check(o, st.N)
    assert not bool((o.kind[0, 4:6] == O.ATTACK).any())                     # no enemy has routed: the reserve waits
    assert int(o.kind[0, 1]) == O.ATTACK and int(o.target[0, 1]) == H + 1   # the line fights its enemy
    u["r"][0, H + 5] = True                                                  # an enemy unit routs
    o = opponents.human(st)
    sent = [i for i in (4, 5) if int(o.kind[0, i]) == O.ATTACK]
    assert len(sent) == 1 and int(o.target[0, sent[0]]) == H + 3            # one reserve unit, at the worst place
    # its attack in force, not there yet: no second one
    u["order_kind"][0, sent[0]], u["order_target"][0, sent[0]] = O.ATTACK, H + 3
    o = opponents.human(st)
    other = 9 - sent[0]
    assert int(o.kind[0, sent[0]]) == O.ATTACK and int(o.kind[0, other]) != O.ATTACK


def test_the_shooters_shoot_the_enemys_shooters_first_then_walk_along_the_flank():
    st = centred()
    u = st.u
    H = st.N // 2
    rng = float(u["range"][0, 6])
    assert rng > 0
    # the enemy's archer (H + 6) within our archer's range: it is the target
    u["x"][0, H + 6], u["z"][0, H + 6] = float(u["x"][0, 6]) + rng - 10.0, float(u["z"][0, 6])
    o = opponents.human(st)
    assert int(o.kind[0, 6]) == O.ATTACK and int(o.target[0, 6]) == H + 6
    # out of range again, our line in melee: the post flank_fwd_m (65) forward along the flank
    u["x"][0, H + 6] = 300.0
    before = opponents.human(st)
    u["m"][0, 1], u["m"][0, H + 1] = True, True                              # (the flags: nobody moved)
    u["target"][0, 1], u["target"][0, H + 1] = H + 1, 1
    after = opponents.human(st)
    assert int(after.kind[0, 6]) == O.MOVE
    assert float(after.x[0, 6]) - float(before.x[0, 6]) == pytest.approx(65.0, abs=1.0)


def test_a_battle_against_human_plays():
    P = load()
    st = army(gap=200, attacker=2)
    men0 = st.u["men"].clone()
    contact = False
    for _ in range(240):                                                     # 120 s: ai_like attacks, human holds
        o = O.merge(opponents.human(st), opponents.ai_like(st), st.u["side"] == 2)
        O.check(o, st.N)
        battle.step(st, o, P)
        contact = contact or bool(st.u["m"].any())
    assert contact
    lost = men0 - st.u["men"]
    assert float(lost[0][st.u["side"][0] == 1].sum()) > 0 and float(lost[0][st.u["side"][0] == 2].sum()) > 0
    # ... and through the league's battles (the learner against the script)
    torch.manual_seed(0)
    actor, crit = policy.Actor(CFG).eval(), critic.Critic(CFG).eval()
    env = rollout.Battles(league.layout(4, 1, opponent="human"), [("arena", "attack")],
                          params=rollout.params_with_limit(20.0))
    for _ in range(3):
        env.step(actor, crit)
    assert bool((env.ctrl == league.CODE["human"]).any())


def custom(side1_units, attacker=2, gap=200.0):
    """Side 1 (human, west) of side1_units [(key, general, dx, z)], side 2 the usual army (east)."""
    units = [{"key": k, "general": g, "x": -gap / 2 + dx, "z": zz, "b": 90} for k, g, dx, zz in side1_units]
    east = {"faction": "wh_main_emp_empire", "units": [
        {"key": "wh_main_emp_cha_general_0", "general": True, "x": gap / 2 + 40, "z": 0, "b": 270}]
        + [{"key": "wh_main_emp_inf_spearmen_0", "x": gap / 2, "z": zz, "b": 270} for zz in (-40, 0, 40)]}
    return scenario.build([{"attacker": attacker, "sides": {1: {"faction": "wh_main_emp_empire", "units": units},
                                                             2: east}}])


SPEAR, ARCHER, LORD = "wh_main_emp_inf_spearmen_0", "wh2_dlc13_emp_inf_archers_0", "wh_main_emp_cha_general_0"


@pytest.mark.parametrize("units", [
    [(LORD, True, -40, 0), (SPEAR, False, 0, -40), (SPEAR, False, 0, 0), (SPEAR, False, 0, 40)],   # no shooters
    [(LORD, True, -40, 0), (ARCHER, False, 0, -30), (ARCHER, False, 0, 30)],                     # no infantry
    [(SPEAR, False, 0, 0)],                                                                       # one unit
    [(LORD, True, 0, 0)],                                                                         # the lord alone
], ids=["no_shooters", "no_infantry", "one_unit", "lord_alone"])
def test_a_battle_against_human_plays_with_any_army(units):
    P = load()
    st = custom(units)
    for _ in range(160):                                                     # 80 s
        o = O.merge(opponents.human(st), opponents.ai_like(st), st.u["side"] == 2)
        O.check(o, st.N)
        assert torch.isfinite(o.x).all() and torch.isfinite(o.z).all()
        battle.step(st, o, P)
    assert torch.isfinite(st.u["x"]).all()

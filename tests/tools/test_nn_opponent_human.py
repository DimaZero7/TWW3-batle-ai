"""tools/nn/train/opponents.py `human`: the scripted defence in depth modelled on the human's battle against g31 - the
line holds, the reserve waits behind it and goes in one at a time where the line is worst, the shooters stand off a
flank; registered at the end of the league's opponents (the others' codes kept), not in the mix; a battle plays.
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


def test_the_line_holds_the_reserve_waits_behind_and_the_shooters_go_to_a_flank():
    st = army()
    o = opponents.human(st)
    O.check(o, st.N)
    # side 1: slot 0 the general, 1-5 spearmen (1-3 the line: 60 % of 5), 6 the archer
    assert (o.kind[0, 1:4] == O.HOLD).all()                                  # the line stands
    assert (o.kind[0, 4:6] == O.MOVE).all() and (o.x[0, 4:6] < -300 - 30).all()   # the reserve behind the line
    assert int(o.kind[0, 6]) == O.MOVE and float(o.z[0, 6]) == pytest.approx(0 + 50, abs=3)   # 50 m beyond the end
    assert abs(float(o.x[0, 6]) - float(st.u["x"][0, 1:4].mean())) < 10.0  # ... level with it (the forward: to the enemy)
    side1 = st.u["side"][0] == 1
    assert not bool((o.kind[0][side1] == O.ATTACK).any())                   # no attack before contact
    # the line's units: the first 60 % of the melee slots, whatever their places
    assert opponents.Human().line_share == 0.6


def test_the_reserve_goes_in_one_at_a_time_where_the_line_is_worst():
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
    sent = [i for i in (4, 5) if int(o.kind[0, i]) == O.ATTACK]
    assert len(sent) == 1 and int(o.target[0, sent[0]]) == H + 3            # one reserve unit, at the worst place
    assert int(o.kind[0, 1]) == O.ATTACK and int(o.target[0, 1]) == H + 1   # the line fights its enemy
    # its attack in force, not there yet: no second one
    u["order_kind"][0, sent[0]], u["order_target"][0, sent[0]] = O.ATTACK, H + 3
    o = opponents.human(st)
    other = 9 - sent[0]
    assert int(o.kind[0, sent[0]]) == O.ATTACK and int(o.kind[0, other]) != O.ATTACK


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

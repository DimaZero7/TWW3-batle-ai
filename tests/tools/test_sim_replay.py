"""Replay phase contracts, independent of combat balance and recorded battle fixtures."""
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn import gamedata
from tools.nn.sim import orders as O, replay


def state(batch=1):
    u = {k: torch.zeros(batch, 2, dtype=torch.bool) for k in ("m", "r", "s", "gone")}
    u.update(side=torch.tensor([[1, 2]] * batch), men=torch.full((batch, 2), 100.))
    return SimpleNamespace(B=batch, N=2, device="cpu", t=torch.zeros(batch), u=u)


def rows():
    return dict(kind=np.array([[O.ATTACK, O.HOLD], [O.ATTACK, O.HOLD],
                               [O.ATTACK, O.HOLD], [O.MOVE, O.HOLD], [O.HOLD, O.HOLD]]),
                target=np.array([[1, -1]] * 3 + [[-1, -1]] * 2),
                x=np.full((5, 2), 100., dtype=np.float32), z=np.zeros((5, 2), dtype=np.float32),
                run=np.array([[True, False]] * 2 + [[False, False]] * 3),
                phase=np.array([[1, 0]] * 2 + [[0, 0], [2, 0], [0, 0]]),
                end=np.array([[2, 1], [2, 2], [3, 3], [4, 4], [5, 5]]),
                phase_target=np.array([[1, -1]] * 3 + [[-1, -1]] * 2))


def test_late_contact_keeps_running_attack_and_target_then_plays_melee_duration():
    st = state(); policy = replay.Replay([rows()])
    policy(st)
    for sec in (1., 2., 3., 4.):
        st.t[:] = sec
        o = policy(st)
        assert o.kind[0, 0] == O.ATTACK and o.target[0, 0] == 1 and o.run[0, 0]
    st.u["m"][0, 0] = True
    st.t[:] = 4.5
    o = policy(st)
    assert o.kind[0, 0] == O.ATTACK and not o.run[0, 0]
    st.t[:] = 5.
    assert policy(st).kind[0, 0] == O.ATTACK
    st.t[:] = 5.5
    assert policy(st).kind[0, 0] == O.MOVE


def test_running_move_keeps_route_until_contact_is_late_then_follows_recorded_target():
    r = rows(); r["kind"][:2, 0] = O.MOVE; r["target"][:2, 0] = -1
    st = state(); policy = replay.Replay([r])
    assert policy(st).kind[0, 0] == O.MOVE
    st.t[:] = 1.
    assert policy(st).kind[0, 0] == O.MOVE
    st.t[:] = 2.
    o = policy(st)
    assert o.kind[0, 0] == O.ATTACK and o.target[0, 0] == 1 and o.run[0, 0]


def test_early_contact_and_breakoff_wait_for_separation():
    st = state(); policy = replay.Replay([rows()])
    policy(st)
    st.t[:] = .5; st.u["m"][0, 0] = True
    assert not policy(st).run[0, 0]
    for sec in (1.5, 2.5, 3.5):
        st.t[:] = sec
        o = policy(st)
        assert o.kind[0, 0] == O.MOVE and o.x[0, 0] == 100
    st.t[:] = 4.; st.u["m"][0, 0] = False
    assert policy(st).kind[0, 0] == O.HOLD


@pytest.mark.parametrize("field", ["r", "s", "gone", "men"])
def test_unavailable_target_releases_approach(field):
    st = state(); policy = replay.Replay([rows()])
    policy(st)
    st.u[field][0, 1] = 0 if field == "men" else True
    st.t[:] = .5
    assert not policy(st).run[0, 0]


def test_copies_have_independent_clocks_and_time_reset_restarts_replay():
    st = state(2); policy = replay.Replay([rows(), rows()])
    policy(st)
    st.t[:] = .5; st.u["m"][0, 0] = True
    o = policy(st)
    assert not o.run[0, 0] and o.run[1, 0]
    st.t[0] = 0; st.u["m"][0, 0] = False
    assert policy(st).run[:, 0].all()


def test_wall_clock_grace_expires_even_when_contact_never_happens():
    st = state(); policy = replay.Replay([rows()], after=replay.hold, grace_s=1.)
    policy(st)
    st.t[:] = 6.
    assert (policy(st).kind == O.HOLD).all()


def test_routing_actor_releases_its_latch_without_changing_other_units():
    st = state(); policy = replay.Replay([rows()])
    policy(st)
    st.t[:] = .5; st.u["r"][0, 0] = True
    o = policy(st)
    assert not o.run[0, 0] and o.kind[0, 1] == O.HOLD


def test_legacy_rows_stay_clock_indexed_and_last_row_is_padded_per_battle():
    a = {k: v for k, v in rows().items() if k not in ("phase", "end", "phase_target")}
    b = {k: v[:2] for k, v in a.items()}
    st = state(2); policy = replay.Replay([a, b])
    st.t[:] = 4.
    assert policy(st).kind[:, 0].tolist() == [O.HOLD, O.ATTACK]


def recording():
    T, N = 7, 2
    f = {k: np.zeros((T, N)) for k in gamedata.FLOAT_FIELDS}
    f.update({k: np.zeros((T, N), dtype=bool) for k in gamedata.BOOL_FIELDS})
    f["men"][:] = 100; f["x"][:, 1] = 10; f["ox"][:] = 100
    f["f"][:2] = True; f["m"][2:5] = True
    tg = np.full((T, N), -1); tg[2:4] = [1, 0]
    return gamedata.Battle(run="test", own_ai="net", enemy_role="attack", result={},
                           t=np.arange(T), f=f, target=tg, names=("a", "b"), keys=("", ""),
                           side=np.array([1, 2]))


def test_inference_preserves_leaver_roles_and_does_not_wrap_run_lookahead():
    b = recording()
    r = replay.recorded_orders(b, [0, 1], 2, fight_nearest=[False, True], leavers=[True, False])
    assert (r["phase"][:2] == 1).all()
    assert r["phase"][4].tolist() == [2, 0]
    assert r["kind"][4].tolist() == [O.MOVE, O.ATTACK]
    # The AI's far point is not an exit while recorded in melee. Once the
    # recording shows separation, its ordinary MOVE may wait for actual separation.
    assert r["phase"][5].tolist() == [2, 2]
    assert not r["run"][-2:].any()
    disabled = replay.recorded_orders(b, [0, 1], 2, leave_m=0)
    assert not (disabled["phase"] == 2).any()


def test_cancelled_move_in_recorded_melee_is_not_an_exit_phase():
    b = recording(); b.target[3, 0] = -1; b.target[4, 0] = 1
    r = replay.recorded_orders(b, [0, 1], 2, leavers=[True, False], fight_nearest=[False, True])
    assert r["kind"][3, 0] == O.MOVE and r["phase"][3, 0] == 0


def test_exit_does_not_skip_a_later_recorded_charge():
    b = recording(); b.f["m"][3:5] = False; b.f["f"][4] = True
    b.target[3:5] = -1
    b.target[5:] = [1, 0]; b.f["m"][5:] = True
    r = replay.recorded_orders(b, [0, 1], 2)
    assert (r["phase"][3] == 2).all() and (r["end"][3] == 4).all()
    assert (r["phase"][4] == 1).all() and (r["end"][4] == 5).all()


def test_shooting_attack_without_recorded_melee_is_not_latched():
    b = recording(); b.f["m"][:] = False; b.f["fire"][:] = True
    r = replay.recorded_orders(b, [0, 1], 2)
    assert not r["phase"].any()


def test_phase_target_overrides_later_target_only_while_available():
    r = rows(); r["target"][2, 0] = -1
    st = state(); st.t[:] = 2
    policy = replay.Replay([r])
    assert policy(st).target[0, 0] == 1
    st.u["r"][0, 1] = True
    assert policy(st).target[0, 0] == -1

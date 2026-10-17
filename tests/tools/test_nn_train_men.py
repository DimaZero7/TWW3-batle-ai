"""The fighters lab in training (tools/nn/train/men.py, run.py --fighters): the switch is off by default; with it every
battle of the batch has its men - at the start, after a restart (built anew, the other battles untouched), in an
evaluation's smaller batch and in the ability branches (the snapshot's men copied, the branches repeat exactly).

torch is not in the project's .venv: skipped there, run in the training container."""
import contextlib

import pytest

torch = pytest.importorskip("torch")

from tools.nn.fighters import hook, state as FS  # noqa: E402
from tools.nn.model import config, policy  # noqa: E402
from tools.nn.model import critic as mcritic  # noqa: E402
from tools.nn.train import ability_branch as ab, league, men as tm, rollout, run  # noqa: E402

TINY = config.preset("v2", d=32, layers=2, heads=2, pointer=16, critic_d=32, critic_layers=1, critic_heads=2,
                     sector_d=16)
MIRROR = [("arena", "attack")]


def tiny(seed=0):
    torch.manual_seed(seed)
    return policy.Actor(TINY).eval(), mcritic.Critic(TINY).eval()


def lab():
    return hook.installed(shape=True, adopt=False, outside=False)


def env_of(n=4, M=None, mix=None):
    return rollout.Battles(league.layout(n, 1, mix or {"self": 0.5, "hold_shoot": 0.5}), MIRROR, men=tm.Men(M=M))


def matches(st):
    """The men are the units' (men and health), and the network's shape input is there."""
    F = st.men
    live = FS.alive(F)
    N = st.N
    men = FS.unit_count(F, live, N)
    hp = FS.unit_sum(F, torch.where(live, F.hp, torch.zeros_like(F.hp)), N)
    present = st.u["side"] > 0
    ok = torch.equal(torch.where(present, men, torch.zeros_like(men)),
                     torch.where(present, st.u["men"].round(), torch.zeros_like(men)))
    ok = ok and bool(((hp - st.u["hp_abs"]).abs() <= 0.6 + 1e-4 * st.u["hp_abs"])[present].all())
    obs = st.observation()
    return ok and bool((obs["shape_ok"] == (present & (men > 0))).all())


def test_the_switch_is_off_by_default():
    args = run.parser().parse_args([])
    assert args.fighters is False and args.fighters_men == tm.M_DEFAULT
    assert run.fighters_men(args) is None
    assert isinstance(run.fighters_installed(args), contextlib.nullcontext)
    env = rollout.Battles(league.layout(2, 1, {"hold_shoot": 1.0}), MIRROR)
    assert env.men is None and not hasattr(env.st, "men") and "shape" not in env.st.observation()
    on = run.parser().parse_args(["--fighters", "--fighters-men", "4096"])
    m = run.fighters_men(on)
    assert isinstance(m, tm.Men) and m.M == 4096


def test_a_batch_steps_with_its_men_and_the_network_sees_their_shape():
    actor, crit = tiny()
    with lab():
        env = env_of()
        st = env.st
        assert st.men.M % tm.ROUND == 0 and matches(st)
        assert bool(st.observation()["shape_ok"].any()) and float(st.u["shape"].abs().sum()) > 0
        for _ in range(4):
            env.step(actor, crit)
        assert env.st is st and matches(st)
        assert int(st.men.tick) == 4 * env.k                 # every simulator step went through the men


def test_a_restart_builds_the_battles_men_anew_and_leaves_the_others():
    actor, crit = tiny()
    with lab():
        env = env_of()
        for _ in range(6):
            env.step(actor, crit)
        F = env.st.men
        before = {k: getattr(F, k).clone() for k in tm.row_fields(F)}
        hurt = float(env.st.u["hp"][0].sum())
        rows = torch.tensor([True, False, False, False])
        env._reset(rows)
        G = env.st.men
        assert matches(env.st)
        assert float(env.st.u["hp"][0].sum()) >= hurt        # battle 0 whole again
        live = FS.alive(G)[0]
        assert bool((G.hp[0][live] == env.st.u["hp_man"][0].gather(0, G.uid[0][live])).all())   # unhurt men
        assert bool((G.tgt[0] == -1).all()) and bool((G.cd[0] == 0).all())
        for k, v in before.items():
            assert torch.equal(getattr(G, k)[1:], v[1:]), k   # the others untouched
        # the restarted battle's men are those a fresh build gives (their places' scatter drawn anew)
        fresh = FS.build(tm._Rows(env.st, torch.tensor([0])), env.params, M=G.M, L=tm.L_DEFAULT,
                         splash_S=tm.SPLASH_S)
        for k in ("uid", "valid", "file", "rank0", "ustart", "order", "seg_first", "hp", "splash_men"):
            assert torch.equal(getattr(G, k)[0], getattr(fresh, k)[0]), k
        env.step(actor, crit)
        assert matches(env.st)


def test_a_battle_too_big_for_the_batch_is_picked_again_or_refused():
    with lab():
        env = env_of(M=None)
        need = int(tm.men_of(env.st).max())
        m = tm.Men(M=need - 1)
        assert bool(m.too_big(env.st, torch.ones(env.B, dtype=torch.bool)).all())
        assert not bool(tm.Men(M=need).too_big(env.st, torch.ones(env.B, dtype=torch.bool)).any())
        with pytest.raises(RuntimeError, match="fighters"):
            env_of(M=need - 1)                                 # the arena's battles never fit


def test_the_evaluations_smaller_batch_keeps_its_battles_men():
    actor, crit = tiny()
    with lab():
        env = env_of()
        for _ in range(3):
            env.step(actor, crit)
        F = env.st.men
        keep = env.narrow([1, 3])
        G = env.st.men
        assert G is not F and G.B == 2 and env.st.men is G
        for k in tm.row_fields(F):
            assert torch.equal(getattr(G, k), getattr(F, k)[keep]), k
        env.step(actor, None)
        assert matches(env.st)


class Grab:
    def __init__(self):
        self.got = None

    def capture(self, env, info):
        if self.got is None:
            self.got = ab.snapshot(env, torch.arange(env.B), info)


def test_the_ability_branches_copy_the_men_and_repeat_exactly():
    actor, crit = tiny()
    with lab():
        env = env_of()
        env.brancher = Grab()
        F0 = None
        env.step(actor, crit)
        snap = env.brancher.got
        assert any(k.startswith("f/") for k in snap)
        e, parts, scripted = ab.branch_env(env, ab.select(snap, torch.tensor([0, 2])))
        G = e.st.men
        assert G is not env.st.men and G.B == 2 and matches(e.st)
        for k in tm.row_fields(G):
            if "f/" + k in snap:
                assert torch.equal(getattr(G, k), snap["f/" + k][[0, 2]]), k
        e.replay(parts, scripted)
        assert matches(e.st) and env.st.men.B == 4          # the batch's own men stay its own
        # the pair of branches twice: the same prices; the same press in both: zero
        b = torch.tensor([0, 1])
        side = torch.ones(2, dtype=torch.long)
        unit = torch.zeros(2, dtype=torch.long)
        slot = torch.zeros(2, dtype=torch.long)
        one = ab.play_pair(env, actor, ab.select(snap, b), unit, side, slot, 2, 3, seed=5)
        two = ab.play_pair(env, actor, ab.select(snap, b), unit, side, slot, 2, 3, seed=5)
        assert torch.equal(one[0], two[0])
        same = ab.play_pair(env, actor, ab.select(snap, b), unit, side, torch.full_like(slot, -1), 2, 3, seed=5)
        assert torch.count_nonzero(same[0]) == 0
        assert F0 is None


def test_an_evaluation_plays_with_the_men():
    from tools.nn.train import evaluate
    actor, _ = tiny()
    made = []

    def men():
        made.append(tm.Men(M=None))
        return made[-1]
    with lab():
        res = evaluate.play(actor, opponents=("nearest",), per_scene=2, limit_s=20.0, scene_list=MIRROR,
                            measure=(), fighters=men)
    assert made and made[0].M is not None
    assert res["by_opponent"]["nearest"]["games"] == 2


def test_a_state_without_men_steps_plainly_while_the_lab_is_installed():
    from tools.nn.sim import battle, replay, scenario
    from tools.nn.sim.params import load
    P = load()
    a = {"attacker": 1, "sides": {
        1: {"faction": "wh_main_emp_empire", "units": [{"key": "wh_main_emp_inf_swordsmen", "x": -15, "z": 0,
                                                       "b": 90, "width": 30}]},
        2: {"faction": "wh2_main_skv_skaven", "units": [{"key": "wh2_main_skv_inf_clanrats_1", "x": 15, "z": 0,
                                                        "b": 270, "width": 30}]}}}
    plain = scenario.build([a], P)
    torch.manual_seed(0)
    for _ in range(20):
        battle.step(plain, replay.nearest_attack(plain), P, P.dt)
    with lab():
        st = scenario.build([a], P)
        assert not hasattr(st, "men")                        # nothing built is adopted
        torch.manual_seed(0)
        for _ in range(20):
            battle.step(st, replay.nearest_attack(st), P, P.dt)
    for k in ("hp_abs", "men", "x", "z", "morale"):
        assert torch.equal(st.u[k], plain.u[k]), k

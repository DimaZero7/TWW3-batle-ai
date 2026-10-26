"""rollout.Battles._past_own (run.py --opponents: fixed past slots): each slot's version decides for its own rows only -
the past rows' orders, actions, commitments and their own memories exactly those of the all-rows way (every version on
every past row, each row taking its slot's). torch: the training container, on the CPU."""
import pytest

torch = pytest.importorskip("torch")

from tools.nn.model import config, policy                       # noqa: E402
from tools.nn.model import critic as mcritic                    # noqa: E402
from tools.nn.sim import orders as O                            # noqa: E402
from tools.nn.train import league, randomise, rollout           # noqa: E402

TINY = config.preset("v2", d=32, layers=2, heads=2, pointer=16, critic_d=32, critic_layers=1, critic_heads=2,
                     sector_d=16)
OTHER = config.preset("v2", d=16, layers=2, heads=2, pointer=16, critic_d=32, critic_layers=1, critic_heads=2,
                      sector_d=16, eyes=False)


def net(cfg, seed):
    torch.manual_seed(seed)
    return policy.Actor(cfg).eval()


def env_of():
    nl = league.named_layout(12, 1, {"a/one.pt": 0.34, "b/two.pt": 0.33, "c/three.pt": 0.17, "self": 0.16})
    env = rollout.Battles(nl.layout, [("arena", "attack")], params=rollout.params_with_limit(30.0),
                          spread=randomise.NONE, past_slots=3, compile=False)
    env.set_past(net(TINY, 1), slot=0)
    env.set_past(net(OTHER, 2), slot=1)                  # (another size: its own memory width)
    env.set_past(net(TINY, 3), slot=2)
    env.fix_slots(nl.row_slot)
    return env, nl


def snapshot(env):
    return ({k: v.clone() for k, v in env.cstate.items()}, [None if h is None else h.clone() for h in env.past_h])


def restore(env, snap):
    cs, hs = snap
    env.cstate = {k: v.clone() for k, v in cs.items()}
    env.past_h = [None if h is None else h.clone() for h in hs]


def test_own_rows_give_the_all_rows_orders_memories_and_commitments():
    env, nl = env_of()
    learner, crit = net(TINY, 0), mcritic.Critic(TINY).eval()
    for _ in range(3):                                            # some decisions: memories and commitments set
        env.step(learner, crit)
    a, frame, _ = env.cur
    snap = snapshot(env)
    env.past_own = False
    o_all, h_all = env._act_past(a, frame, greedy=True)
    cs_all = {k: v.clone() for k, v in env.cstate.items()}
    restore(env, snap)
    env.past_own = True
    o_own, h_own = env._act_past(a, frame, greedy=True)
    assert env._slot_idx is not None and env._slot_idx[1] is not None      # (the own-rows way ran)
    for f in O.FIELDS:
        assert torch.equal(getattr(o_all, f), getattr(o_own, f)), f
    assert all(torch.equal(cs_all[k], env.cstate[k]) for k in cs_all)
    slot = env.row_slot[env.rows_past % env.B]
    for k in (0, 1, 2):
        mine = slot == k
        assert int(mine.sum()) > 0
        assert torch.allclose(h_all[k][mine], h_own[k][mine], atol=1e-6)


def test_battles_go_on_with_own_rows_and_without_fixed_slots_all_rows_stay():
    env, _ = env_of()
    learner, crit = net(TINY, 0), mcritic.Critic(TINY).eval()
    for _ in range(4):
        env.step(learner, crit)
    assert all(h is None or h.shape[0] == len(env.rows_past) for h in env.past_h)
    env.row_slot = None                                          # (no fixed slots: the all-rows way)
    assert env._slot_rows([0, 1, 2]) is None

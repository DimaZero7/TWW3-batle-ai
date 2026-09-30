"""tools.nn.model network (PyTorch): masks, unit order, adapters, orders out (levels 2 and 3).

torch is not in the project's .venv: these tests are skipped there and run in the training
container (snake-ai-trainer; its image has no pytest, so put a pure-Python pytest on PYTHONPATH).
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn import gamedata  # noqa: E402
from tools.nn.model import config, critic, decide, factions, heads, lora, policy, sources  # noqa: E402
from tools.nn.model import observation as ob  # noqa: E402
from tools.nn.sim import orders as sim_orders  # noqa: E402

CFG = config.preset("small", d=64, layers=2, heads=4, critic_d=64, critic_layers=2, critic_heads=4)


def setup_obs(seed=0, own=5, enemy=6, batch=2, **state_changes):
    setup, state = sources.synthetic(batch=batch, own=own, enemy=enemy, seed=seed)
    state.update(state_changes)
    obs, _ = ob.observe(state, setup, 1)
    return setup, state, obs


def model(cfg=CFG, seed=0):
    torch.manual_seed(seed)
    return policy.Actor(cfg).eval()


def test_the_observation_is_the_same_on_numpy_and_torch():
    setup, state, obs = setup_obs()
    tstate = {k: torch.as_tensor(v) for k, v in state.items()}
    tobs, _ = ob.observe(tstate, setup, 1)
    for k in ("tokens", "ctx", "pos"):
        assert np.allclose(getattr(obs, k), getattr(tobs, k).numpy(), atol=1e-5), k
    for k in ("own", "attend", "ctrl", "target_ok"):
        assert np.array_equal(getattr(obs, k), getattr(tobs, k).numpy()), k


def test_only_visible_living_enemies_can_be_targets_and_only_own_units_act():
    hidden = np.zeros((2, 11), bool)
    hidden[:, -2:] = True
    setup, state, obs = setup_obs(vis=~hidden)
    logits, _ = model()(policy.to_torch(obs))
    t = logits["target"]
    assert torch.all(t[:, :, -2:] <= heads.NEG / 2)                      # invisible
    assert torch.all(t[:, :, :5] <= heads.NEG / 2)                       # own units are not targets
    orders, *_ = decide.act(model(), obs, setup)
    enemy = torch.as_tensor(setup.side == 2)
    assert torch.all(orders.kind[enemy] == heads.HOLD) and torch.all(orders.target[enemy] == -1)
    attack = orders.kind == heads.ATTACK
    assert torch.all(torch.as_tensor(obs.target_ok).gather(1, orders.target.clamp(min=0))[attack])


def test_permuting_units_permutes_the_outputs():
    setup, state, obs = setup_obs()
    perm = np.random.default_rng(5).permutation(11)
    psetup = ob.Setup(keys=[[k[i] for i in perm] for k in setup.keys], side=setup.side[:, perm],
                      bounds=setup.bounds, factions=setup.factions, attacker=setup.attacker)
    pobs, _ = ob.observe({k: (v[:, perm] if np.ndim(v) == 2 else v) for k, v in state.items()}, psetup, 1)
    net = model()
    a, ha = net(policy.to_torch(obs))
    b, hb = net(policy.to_torch(pobs))
    p = torch.as_tensor(perm)
    for k in ("kind", "point", "run"):
        assert torch.allclose(a[k][:, p], b[k], atol=1e-4), k
    ok = a["target"] > heads.NEG / 2
    assert torch.allclose(a["target"][:, p][:, :, p][ok[:, p][:, :, p]], b["target"][ok[:, p][:, :, p]], atol=1e-4)
    assert torch.allclose(ha[:, 1:][:, p], hb[:, 1:], atol=1e-4)


def test_memory_carries_between_decisions():
    setup, state, obs = setup_obs()
    net = model()
    o = policy.to_torch(obs)
    a, h = net(o)
    b, _ = net(o, h)
    assert h.abs().sum() > 0 and not torch.allclose(a["kind"], b["kind"])


def test_adapters_start_as_no_change_and_train_alone():
    cfg = config.preset("small", d=64, layers=2, heads=4, lora_rank=4)
    setup, state, obs = setup_obs()
    net = model(cfg)
    o = policy.to_torch(obs)
    with_adapter, _ = net(o)
    without, _ = net(o, use_adapter=False)
    assert torch.allclose(with_adapter["kind"], without["kind"])
    lora.freeze_base(net)
    trainable = [p for p in net.parameters() if p.requires_grad]
    assert trainable and all(any(p is q for q in lora.adapter_parameters(net)) for p in trainable)
    per_adapter = sum(p.numel() for p in trainable) / factions.ADAPTERS     # one pair "faction + role"
    assert per_adapter < 0.05 * policy.parameters(net)


def test_log_prob_of_a_sampled_action_is_finite_and_zero_for_units_without_orders():
    setup, state, obs = setup_obs()
    o = policy.to_torch(obs)
    logits, _ = model()(o)
    a = heads.sample(logits)
    lp, ent = heads.log_prob(logits, a, o["ctrl"])
    assert torch.isfinite(lp).all() and torch.all(lp[~o["ctrl"]] == 0) and torch.all(lp[o["ctrl"]] <= 0)
    assert torch.all(ent[o["ctrl"]] > 0)


def test_move_points_are_inside_the_map_and_hold_stays_in_place():
    setup, state, obs = setup_obs(batch=4)
    orders, *_ = decide.act(model(), obs, setup, temperature=5.0)
    b = setup.bounds
    x, z = orders.x.numpy(), orders.z.numpy()
    assert np.all((x >= b[:, :1] - 1e-3) & (x <= b[:, 1:2] + 1e-3) & (z >= b[:, 2:3] - 1e-3) & (z <= b[:, 3:4] + 1e-3))
    hold = orders.kind.numpy() == heads.HOLD
    alive = state["men"] > 0
    assert np.allclose(x[hold & alive], state["x"][hold & alive], atol=1e-2)


def test_the_critic_sees_the_full_state():
    setup, state = sources.synthetic(batch=2, own=5, enemy=6)
    full, _ = ob.observe(state, setup, 1, full=True)
    torch.manual_seed(0)
    net = critic.Critic(CFG).eval()
    v = net(policy.to_torch(full))
    changed = dict(state, mp=np.where(setup.side == 2, -state["mp"], state["mp"]))
    full2, _ = ob.observe(changed, setup, 1, full=True)
    assert v.shape == (2,) and not torch.allclose(v, net(policy.to_torch(full2)))


def test_sizes_of_the_presets():
    small = policy.parameters(policy.Actor(config.SMALL))
    target = policy.parameters(policy.Actor(config.TARGET))
    assert small < 2e6 and 5e6 <= target <= 20e6


RUNS = gamedata.runs(arena="whole_emp_v_skv")


@pytest.mark.skipif(not RUNS, reason="no recorded Empire vs Skaven battles (build/ is not in Git)")
def test_recorded_battles_to_orders():
    rec = sources.batch([gamedata.load(r) for r in RUNS[:2]])
    net = model()
    for side in (1, 2):
        memory, h = None, None
        for ti in range(0, 40, 4):
            obs, memory = ob.observe(rec.state(ti), rec.setup, side, memory)
            orders, h, _, _ = decide.act(net, obs, rec.setup, h)
        own = torch.as_tensor(rec.setup.side == side)
        sim_orders.check(orders, rec.setup.side.shape[1])
        assert torch.all(orders.kind[~own] == heads.HOLD)


def sim_state(B=2, own=5, enemy=6):
    """A simulator State (tools/nn/sim/state.py) filled with a made-up battle."""
    from tools.nn.sim import state as sim_state_module
    setup, st = sources.synthetic(batch=B, own=own, enemy=enemy)
    s = sim_state_module.empty(B, own + enemy)
    for k, v in st.items():
        if k in s.u:
            s.u[k] = torch.as_tensor(v).to(s.u[k].dtype)
    s.u["side"] = torch.as_tensor(setup.side)
    s.u["fat"] = torch.randint(0, 6, (B, own + enemy)).float()
    s.t = torch.as_tensor(st["t"]).float()
    s.keys = setup.keys
    return s


def test_simulator_state_to_orders():
    s = sim_state()
    setup, st = sources.from_sim(s)
    net = model()
    memory, h = None, None
    for step in range(3):
        st = dict(st, t=st["t"] + step)
        obs, memory = ob.observe(st, setup, 2, memory)
        assert obs.tokens.shape == (2, 11, ob.TOKEN) and torch.isfinite(obs.tokens).all()
        assert torch.all(obs.tokens[..., ob.INDEX["fatigue_known"]][torch.as_tensor(setup.side == 2)] == 1)
        orders, h, _, _ = decide.act(net, obs, setup, h)
    sim_orders.check(orders, s.N)
    other = torch.as_tensor(setup.side == 1)
    assert torch.all(orders.kind[other] == heads.HOLD)

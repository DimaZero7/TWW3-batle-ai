"""tools.nn.model network (PyTorch): masks, unit order, old checkpoints, orders out (levels 2 and 3).

torch is not in the project's .venv: these tests are skipped there and run in the training
container (snake-ai-trainer; its image has no pytest, so put a pure-Python pytest on PYTHONPATH).
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.config import ROOT  # noqa: E402
from tools.nn import gamedata  # noqa: E402
from tools.nn.model import config, critic, decide, heads, policy, sources  # noqa: E402
from tools.nn.model import effects as mfx  # noqa: E402
from tools.nn.model import observation as ob  # noqa: E402
from tools.nn.sim import orders as sim_orders  # noqa: E402

CFG = config.preset("small", d=64, layers=2, heads=4, critic_d=64, critic_layers=2, critic_heads=4)


def setup_obs(seed=0, own=5, enemy=6, batch=2, **state_changes):
    setup, state = sources.synthetic(batch=batch, own=own, enemy=enemy, seed=seed)
    state.update(state_changes)
    obs, _ = ob.observe(state, setup, 1)
    return setup, state, obs


def critic_state(data):
    """A checkpoint's critic without the per-unit value head of its time (Critic.load drops it too)."""
    return {k: v for k, v in data["critic"].items() if not k.startswith("unit_value.")}


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


def test_a_checkpoint_of_the_lora_wrapper_loads_and_acts_the_same():
    # until 03.10 the attention blocks' layers were LoRA wrappers (adapters never on): qkv.base.weight
    setup, state, obs = setup_obs()
    net = model()
    old = {}
    for k, v in net.state_dict().items():
        for name in ("qkv", "out", "f1", "f2"):
            k = k.replace(f".{name}.weight", f".{name}.base.weight").replace(f".{name}.bias", f".{name}.base.bias")
        old[k] = v
    assert any(".qkv.base.weight" in k for k in old)
    again = policy.Actor(CFG).eval()
    again.load_state_dict(old)
    o = policy.to_torch(obs)
    with torch.no_grad():
        assert torch.equal(net(o)[0]["kind"], again(o)[0]["kind"])
    c = critic.Critic(CFG)
    cold = {k.replace(".qkv.weight", ".qkv.base.weight"): v for k, v in c.state_dict().items()}
    assert all(torch.equal(v, c.state_dict()[k]) for k, v in critic.Critic(CFG).load(cold).state_dict().items())


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


# --- abilities: the pointer over a unit's ability slots (tools/nn/model/heads.py) ---

LORDS = ["wh_main_emp_cha_general_0", "wh2_main_skv_cha_warlord_0", "wh_main_emp_inf_spearmen_0"]


def ability_obs(batch=2, seed=7, **timers):
    setup, state = sources.synthetic(batch=batch, own=4, enemy=4, seed=seed, keys=LORDS)
    N = 8
    state.update(men=np.maximum(state["men"], 1.0), ms=np.full((batch, N), 2.0), r=np.zeros((batch, N), bool),
                 s=np.zeros((batch, N), bool))
    for k in range(3):
        for t in ("on", "cd"):
            state[f"ab{k}_{t}"] = timers.get(f"ab{k}_{t}", np.zeros((batch, N)))
    obs, _ = ob.observe(state, setup, 1)
    return setup, state, obs


def test_abilities_are_the_same_on_numpy_and_torch():
    setup, state, obs = ability_obs()
    tobs, _ = ob.observe({k: torch.as_tensor(v) for k, v in state.items()}, setup, 1)
    assert np.allclose(obs.abil, tobs.abil.numpy(), atol=1e-6)
    assert np.array_equal(obs.abil_ok, tobs.abil_ok.numpy()) and obs.abil_ok.any()


def test_only_ready_abilities_of_own_units_can_be_chosen():
    cd = np.zeros((2, 8))
    cd[:, :] = 30.0                                     # every slot 0 recharging
    setup, state, obs = ability_obs(ab0_cd=cd)
    o = policy.to_torch(obs)
    logits, _ = model()(o)
    a = logits["ability"]
    assert a.shape == (2, 8, 4) and torch.isfinite(a[..., 0]).all()
    allowed = o["abil_ok"] & o["ctrl"][..., None]
    assert torch.all(a[..., 1:][~allowed] <= heads.NEG / 2) and torch.all(a[..., 1:][allowed] > heads.NEG / 2)
    assert not allowed[..., 0].any() and allowed.any()
    for _ in range(5):
        orders, _, _, act = decide.act(model(), obs, setup, temperature=3.0)
        chosen = orders.ability
        ok = (chosen == -1) | allowed.gather(2, chosen.clamp(min=0)[..., None])[..., 0]
        assert torch.all(ok) and torch.all(chosen[torch.as_tensor(setup.side == 2)] == -1)
        sim_orders.check(orders, 8)


def test_permuting_ability_slots_permutes_the_choice():
    setup, state, obs = ability_obs()
    o = policy.to_torch(obs)
    perm = torch.tensor([2, 0, 1])
    p = dict(o, abil=o["abil"][:, :, perm], abil_ok=o["abil_ok"][:, :, perm])
    net = model()
    a, _ = net(o)
    b, _ = net(p)
    assert torch.allclose(a["ability"][..., 1:][..., perm], b["ability"][..., 1:], atol=1e-5)
    assert torch.allclose(a["ability"][..., 0], b["ability"][..., 0], atol=1e-5)
    assert torch.allclose(a["kind"], b["kind"], atol=1e-5)


def test_an_unseen_ability_still_gives_a_valid_choice():
    from tools.nn.model import abilities as mab
    setup, state, obs = ability_obs()
    o = policy.to_torch(obs)
    torch.manual_seed(3)
    owned = o["abil"][..., mab.INDEX["owned"]] > 0.5
    new = o["abil"].clone()
    new[..., mab.DYNAMIC:] = torch.where(owned[..., None], torch.rand_like(new[..., mab.DYNAMIC:]) * 2 - 1,
                                         new[..., mab.DYNAMIC:])
    logits, _ = model()(dict(o, abil=new))
    assert torch.isfinite(logits["ability"][..., 0]).all() and torch.isfinite(logits["kind"]).all()
    a = heads.sample(logits, abilities=True)
    allowed = torch.cat([torch.ones_like(o["abil_ok"][..., :1]), o["abil_ok"] & o["ctrl"][..., None]], -1)
    assert torch.all(allowed.gather(2, (a.ability + 1)[..., None]))
    lp, _ = heads.log_prob(logits, a, o["ctrl"])
    assert torch.isfinite(lp).all()


def test_the_ability_counts_in_log_prob_only_when_chosen():
    setup, state, obs = ability_obs()
    o = policy.to_torch(obs)
    logits, _ = model()(o)
    torch.manual_seed(0)
    a = heads.sample(logits)
    assert a.ability is None
    base = {k: v for k, v in logits.items() if k != "ability"}
    lp, _ = heads.log_prob(logits, a, o["ctrl"])
    assert torch.allclose(lp, heads.log_prob(base, a, o["ctrl"])[0])
    a.ability = torch.full_like(a.kind, -1)
    lp2, ent2 = heads.log_prob(logits, a, o["ctrl"])
    has = (o["abil_ok"] & o["ctrl"][..., None]).any(-1)
    assert torch.all(lp2[has] < lp[has]) and torch.allclose(lp2[~has], lp[~has])


def test_an_actor_saved_before_abilities_loads_and_its_input_starts_silent():
    setup, state, obs = ability_obs()
    o = policy.to_torch(obs)
    net = model()
    old = {k: v for k, v in net.state_dict().items() if not k.startswith(policy.ABILITY_PARAMS)}
    fresh = model(seed=9)
    fresh.load_state_dict(old)
    a, _ = net(o)
    b, _ = fresh(o)
    assert torch.allclose(a["kind"], b["kind"], atol=1e-6) and torch.allclose(a["target"], b["target"], atol=1e-6)
    without, _ = net({k: v for k, v in o.items() if k not in ("abil", "abil_ok")})
    assert torch.allclose(a["kind"], without["kind"], atol=1e-6) and "ability" not in without
    with pytest.raises(RuntimeError):
        fresh.load_state_dict({k: v for k, v in old.items() if not k.startswith("heads.kind")})


# --- the damage timers and the context without the time limit (observation.TIMERS) ---

def timer_states(batch=2):
    """Three decisions of a made-up battle: side 1 strikes an enemy at t 31, side 2 strikes back at t 32
    (every unit standing: side 1's blow is the attacker's progress too)."""
    setup, state = sources.synthetic(batch=batch, own=3, enemy=3, seed=5, keys=LORDS)
    N = 6
    state.update(men=np.maximum(state["men"], 1.0), ms=np.full((batch, N), 2.0), r=np.zeros((batch, N), bool),
                 s=np.zeros((batch, N), bool))
    hp = np.full((batch, N), 0.9)
    hit1, hit2 = hp.copy(), hp.copy()
    hit1[:, 4] = 0.7
    hit2[:, 4], hit2[:, 1] = 0.7, 0.5
    for k in range(3):
        for t in ("on", "cd"):
            state[f"ab{k}_{t}"] = np.zeros((batch, N))
    return setup, [dict(state, t=np.full(batch, 30.0 + i), hp=h) for i, h in enumerate((hp, hit1, hit2))]


def test_the_damage_timers_are_the_same_on_numpy_and_torch():
    setup, states = timer_states()
    mn = mt = None
    for st in states:
        a, mn = ob.observe(st, setup, 1, mn)
        b, mt = ob.observe({k: torch.as_tensor(v) for k, v in st.items()}, setup, 1, mt)
        assert np.allclose(a.ctx, b.ctx.numpy(), atol=1e-6)
    assert a.ctx[0, [ob.CTX[n] for n in ob.TIMERS[1:]]].tolist() == pytest.approx([1, 1 / ob.SINCE, 1, 0])
    assert a.ctx[0, ob.CTX["progress_rate"]] > 0                       # the attacker's progress: on both too


def old_context(ctx, t=None):
    """The context as networks before the damage timers saw it: a t / 3600 column (t None: 0), no TIMERS
    (and no PROGRESS)."""
    time = torch.zeros_like(ctx[:, :1]) if t is None else (t / 3600.0).clamp(0, 1)[:, None].to(ctx)
    return torch.cat([ctx[:, :ob.OLD_TIME], time, ctx[:, ob.OLD_TIME:ob.CONTEXT_BASE], ctx[:, ob.CONTEXT:]], 1)


def old_networks(cfg):
    """An actor and a critic with the older context input (a time column, no TIMERS, no PROGRESS)."""
    actor, crit = policy.Actor(cfg), critic.Critic(cfg)
    n = len(ob.TIMERS) + len(ob.PROGRESS) - 1
    actor.encoder.ctx[0] = torch.nn.Linear(ob.CONTEXT - n, cfg.d)
    crit.encoder.ctx[0] = torch.nn.Linear(ob.CONTEXT_FULL - n, cfg.critic_d)
    return actor.eval(), crit.eval()


def same_outputs(old, new, old_crit, new_crit, setup, states, to_old=old_context, atol=1e-5):
    """Runs both through the decisions (memory carried); asserts equal logits, greedy actions, values
    (to_old: the context as the old networks saw it)."""
    m = mc = h_old = h_new = None
    for st in states:
        obs, m = ob.observe(st, setup, 1, m)
        cobs, mc = ob.observe(st, setup, 1, mc, full=True)
        o = policy.to_torch(obs)
        with torch.no_grad():
            lo, h_old = old(dict(o, ctx=to_old(o["ctx"])), h_old)
            ln, h_new = new(o, h_new)
            c = policy.to_torch(cobs)
            v_old = old_crit({k: c[k] for k in ("tokens", "own", "attend", "pos")} | {"ctx": to_old(c["ctx"])})
            v_new = new_crit(c)
        assert set(lo) == set(ln)
        for k in lo:
            assert torch.allclose(lo[k], ln[k], atol=atol), k
        assert torch.allclose(h_old, h_new, atol=atol) and torch.allclose(v_old, v_new, atol=atol)
        a, b = heads.sample(lo, greedy=True, abilities=True), heads.sample(ln, greedy=True, abilities=True)
        assert torch.equal(a.kind, b.kind) and torch.equal(a.target, b.target) and torch.equal(a.point, b.point)
    assert obs.ctx[:, ob.CTX["dealt_any"]].sum() > 0          # the timers were not all zero


def test_networks_saved_before_the_damage_timers_load_and_act_the_same():
    setup, states = timer_states()
    torch.manual_seed(3)
    old, old_crit = old_networks(CFG)
    new, new_crit = model(seed=8), critic.Critic(CFG).eval()
    new.load_state_dict(old.state_dict())
    new_crit.load(old_crit.state_dict())
    assert torch.all(new.encoder.ctx[0].weight[:, ob.CONTEXT_BASE:] == 0)
    same_outputs(old, new, old_crit, new_crit, setup, states)


REAL = [p for p in (ROOT / "build/nn-train/test5/t0_gold30/m20.pt",
                    ROOT / "build/nn-train/runs/long_ai/best.pt") if p.exists()]


@pytest.mark.skipif(not REAL, reason="no trained checkpoints (build/ is not in Git)")
@pytest.mark.parametrize("path", REAL, ids=[p.parent.name + "/" + p.name for p in REAL])
def test_a_trained_checkpoint_acts_as_before_the_damage_timers(path):
    from tools.nn.train import checkpoint
    data = checkpoint.read(path)
    cfg = checkpoint.config_of(data)
    new, new_crit = checkpoint.load_policy(path), checkpoint.load_critic(path).eval()
    old, old_crit = old_networks(cfg)
    # parts a checkpoint older than abilities lacks start fresh: the same in both
    fresh = {k: v for k, v in new.state_dict().items() if k.startswith(policy.ABILITY_PARAMS)}
    torch.nn.Module.load_state_dict(old, {**fresh, **data["actor"]})
    torch.nn.Module.load_state_dict(old_crit, critic_state(data))
    setup, states = timer_states()
    same_outputs(old, new, old_crit, new_crit, setup, states)


# --- the attacker's progress (observation.PROGRESS), appended after the damage timers ---

def no_progress(ctx):
    """The context as networks before PROGRESS saw it (the critic's enemy character stays after it)."""
    at = ob.CONTEXT - len(ob.PROGRESS)
    return torch.cat([ctx[:, :at], ctx[:, ob.CONTEXT:]], 1)


def pre_progress_networks(cfg):
    actor, crit = policy.Actor(cfg), critic.Critic(cfg)
    actor.encoder.ctx[0] = torch.nn.Linear(ob.CONTEXT - len(ob.PROGRESS), cfg.d)
    crit.encoder.ctx[0] = torch.nn.Linear(ob.CONTEXT_FULL - len(ob.PROGRESS), cfg.critic_d)
    return actor.eval(), crit.eval()


def progress_cols(net):
    at = ob.CONTEXT - len(ob.PROGRESS)
    return net.encoder.ctx[0].weight[:, at:ob.CONTEXT]


def test_networks_saved_before_the_progress_inputs_load_and_act_the_same():
    setup, states = timer_states()
    torch.manual_seed(5)
    old, old_crit = pre_progress_networks(CFG)
    new, new_crit = model(seed=9), critic.Critic(CFG).eval()
    new.load_state_dict(old.state_dict())
    new_crit.load(old_crit.state_dict())
    assert torch.all(progress_cols(new) == 0) and torch.all(progress_cols(new_crit) == 0)
    # the critic's enemy character keeps its weights, after the new columns
    assert torch.equal(new_crit.encoder.ctx[0].weight[:, ob.CONTEXT:],
                       old_crit.encoder.ctx[0].weight[:, ob.CONTEXT - len(ob.PROGRESS):])
    same_outputs(old, new, old_crit, new_crit, setup, states, to_old=no_progress)
    mem = None
    for st in states:
        obs, mem = ob.observe(st, setup, 1, mem)
    assert obs.ctx[:, ob.CTX["progress_rate"]].sum() > 0 and obs.ctx[:, ob.CTX["progress_any"]].sum() > 0


PRE_PROGRESS = [p for p in (ROOT / "build/nn-train/test5/r1_idleprog/m15.pt",) if p.exists()]


@pytest.mark.skipif(not PRE_PROGRESS, reason="no checkpoint saved before PROGRESS (build/ is not in Git)")
@pytest.mark.parametrize("path", PRE_PROGRESS, ids=[p.parent.name + "/" + p.name for p in PRE_PROGRESS])
def test_a_checkpoint_saved_before_the_progress_inputs_acts_as_before(path):
    from tools.nn.train import checkpoint
    data = checkpoint.read(path)
    cfg = checkpoint.config_of(data)
    new, new_crit = checkpoint.load_policy(path), checkpoint.load_critic(path).eval()
    assert torch.all(progress_cols(new) == 0) and torch.all(progress_cols(new_crit) == 0)
    old, old_crit = pre_progress_networks(cfg)
    torch.nn.Module.load_state_dict(old, data["actor"])
    torch.nn.Module.load_state_dict(old_crit, critic_state(data))
    setup, states = timer_states()
    same_outputs(old, new, old_crit, new_crit, setup, states, to_old=no_progress, atol=1e-4)


# --- the second Empire wave's inputs (02.10.2026): passport features and ability conditions appended ---

WAVE2 = ["wh_dlc04_emp_inf_flagellants_0", "wh_main_emp_inf_greatswords", "wh_dlc04_emp_inf_free_company_militia_0",
         "wh_main_emp_cha_general_0", "wh2_main_skv_cha_warlord_0", "wh2_main_skv_inf_skavenslave_slingers_0"]
NEW_PASSPORT = 2 + 3          # attributes mounted_fire_move, guerrilla_deploy; direct, spread, muzzle_velocity
NEW_ABILITY = 8               # abilities.WHEN
NEW_EFFECTS = mfx.SIZE        # the innate effects' columns (appended after the second wave)
NEW_VOLLEY = len(ob.VOLLEY)   # the volley input, appended after the effects (03.10.2026)
NEW_ARC = len(ob.ARC)         # the fire arc, appended after the volley input (07.10.2026)
NEW_ORDER = len(ob.ORDER)     # the order in force, appended after the fire arc (07.10.2026)
T0 = ob.TOKEN - NEW_ORDER - NEW_VOLLEY - NEW_ARC    # the token before the volley input
T1 = ob.TOKEN - NEW_ORDER - NEW_ARC       # the token before the fire arc


def test_the_second_wave_is_seen_in_the_passport_and_the_ability_slots():
    from tools.nn.model import abilities as mab
    from tools.nn.model import passport
    assert len(passport.FLIGHT) + 2 == NEW_PASSPORT and len(mab.WHEN) == NEW_ABILITY
    f = passport.table(WAVE2[:3] + ["wh2_dlc13_emp_inf_archers_0"])
    at = len(passport.features(passport.load()[WAVE2[0]])) - len(passport.FLIGHT) - len(passport.ATTRIBUTES)
    unbreakable = at + passport.ATTRIBUTES.index("unbreakable")
    direct = len(f[0]) - 3
    assert f[0, unbreakable] == 1 and f[1, unbreakable] == 0                  # flagellants never rout
    assert f[2, direct] == 1 and f[3, direct] == 0                            # pistols flat, arrows arc
    assert f[2, at + passport.ATTRIBUTES.index("mounted_fire_move")] == 1
    feat, owned, usable = mab.slots([WAVE2[0]])
    names = mab.STATIC_NAMES
    assert owned[0].sum() == 2 and not usable[0].any()                        # the game fires them, not orders
    penitent = feat[0, 0]                                                     # actives first (sim slot order)
    assert penitent[names.index("auto")] == 1 and penitent[names.index("when_losing_melee")] == 1
    assert penitent[names.index("off_out_of_melee")] == 1
    assert feat[0, 1, names.index("off_morale_below_half")] == 1             # frenzy


def narrow_networks(cfg):
    """An actor and critic as saved before the second wave: NEW_PASSPORT fewer token inputs, NEW_ABILITY
    fewer ability inputs (all appended at the ends)."""
    from tools.nn.model import abilities as mab
    actor, crit = policy.Actor(cfg), critic.Critic(cfg)
    actor.encoder.unit[0] = torch.nn.Linear(T0 - NEW_EFFECTS - NEW_PASSPORT, cfg.d)
    actor.abilities.net[0] = torch.nn.Linear(mab.SIZE - NEW_ABILITY, cfg.d)
    actor.heads.ability_k[0] = torch.nn.Linear(mab.SIZE - NEW_ABILITY, cfg.d)
    crit.encoder.unit[0] = torch.nn.Linear(T0 - NEW_EFFECTS - NEW_PASSPORT, cfg.critic_d)
    return actor.eval(), crit.eval()


def narrow(o):
    """The observation as the networks before the second wave saw it."""
    out = dict(o, tokens=o["tokens"][..., :T0 - NEW_EFFECTS - NEW_PASSPORT])
    if "abil" in o:
        out["abil"] = o["abil"][..., :o["abil"].shape[-1] - NEW_ABILITY]
    return out


def wave2_obs():
    setup, state = sources.synthetic(batch=2, own=4, enemy=4, seed=11, keys=WAVE2)
    N = 8
    state.update(men=np.maximum(state["men"], 1.0), ms=np.full((2, N), 2.0), r=np.zeros((2, N), bool),
                 s=np.zeros((2, N), bool))
    for k in range(3):
        for t in ("on", "cd"):
            state[f"ab{k}_{t}"] = np.zeros((2, N))
    obs, _ = ob.observe(state, setup, 1)
    cobs, _ = ob.observe(state, setup, 1, full=True)
    return obs, cobs


def test_networks_saved_before_the_second_wave_load_and_act_the_same_and_train():
    obs, cobs = wave2_obs()
    torch.manual_seed(4)
    old, old_crit = narrow_networks(CFG)
    new, new_crit = model(seed=12), critic.Critic(CFG).eval()
    new.load_state_dict(old.state_dict())
    new_crit.load(old_crit.state_dict())
    o, c = policy.to_torch(obs), policy.to_torch(cobs)
    assert o["tokens"][..., T0 - NEW_EFFECTS - NEW_PASSPORT:].abs().sum() > 0    # the new inputs are not all zero
    with torch.no_grad():
        lo, _ = old(narrow(o))
        ln, _ = new(o)
        keys = ("tokens", "own", "attend", "pos", "ctx")
        vo, vn = old_crit(narrow({k: c[k] for k in keys})), new_crit({k: c[k] for k in keys})
    for k in lo:
        assert torch.allclose(lo[k], ln[k], atol=1e-5), k
    assert torch.allclose(vo, vn, atol=1e-5)
    # forward and backward with the new inputs: the new columns get a gradient (they can learn)
    # (the ability encoder's last layer starts at zero: its first layer learns once that one has moved;
    # the flagellants' abilities are never orderable, so the head's key gets nothing from them)
    new.train()
    with torch.no_grad():
        new.abilities.net[2].weight.normal_(std=0.01)
    out, _ = new(o)
    loss = sum(v.masked_fill(v < -1e8, 0).sum() for v in out.values())
    loss.backward()
    assert new.encoder.unit[0].weight.grad[:, T0 - NEW_EFFECTS - NEW_PASSPORT:T0 - NEW_EFFECTS].abs().sum() > 0
    assert new.abilities.net[0].weight.grad[:, -NEW_ABILITY:].abs().sum() > 0


CHAIN = ROOT / "build/nn-train/test5/it4/m10.pt"


@pytest.mark.skipif(not CHAIN.exists(), reason="no chain checkpoint (build/ is not in Git)")
def test_the_chain_checkpoint_loads_through_the_conversion_and_acts_as_before():
    from tools.nn.train import checkpoint
    data = checkpoint.read(CHAIN)
    cfg = checkpoint.config_of(data)
    new = checkpoint.load_policy(CHAIN)
    new_crit = checkpoint.load_critic(CHAIN)
    old, old_crit = narrow_networks(cfg)
    fresh = {k: v for k, v in new.state_dict().items() if k.startswith(policy.ABILITY_PARAMS)
             and k not in data["actor"]}
    torch.nn.Module.load_state_dict(old, {**fresh, **data["actor"]})
    if new_crit is not None:
        fresh = {k: v for k, v in new_crit.state_dict().items() if k not in data["critic"]}
        torch.nn.Module.load_state_dict(old_crit, {**fresh, **critic_state(data)})
    obs, cobs = wave2_obs()
    o, c = policy.to_torch(obs), policy.to_torch(cobs)
    with torch.no_grad():
        lo, _ = old.eval()(narrow(o))
        ln, _ = new(o)
    for k in lo:
        assert torch.allclose(lo[k], ln[k], atol=1e-4), k
    if new_crit is not None:
        keys = ("tokens", "own", "attend", "pos", "ctx")
        with torch.no_grad():
            assert torch.allclose(old_crit.eval()(narrow({k: c[k] for k in keys})),
                                  new_crit.eval()({k: c[k] for k in keys}), atol=1e-4)


# --- innate effects (tools/nn/model/effects.py): per effect (owned, on) appended at the token's end ---

FX_UNITS = ["wh_dlc04_emp_inf_flagellants_0", "wh2_main_skv_inf_clanrats_1", "wh2_main_skv_inf_skavenslaves_0",
            "wh_main_emp_cha_general_0", "wh2_main_skv_inf_night_runners_1", "wh2_main_skv_cha_warlord_0"]


def fx_obs(batch=2, seed=13, **changes):
    setup, state = sources.synthetic(batch=batch, own=3, enemy=3, seed=seed, keys=FX_UNITS)
    N = 6
    state.update(men=np.maximum(state["men"], 1.0))
    for k in range(3):
        for t in ("on", "cd"):
            state[f"ab{k}_{t}"] = np.zeros((batch, N))
    state.update(changes)
    obs, _ = ob.observe(state, setup, 1)
    cobs, _ = ob.observe(state, setup, 1, full=True)
    return setup, state, obs, cobs


def fx_cols(tokens):
    return tokens[..., T0 - NEW_EFFECTS:T0]


def test_the_effects_are_owned_by_the_catalogue_and_the_same_on_numpy_and_torch():
    setup, state, obs, _ = fx_obs()
    keys = mfx.keys()
    owned = fx_cols(obs.tokens)[..., 0::2]
    for b in range(2):
        for n, key in enumerate(setup.keys[b]):
            want = {keys.index(e) for e in mfx.load()["units"][key]}
            assert set(np.nonzero(owned[b, n])[0]) == want, key
    t, _ = ob.observe({k: torch.as_tensor(v) for k, v in state.items()}, setup, 1)
    assert np.allclose(obs.tokens, t.tokens.numpy(), atol=1e-6)
    assert ob.NAMES[T0 - NEW_EFFECTS:T0] == mfx.NAMES


def test_on_follows_the_conditions_and_enemies_show_it_only_while_seen():
    sin = mfx.keys().index("wh2_main_unit_passive_strength_in_numbers")
    N = 6
    hp = np.full((2, N), 0.9)
    setup, state, obs, cobs = fx_obs(hp=hp, vis=np.ones((2, N), bool))
    on = fx_cols(obs.tokens)[..., 1::2]
    owners = fx_cols(obs.tokens)[..., 0::2][..., sin] > 0
    assert owners.any() and np.all(on[..., sin][owners] == 1)
    setup, state, obs, cobs = fx_obs(hp=np.full((2, N), 0.3), vis=np.ones((2, N), bool))
    assert np.all(fx_cols(obs.tokens)[..., 1::2][..., sin] == 0)              # below half: off
    hidden = np.ones((2, N), bool)
    hidden[:, 3:] = False
    setup, state, obs, cobs = fx_obs(hp=hp, vis=hidden)
    assert np.all(fx_cols(obs.tokens)[:, 3:, 1::2] == 0)                       # unseen enemies: not shown
    assert np.all(fx_cols(obs.tokens)[:, 3:, 0::2] == fx_cols(cobs.tokens)[:, 3:, 0::2])  # owned: known
    assert fx_cols(cobs.tokens)[:, 3:, 1::2].sum() > 0                          # the critic sees all


def test_the_simulator_and_the_observed_fields_agree_on_what_is_on():
    from tools.nn.sim import battle, replay, scenario
    from tools.nn.sim.params import load
    P = load()

    def side(rows):
        return [{"key": k, "x": x, "z": 0.0, "b": b, "general": "cha" in k} for k, x, b in rows]
    a = {"attacker": 1, "sides": {1: {"faction": "wh_main_emp_empire", "units": side(
        [(FX_UNITS[3], -80, 90), (FX_UNITS[0], -25, 90)])}, 2: {"faction": "wh2_main_skv_skaven", "units": side(
            [(FX_UNITS[5], 80, 270), (FX_UNITS[1], 25, 270), (FX_UNITS[2], 30, 270)])}}}
    st = scenario.build([a], P)
    timed = [mfx.keys().index(k) for k, e in mfx.load()["effects"].items() if e.get("timed")]
    for _ in range(120):
        battle.step(st, replay.nearest_attack(st), P)
        su, ss = sources.from_sim(st)
        sim, _ = ob.observe(ss, su, 1, full=True)
        rec, _ = ob.observe({k: v for k, v in ss.items() if k != "fx_on"}, su, 1, full=True)
        s_on, r_on = fx_cols(sim.tokens)[..., 1::2], fx_cols(rec.tokens)[..., 1::2]
        keep = [i for i in range(len(mfx.keys())) if i not in timed]
        alive = (ss["men"] > 0)
        assert torch.equal(s_on[..., keep][alive], r_on[..., keep][alive])


def pre_effects_networks(cfg):
    """An actor and critic as saved before the innate effects: NEW_EFFECTS fewer token inputs."""
    actor, crit = policy.Actor(cfg), critic.Critic(cfg)
    actor.encoder.unit[0] = torch.nn.Linear(T0 - NEW_EFFECTS, cfg.d)
    crit.encoder.unit[0] = torch.nn.Linear(T0 - NEW_EFFECTS, cfg.critic_d)
    return actor.eval(), crit.eval()


def without_effects(o):
    return dict(o, tokens=o["tokens"][..., :T0 - NEW_EFFECTS])


def test_networks_saved_before_the_effects_load_act_the_same_and_learn_them():
    setup, state, obs, cobs = fx_obs()
    torch.manual_seed(5)
    old, old_crit = pre_effects_networks(CFG)
    new, new_crit = model(seed=14), critic.Critic(CFG).eval()
    new.load_state_dict(old.state_dict())
    new_crit.load(old_crit.state_dict())
    o, c = policy.to_torch(obs), policy.to_torch(cobs)
    assert fx_cols(o["tokens"]).abs().sum() > 0
    keys = ("tokens", "own", "attend", "pos", "ctx")
    with torch.no_grad():
        lo, _ = old(without_effects(o))
        ln, _ = new(o)
        zeroed, _ = new(dict(o, tokens=torch.cat([without_effects(o)["tokens"],
                                                  torch.zeros_like(o["tokens"][..., T0 - NEW_EFFECTS:])], -1)))
        vo, vn = old_crit(without_effects({k: c[k] for k in keys})), new_crit({k: c[k] for k in keys})
    for k in lo:
        assert torch.allclose(lo[k], ln[k], atol=1e-5) and torch.allclose(ln[k], zeroed[k], atol=1e-6), k
    assert torch.allclose(vo, vn, atol=1e-5)
    new.train()
    out, _ = new(o)
    sum(v.masked_fill(v < -1e8, 0).sum() for v in out.values()).backward()
    assert new.encoder.unit[0].weight.grad[:, T0 - NEW_EFFECTS:T0].abs().sum() > 0


CHAIN5 = ROOT / "build/nn-train/test5/it5/m20.pt"


@pytest.mark.skipif(not CHAIN5.exists(), reason="no chain checkpoint (build/ is not in Git)")
def test_the_it5_checkpoint_loads_and_acts_as_before_the_effects():
    from tools.nn.train import checkpoint
    data = checkpoint.read(CHAIN5)
    cfg = checkpoint.config_of(data)
    new = checkpoint.load_policy(CHAIN5)
    new_crit = checkpoint.load_critic(CHAIN5)
    old, old_crit = pre_effects_networks(cfg)
    torch.nn.Module.load_state_dict(old, data["actor"])
    setup, state, obs, cobs = fx_obs()
    o, c = policy.to_torch(obs), policy.to_torch(cobs)
    with torch.no_grad():
        lo, _ = old(without_effects(o))
        ln, _ = new(o)
    for k in lo:
        assert torch.allclose(lo[k], ln[k], atol=1e-4), k
    if new_crit is not None:
        torch.nn.Module.load_state_dict(old_crit, critic_state(data))
        keys = ("tokens", "own", "attend", "pos", "ctx")
        with torch.no_grad():
            assert torch.allclose(old_crit.eval()(without_effects({k: c[k] for k in keys})),
                                  new_crit.eval()({k: c[k] for k in keys}), atol=1e-4)


# --- the volley input (observation.VOLLEY, 03.10.2026): one own-only column appended after the effects ---

VOLLEY_KEYS = ["wh2_main_skv_inf_night_runners_1", "wh2_dlc13_emp_inf_archers_0", "wh_main_emp_inf_swordsmen",
               "wh2_main_skv_cha_warlord_0"]


def volley_states(batch=2):
    """(setup, six decisions) of a made-up battle whose missile units' projectiles fall at decisions 2
    and 5: the volley input goes 1, 1, 0, 1/8 or 1/10, 2/8 or 2/10, 0."""
    setup, state = sources.synthetic(batch=batch, own=3, enemy=3, seed=21, keys=VOLLEY_KEYS)
    N = 6
    state.update(men=np.maximum(state["men"], 1.0))
    for k in range(3):
        for t in ("on", "cd"):
            state[f"ab{k}_{t}"] = np.zeros((batch, N))
    a0 = setup.ammo0.astype(float)
    return setup, [dict(state, t=np.full(batch, 30.0 + i), a=np.maximum(a0 - 100 * n, 0))
                   for i, n in enumerate((0, 0, 1, 1, 1, 2))]


def pre_volley_networks(cfg):
    """An actor and critic as saved before the volley input: NEW_VOLLEY fewer token inputs."""
    actor, crit = policy.Actor(cfg), critic.Critic(cfg)
    actor.encoder.unit[0] = torch.nn.Linear(T0, cfg.d)
    crit.encoder.unit[0] = torch.nn.Linear(T0, cfg.critic_d)
    return actor.eval(), crit.eval()


def without_volley(o):
    return dict(o, tokens=o["tokens"][..., :T0])


def same_without_volley(old, new, old_crit, new_crit, atol=1e-5):
    """Old (no volley input) and new networks through the decisions of volley_states (memory carried):
    equal logits, memory, greedy orders and values, while the new input takes values inside 0..1."""
    setup, states = volley_states()
    m = mc = h_old = h_new = None
    keys = ("tokens", "own", "attend", "pos", "ctx")
    inside = 0
    for st in states:
        obs, m = ob.observe(st, setup, 1, m)
        cobs, mc = ob.observe(st, setup, 1, mc, full=True)
        o, c = policy.to_torch(obs), policy.to_torch(cobs)
        v = o["tokens"][..., ob.INDEX["volley_ready"]]
        inside += int(((v > 0) & (v < 1)).sum())
        with torch.no_grad():
            lo, h_old = old(without_volley(o), h_old)
            ln, h_new = new(o, h_new)
            vo = old_crit(without_volley({k: c[k] for k in keys}))
            vn = new_crit({k: c[k] for k in keys})
        for k in lo:
            assert torch.allclose(lo[k], ln[k], atol=atol), k
        assert torch.allclose(h_old, h_new, atol=atol) and torch.allclose(vo, vn, atol=atol)
        a, b = heads.sample(lo, greedy=True, abilities=True), heads.sample(ln, greedy=True, abilities=True)
        assert torch.equal(a.kind, b.kind) and torch.equal(a.target, b.target) and torch.equal(a.point, b.point)
    assert inside > 0


def test_networks_saved_before_the_volley_input_load_act_the_same_and_learn_it():
    torch.manual_seed(6)
    old, old_crit = pre_volley_networks(CFG)
    new, new_crit = model(seed=15), critic.Critic(CFG).eval()
    new.load_state_dict(old.state_dict())
    new_crit.load(old_crit.state_dict())
    assert torch.all(new.encoder.unit[0].weight[:, T0:] == 0) and torch.all(new_crit.encoder.unit[0].weight[:, T0:] == 0)
    same_without_volley(old, new, old_crit, new_crit)
    setup, states = volley_states()
    obs, _ = ob.observe(states[0], setup, 1)
    new.train()
    out, _ = new(policy.to_torch(obs))
    sum(v.masked_fill(v < -1e8, 0).sum() for v in out.values()).backward()
    assert new.encoder.unit[0].weight.grad[:, T0:].abs().sum() > 0


PRE_VOLLEY = [p for p in (ROOT / "build/nn-train/test5/w3_kiting30/m15.pt",      # wide (the kiting run)
                          ROOT / "build/nn-train/wide/w0.pt",                    # wide, widened from the small
                          ROOT / "build/nn-train/test5/r7_lostworst/m15.pt")     # small
              if p.exists()]


@pytest.mark.skipif(not PRE_VOLLEY, reason="no checkpoint saved before the volley input (build/ is not in Git)")
@pytest.mark.parametrize("path", PRE_VOLLEY, ids=[p.parent.name + "/" + p.name for p in PRE_VOLLEY])
def test_a_checkpoint_saved_before_the_volley_input_acts_as_before(path):
    from tools.nn.train import checkpoint
    data = checkpoint.read(path)
    cfg = checkpoint.config_of(data)
    new = checkpoint.load_policy(path)
    assert data["actor"]["encoder.unit.0.weight"].shape[1] == T0 and torch.all(new.encoder.unit[0].weight[:, T0:] == 0)
    old, old_crit = pre_volley_networks(cfg)
    torch.nn.Module.load_state_dict(old, data["actor"])
    # the critic: in the checkpoint, or beside it (widen.py --critic-dst <stem>_critic.pt), else a fresh one
    crit_path = path.with_name(path.stem + "_critic.pt")
    crit_data = (critic_state(data) if "critic" in data else
                 critic_state(checkpoint.read(crit_path)) if crit_path.exists() else old_crit.state_dict())
    torch.nn.Module.load_state_dict(old_crit, crit_data)
    new_crit = critic.Critic(cfg).eval()
    new_crit.load(dict(crit_data))
    same_without_volley(old, new, old_crit, new_crit, atol=1e-4)



# --- the fire arc (observation.ARC, 07.10.2026): one column, both sides, appended after the volley input ---

ARC_KEYS = ["wh2_main_skv_inf_night_runners_0", "wh2_dlc13_emp_inf_archers_0", "wh_main_emp_inf_swordsmen",
            "wh_main_emp_inf_handgunners"]


def arc_states(batch=2):
    """(setup, three decisions) of a made-up battle with throwing stars (arc 1.0), archers and handgunners (30 deg
    each side: 1/6) and swordsmen (no missile: 0) on both sides."""
    setup, state = sources.synthetic(batch=batch, own=4, enemy=4, seed=22, keys=ARC_KEYS)
    N = 8
    for k in range(3):
        for t in ("on", "cd"):
            state[f"ab{k}_{t}"] = np.zeros((batch, N))
    return setup, [dict(state, t=np.full(batch, 30.0 + i)) for i in range(3)]


def test_the_fire_arc_input_is_the_passports_arc_on_both_sides():
    setup, states = arc_states()
    obs, _ = ob.observe(states[0], setup, 1)
    col = obs.tokens[..., ob.INDEX["fire_arc"]]
    want = {"wh2_main_skv_inf_night_runners_0": 1.0, "wh2_dlc13_emp_inf_archers_0": 30 / 180,
            "wh_main_emp_inf_swordsmen": 0.0, "wh_main_emp_inf_handgunners": 30 / 180}
    assert ob.NAMES[T1] == "fire_arc" and ob.NAMES[T1 + 1:] == ob.ORDER
    for b, keys in enumerate(setup.keys):
        for i, k in enumerate(keys):
            assert col[b, i] == pytest.approx(want[k], abs=1e-6), (b, i, k)     # own (0-3) and enemy (4-7)


def pre_arc_networks(cfg):
    """An actor and critic as saved before the fire arc: NEW_ARC fewer token inputs."""
    actor, crit = policy.Actor(cfg), critic.Critic(cfg)
    actor.encoder.unit[0] = torch.nn.Linear(T1, cfg.d)
    crit.encoder.unit[0] = torch.nn.Linear(T1, cfg.critic_d)
    return actor.eval(), crit.eval()


def same_without_arc(old, new, old_crit, new_crit, atol=1e-5):
    """Old (no fire arc) and new networks through the decisions of arc_states (memory carried): equal logits,
    memory, greedy orders and values, while the new input is not zero."""
    setup, states = arc_states()
    m = mc = h_old = h_new = None
    keys = ("tokens", "own", "attend", "pos", "ctx")
    for st in states:
        obs, m = ob.observe(st, setup, 1, m)
        cobs, mc = ob.observe(st, setup, 1, mc, full=True)
        o, c = policy.to_torch(obs), policy.to_torch(cobs)
        assert o["tokens"][..., T1:].abs().sum() > 0
        cut = lambda d: dict(d, tokens=d["tokens"][..., :T1])
        with torch.no_grad():
            lo, h_old = old(cut(o), h_old)
            ln, h_new = new(o, h_new)
            vo = old_crit(cut({k: c[k] for k in keys}))
            vn = new_crit({k: c[k] for k in keys})
        for k in lo:
            assert torch.allclose(lo[k], ln[k], atol=atol), k
        assert torch.allclose(h_old, h_new, atol=atol) and torch.allclose(vo, vn, atol=atol)
        a, b = heads.sample(lo, greedy=True, abilities=True), heads.sample(ln, greedy=True, abilities=True)
        assert torch.equal(a.kind, b.kind) and torch.equal(a.target, b.target) and torch.equal(a.point, b.point)


def test_networks_saved_before_the_fire_arc_load_act_the_same_and_learn_it():
    torch.manual_seed(7)
    old, old_crit = pre_arc_networks(CFG)
    new, new_crit = model(seed=16), critic.Critic(CFG).eval()
    new.load_state_dict(old.state_dict())
    new_crit.load(old_crit.state_dict())
    assert torch.all(new.encoder.unit[0].weight[:, T1:] == 0) and torch.all(new_crit.encoder.unit[0].weight[:, T1:] == 0)
    same_without_arc(old, new, old_crit, new_crit)
    setup, states = arc_states()
    obs, _ = ob.observe(states[0], setup, 1)
    new.train()
    out, _ = new(policy.to_torch(obs))
    sum(v.masked_fill(v < -1e8, 0).sum() for v in out.values()).backward()
    assert new.encoder.unit[0].weight.grad[:, T1:].abs().sum() > 0


PRE_ARC = [p for p in (ROOT / "build/nn-train/test5/s44_defonly/m20.pt",)      # the chain's last step before it
           if p.exists()]


@pytest.mark.skipif(not PRE_ARC, reason="no checkpoint saved before the fire arc (build/ is not in Git)")
@pytest.mark.parametrize("path", PRE_ARC, ids=[p.parent.name + "/" + p.name for p in PRE_ARC])
def test_a_checkpoint_saved_before_the_fire_arc_acts_as_before(path):
    from tools.nn.train import checkpoint
    data = checkpoint.read(path)
    cfg = checkpoint.config_of(data)
    new = checkpoint.load_policy(path)
    assert data["actor"]["encoder.unit.0.weight"].shape[1] == T1 and torch.all(new.encoder.unit[0].weight[:, T1:] == 0)
    old, old_crit = pre_arc_networks(cfg)
    torch.nn.Module.load_state_dict(old, data["actor"])
    crit_path = path.with_name(path.stem + "_critic.pt")
    crit_data = (critic_state(data) if "critic" in data else
                 critic_state(checkpoint.read(crit_path)) if crit_path.exists() else old_crit.state_dict())
    torch.nn.Module.load_state_dict(old_crit, crit_data)
    new_crit = critic.Critic(cfg).eval()
    new_crit.load(dict(crit_data))
    same_without_arc(old, new, old_crit, new_crit, atol=1e-4)


# --- the grid move point (ModelConfig.grid) and the order-in-force input (ORDER), 07.10.2026 ---

GRID = config.preset("small", d=64, layers=2, heads=4, critic_d=64, critic_layers=2, critic_heads=4, grid=32)


def test_the_grid_head_picks_a_cell_whose_centre_is_the_point_inside_the_map():
    setup, state, obs = setup_obs()
    actor = model(GRID)
    o = policy.to_torch(obs)
    with torch.no_grad():
        logits, _ = actor(o)
    assert logits["point"].shape[-1] == 32 * 32 == GRID.points
    a = heads.sample(logits)
    a.kind = torch.where(o["ctrl"], torch.full_like(a.kind, heads.MOVE), a.kind)
    bounds = torch.as_tensor(setup.bounds)
    orders = decide.to_orders(GRID, a, o, decide.frame_to(obs.frame, None), bounds)
    own = o["ctrl"]
    assert ((orders.x >= bounds[:, :1] - 1e-3) & (orders.x <= bounds[:, 1:2] + 1e-3))[own].all()
    assert ((orders.z >= bounds[:, 2:3] - 1e-3) & (orders.z <= bounds[:, 3:4] + 1e-3))[own].all()
    # the cell's point does not depend on where the unit stands: the same cell is the same place
    fr = decide.frame_to(obs.frame, None)
    f, l = fr.point(orders.x, orders.z)
    want = heads.cell_centres(GRID)[a.point]
    inside = (want.abs() < 700).all(-1) & own                  # cells away from the map's edge: not clipped
    assert torch.allclose(torch.stack([f, l], -1)[inside], want[inside], atol=1e-2)
    lp, ent = heads.log_prob(logits, a, o["ctrl"])
    assert torch.isfinite(lp).all() and torch.isfinite(ent).all()


def test_a_fresh_grid_head_prefers_the_cells_near_the_unit():
    setup, state, obs = setup_obs()
    o = policy.to_torch(obs)
    with torch.no_grad():
        logits, _ = model(GRID)(o)
    c = heads.cell_centres(GRID)
    pos = o["pos"] * ob.POS
    best = c[logits["point"].argmax(-1)]
    step = 2 * GRID.grid_half / GRID.grid
    near = ((best - pos) ** 2).sum(-1).sqrt() < 2 * step
    assert near[o["own"]].all()


def test_a_bin_checkpoint_loads_into_the_grid_actor_with_only_the_point_head_fresh(tmp_path):
    from tools.nn.train import checkpoint, ppo, run
    old = model(CFG, seed=3)
    new = run.with_grid(old, 32)
    assert new.cfg.grid == 32 and not hasattr(new.heads, "point")
    ns, os_ = new.state_dict(), old.state_dict()
    for k, v in os_.items():
        if not k.startswith("heads.point."):
            assert torch.equal(ns[k], v), k
    assert {k for k in ns if k not in os_} == {"heads.cell_q.weight", "heads.cell_q.bias", "heads.cell_k",
                                               "heads.cell_near.weight", "heads.cell_near.bias"}
    setup, state, obs = setup_obs()
    o = policy.to_torch(obs)
    with torch.no_grad():
        lo, _ = old(o)
        ln, _ = new(o)
    for k in ("kind", "target", "run"):
        assert torch.allclose(lo[k], ln[k], atol=1e-5), k
    # a bin actor's whole state dict (with its point head) loads into a grid actor too
    policy.Actor(GRID if CFG.d == GRID.d else new.cfg).load_state_dict(old.state_dict())
    # a grid checkpoint round trip: the config carries the grid
    path = checkpoint.save(tmp_path / "g.pt", new)
    back = checkpoint.load_policy(path)
    assert back.cfg.grid == 32 and torch.equal(back.heads.cell_k, new.heads.cell_k)
    # the anchor's KL needs only the kind and the target: it is defined between a grid and a bin actor
    assert torch.isfinite(ppo.anchor_kl(ln, lo, o["ctrl"]))


def test_the_teachers_label_on_the_grid_is_the_nearest_cell():
    from tools.nn.train.drills import teach
    setup, state, obs = setup_obs()
    o = policy.to_torch(obs)
    fr = decide.frame_to(obs.frame, None)
    c = heads.cell_centres(GRID)
    B, N = o["pos"].shape[:2]
    want = torch.tensor([5, 700])[:B, None].expand(B, N)
    x, z = fr.world(c[want][..., 0] + 3.0, c[want][..., 1] - 4.0)
    assert torch.equal(teach.point_bins(GRID, o, fr, x, z), want)


def test_a_repeated_cell_target_or_kind_is_free_and_a_new_one_is_a_change():
    from tools.nn.sim import orders as O
    from tools.nn.train import reward
    u = {"order_kind": torch.tensor([[O.MOVE, O.MOVE, O.ATTACK, O.HOLD]]),
         "order_target": torch.tensor([[-1, -1, 2, -1]]),
         "ox": torch.tensor([[100.0, 100.0, 0, 0]]), "oz": torch.tensor([[50.0, 50.0, 0, 0]]),
         "side": torch.tensor([[1, 1, 1, 1]]), "men": torch.ones(1, 4), "r": torch.zeros(1, 4, dtype=torch.bool),
         "s": torch.zeros(1, 4, dtype=torch.bool), "gone": torch.zeros(1, 4, dtype=torch.bool),
         "ms": torch.full((1, 4), 2)}
    no = torch.zeros(1, 4, dtype=torch.bool)
    same = O.Orders(kind=torch.tensor([[O.MOVE, O.KEEP, O.ATTACK, O.HOLD]]), x=torch.tensor([[100.0, 0, 0, 0]]),
                    z=torch.tensor([[50.0, 0, 0, 0]]), target=torch.tensor([[-1, -1, 2, -1]]), run=no)
    assert not reward.order_changes(u, same).any()
    cell = 2 * GRID.grid_half / GRID.grid
    other = O.Orders(kind=torch.tensor([[O.MOVE, O.MOVE, O.ATTACK, O.MOVE]]),
                     x=torch.tensor([[100.0 + cell, 0, 0, 0]]), z=torch.tensor([[50.0, 0, 0, 0]]),
                     target=torch.tensor([[-1, -1, 3, -1]]), run=no)
    assert reward.order_changes(u, other).tolist() == [[True, True, True, True]]

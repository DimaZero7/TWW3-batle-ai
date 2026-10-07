"""v2 (tools/nn/model config sectors > 0): the map's sector tokens, the chained heads, the commitment (torch).

torch is not in the project's .venv: skipped there, run in the training container (snake-ai-trainer, pytest on
PYTHONPATH).
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn.model import chain, commit, config, decide, heads, policy, sectors, sources  # noqa: E402
from tools.nn.model import observation as ob  # noqa: E402
from tools.nn.model.frame import Frame  # noqa: E402
from tools.nn.sim.orders import ATTACK, HOLD, KEEP, MOVE, WITHDRAW  # noqa: E402

CFG = config.preset("v2", d=64, layers=2, heads=4, pointer=32, critic_d=64, critic_layers=2, critic_heads=4,
                    sector_d=32)


def model(cfg=CFG, seed=0):
    torch.manual_seed(seed)
    return policy.Actor(cfg).eval()


def setup_obs(seed=0, own=5, enemy=6, batch=2, **changes):
    setup, state = sources.synthetic(batch=batch, own=own, enemy=enemy, seed=seed)
    state.update(changes)
    obs, _ = ob.observe(state, setup, 1)
    return setup, state, obs


def calm(batch=2, own=5, enemy=6):
    """A synthetic battle where every unit stands steady, alive, seen, out of melee and unthreatened."""
    setup, state = sources.synthetic(batch=batch, own=own, enemy=enemy, seed=0)
    shape = state["ms"].shape
    state.update(ms=np.ones(shape), men=np.full(shape, 100.0), vis=np.ones(shape, bool), r=np.zeros(shape, bool),
                 m=np.zeros(shape, bool), lf=np.zeros(shape, bool), rf=np.zeros(shape, bool), bf=np.zeros(shape, bool))
    if "s" in state:
        state["s"] = np.zeros(shape, bool)
    obs, _ = ob.observe(state, setup, 1)
    return setup, state, obs


def v2_obs(obs, setup, state=None, t=0.0):
    o = policy.to_torch(obs)
    fr = decide.frame_to(obs.frame, "cpu")
    st = state or commit.start(*o["own"].shape)
    return decide.v2_inputs(model(), o, fr, torch.as_tensor(setup.bounds), st, torch.full((o["own"].shape[0],), t))


def test_actor_of_a_v2_config_is_the_v2_actor_and_old_presets_are_not():
    assert isinstance(model(), policy.ActorV2)
    assert not isinstance(policy.Actor(config.preset("small", d=64, layers=2)), policy.ActorV2)
    assert config.V2.points == 64 * 64


def test_cells_and_sectors_are_one_grid():
    cfg = CFG
    p = torch.arange(cfg.points)
    s, f = sectors.split(cfg, p)
    assert torch.equal(sectors.point_index(cfg, s, f), p)
    c = sectors.cells(cfg)                                     # [S, 16, 2]
    # every cell's centre lies in its sector: within half a sector of the sector's centre
    assert torch.all((c - sectors.centres(cfg)[:, None]).abs() < sectors.step(cfg) / 2)
    # the cells together are the grid of 25 m, 64 x 64: cell p's centre is the plain grid's cell p
    flat = config.preset("small", grid=64, grid_half=CFG.grid_half)
    assert torch.allclose(c.reshape(-1, 2)[heads._cell_order(cfg, p)], heads.cell_centres(flat, None))


def test_sector_features_count_own_and_visible_enemies_where_they_stand():
    setup, state, obs = setup_obs()
    o = v2_obs(obs, setup)
    cells_in = sectors.inside(CFG, o["geo"])
    f = sectors.features(CFG, o, cells_in)                     # [B, S, F]
    i = sectors.FEATURES.index
    own = (o["own"] & o["attend"]).sum(-1).float()
    assert torch.allclose(f[..., i("own_units")].sum(-1) * sectors.UNITS, own)
    vis = (~o["own"] & o["attend"] & (o["tokens"][..., ob.INDEX["visible"]] > 0.5)).sum(-1).float()
    assert torch.allclose(f[..., i("enemy_units")].sum(-1) * sectors.UNITS, vis)
    # a unit's sector is the one its position falls in
    pos = o["pos"][0, 0] * ob.POS
    j = ((pos + CFG.grid_half) // sectors.step(CFG)).long()
    assert f[0, j[0] * CFG.sectors + j[1], i("own_units")] > 0
    # hidden enemies are not counted as visible
    hidden = np.zeros((2, 11), bool)
    hidden[:, 5:] = True
    setup, state, obs = setup_obs(vis=~hidden)
    o = v2_obs(obs, setup)
    assert float(sectors.features(CFG, o, sectors.inside(CFG, o["geo"]))[..., i("enemy_units")].sum()) == 0


def test_cells_outside_the_map_are_never_chosen_and_the_point_is_the_cells_centre():
    setup, state, obs = setup_obs()
    setup.bounds[:] = [-300, 300, -200, 200]                   # a small map: most of the grid is outside
    obs, _ = ob.observe(state, setup, 1)
    net = model()
    for seed in range(3):
        torch.manual_seed(seed)
        st = {}
        orders, _, logits, a = decide.act(net, obs, setup, commit_state=st, t=np.zeros(2))
        move = (a.kind == MOVE) | (a.kind == WITHDRAW)
        assert torch.all(orders.x[move].abs() <= 300) and torch.all(orders.z[move].abs() <= 200)
        o = v2_obs(obs, setup)
        ok = sectors.inside(CFG, o["geo"]).reshape(2, -1)                   # [B, S * 16] sector-major
        row = heads._cell_order(CFG, a.point)
        assert torch.all(ok.gather(1, row)[o["ctrl"]])


def test_the_chain_conditions_on_the_earlier_parts_and_log_prob_matches_the_sample():
    setup, state, obs = setup_obs()
    o = v2_obs(obs, setup)
    net = model()
    torch.manual_seed(1)
    logits, a, _ = net.act(o)
    again, _ = net(o, action=a)
    for k in logits:
        assert torch.allclose(logits[k], again[k], atol=1e-5), k
    lp, ent = heads.log_prob(logits, a, o["ctrl"])
    assert torch.all(torch.isfinite(lp)) and torch.all(lp <= 1e-6)
    assert torch.all(lp[~o["ctrl"]] == 0)
    # another kind gives another place's logits (the place knows the kind)
    other = heads.Action(torch.where(a.kind == MOVE, WITHDRAW, MOVE), a.point, a.target, a.run, a.ability, a.commit)
    o2, _ = net(o, action=other)
    assert not torch.allclose(o2["sector"], again["sector"])
    # another sector gives another cell's logits (the cell knows the sector)
    s, f = sectors.split(CFG, a.point)
    moved = heads.Action(a.kind, sectors.point_index(CFG, (s + 17) % 256, f), a.target, a.run, a.ability, a.commit)
    o3, _ = net(o, action=moved)
    assert not torch.allclose(o3["fine"], again["fine"])
    assert torch.allclose(o3["sector"], again["sector"])


def test_log_prob_counts_each_part_only_where_it_matters():
    setup, state, obs = setup_obs()
    o = v2_obs(obs, setup)
    net = model()
    logits, a, _ = net.act(o)
    B, N = a.kind.shape
    full = lambda k: torch.full((B, N), k, dtype=torch.long)
    base = dict(point=a.point, target=a.target.clamp(min=0) * 0 - 1, run=a.run, ability=None, commit=a.commit)
    hold = heads.Action(kind=full(HOLD), **base)
    lp_hold, _ = heads.log_prob(logits, hold, o["ctrl"])
    kind_lp = torch.log_softmax(logits["kind"], -1)[..., HOLD]
    c_lp = torch.log_softmax(logits["commit"], -1).gather(-1, a.commit[..., None])[..., 0]
    assert torch.allclose(lp_hold[o["ctrl"]], (kind_lp + c_lp)[o["ctrl"]], atol=1e-5)
    keep = heads.Action(kind=full(KEEP), **base)
    lp_keep, _ = heads.log_prob(logits, keep, o["ctrl"])
    assert torch.allclose(lp_keep[o["ctrl"]], torch.log_softmax(logits["kind"], -1)[..., KEEP][o["ctrl"]], atol=1e-5)


def test_a_held_unit_may_only_keep_and_its_choice_has_no_log_prob():
    setup, state, obs = calm()
    o = policy.to_torch(obs)
    B, N = o["own"].shape
    st = settled(commit.start(B, N), o)
    st["until"][:, :2] = 10.0                                   # units 0 and 1 committed until t 10
    o = decide.v2_inputs(model(), o, decide.frame_to(obs.frame, "cpu"), torch.as_tensor(setup.bounds), st,
                         torch.full((B,), 3.0))
    assert torch.all(~o["free"][:, :2]) and torch.all(o["free"][:, 2:5])
    assert torch.allclose(o["commit"][:, :2, 0], torch.tensor(7.0 / commit.LEFT_MAX))
    net = model()
    for seed in range(5):
        torch.manual_seed(seed)
        logits, a, _ = net.act(o)
        assert torch.all(a.kind[:, :2] == KEEP)
        lp, ent = heads.log_prob(logits, a, o["ctrl"])
        assert torch.all(lp[:, :2].abs() < 1e-6) and torch.all(ent[:, :2].abs() < 1e-4)


def settled(st, o):
    """The state as if the last decision saw the same flags (no event now)."""
    st["melee"], st["threat"], st["rout"], st["lord"] = commit._flags(o)
    return st


def _obs_with(obs, **cols):
    o = policy.to_torch(obs)
    tok = o["tokens"].clone()
    for name, (unit, v) in cols.items():
        tok[:, unit, ob.INDEX[name]] = v
    return dict(o, tokens=tok)


def test_the_commitment_starts_with_a_new_order_runs_out_and_keep_starts_none():
    setup, state, obs = calm()
    o = policy.to_torch(obs)
    B, N = o["own"].shape
    st = commit.start(B, N)
    o1 = commit.inputs(st, o, torch.zeros(B))
    kind = torch.full((B, N), KEEP)
    kind[:, 0], kind[:, 1] = MOVE, ATTACK
    target = torch.full((B, N), -1)
    target[:, 1] = 7
    a = heads.Action(kind, torch.zeros_like(kind), target, torch.zeros((B, N), dtype=torch.bool), None,
                     torch.tensor([2, 0, 3, 1, 1] + [0] * (N - 5)).expand(B, N))
    st = commit.apply(st, o1, torch.zeros(B), a)
    assert torch.allclose(st["until"][:, :2], torch.tensor([8.0, 2.0]))
    assert torch.all(st["until"][:, 2:] == -1) and torch.all(st["target"][:, 1] == 7)
    o2 = commit.inputs(st, o, torch.full((B,), 1.0))
    assert torch.all(~o2["free"][:, :2]) and torch.all(o2["free"][:, 2:5])
    o3 = commit.inputs(st, o, torch.full((B,), 2.5))        # unit 1's 2 s ran out, unit 0's 8 s not
    assert torch.all(o3["free"][:, 1]) and torch.all(~o3["free"][:, 0])
    # the held unit keeps its commitment through the next decision's apply
    keep = heads.Action(torch.full((B, N), KEEP), a.point, a.target, a.run, None, a.commit)
    st2 = commit.apply(st, o3, torch.full((B,), 2.5), keep)
    assert torch.allclose(st2["until"][:, 0], torch.tensor(8.0)) and torch.all(st2["until"][:, 1] == -1)


@pytest.mark.parametrize("event", ["melee", "threat", "target_gone", "target_routs", "lord", "rally"])
def test_an_event_interrupts_the_commitment(event):
    setup, state, obs = calm()
    o = policy.to_torch(obs)
    B, N = o["own"].shape
    st = settled(commit.start(B, N), o)
    st["until"][:, 0] = 100.0
    st["target"][:, 0] = 7
    assert torch.all(~commit.inputs(st, o, torch.zeros(B))["free"][:, 0])
    if event == "melee":
        o = _obs_with(obs, melee=(0, 1.0))
    elif event == "threat":
        o = _obs_with(obs, threat_rear=(0, 1.0))
    elif event == "target_gone":
        o = dict(o, target_ok=o["target_ok"].clone())
        o["target_ok"][:, 7] = False
    elif event == "target_routs":
        o = _obs_with(obs, state_routing=(7, 1.0))
    elif event == "lord":
        ctx = o["ctx"].clone()
        ctx[:, commit.OWN_LORD] = 1.0
        o = dict(o, ctx=ctx)
    elif event == "rally":
        st["rout"][:, 0] = True                              # it was routing at the last decision, now it is not
    assert torch.all(commit.inputs(st, o, torch.zeros(B))["free"][:, 0])
    # the same flag on at the last decision too: no new event, the commitment holds (except the target's, the lord's
    # and the rally's, which are states, not edges)
    if event in ("melee", "threat"):
        st2 = dict(st, melee=st["melee"] | (event == "melee"), threat=st["threat"] | (event == "threat"))
        assert torch.all(~commit.inputs(st2, o, torch.zeros(B))["free"][:, 0])


def test_the_companion_path_keeps_the_commitment_between_decisions():
    setup, state, obs = calm()
    net = model()
    st = {}
    torch.manual_seed(0)
    _, h, _, a = decide.act(net, obs, setup, commit_state=st, t=np.zeros(2))
    new = (a.kind != KEEP) & torch.as_tensor(obs.ctrl)
    assert torch.all(st["until"][new] >= 2.0) and torch.all(st["until"][~new] == -1)
    _, _, _, a2 = decide.act(net, obs, setup, h, commit_state=st, t=np.full(2, 1.0))
    assert torch.all(a2.kind[new] == KEEP)


def test_a_v2_checkpoint_saves_loads_and_acts_the_same(tmp_path):
    from tools.nn.train import checkpoint
    net = model()
    path = checkpoint.save(tmp_path / "v2.pt", net, None, "v2")
    again = checkpoint.load_policy(path)
    assert isinstance(again, policy.ActorV2)
    setup, state, obs = setup_obs()
    o = v2_obs(obs, setup)
    torch.manual_seed(3)
    l1, a1, _ = net.act(o)
    l2, _ = again(o, action=a1)
    for k in l1:
        assert torch.allclose(l1[k], l2[k], atol=1e-5), k


def test_sequence_matches_step_by_step_decisions():
    setup, state, obs = setup_obs()
    o = v2_obs(obs, setup)
    net = model()
    T = 3
    h, logits, acts = None, [], []
    for _ in range(T):
        lg, a, h = net.act(o, h)
        logits.append(lg)
        acts.append(a)
    seq = {k: torch.stack([v] * T) for k, v in o.items()}
    act = heads.Action(**{f: torch.stack([getattr(a, f) for a in acts]) for f in acts[0].parts()})
    out, h2 = net.sequence(seq, None, torch.zeros((T, 2), dtype=torch.bool), act)
    for t in range(T):
        for k in ("kind", "sector", "fine", "commit"):
            assert torch.allclose(out[k][t], logits[t][k], atol=1e-4), (t, k)
    assert torch.allclose(h, h2, atol=1e-5)


# --- training: the rollout keeps the commitment, PPO trains every head, run.py's options ---

from tools.nn.model import critic as mcritic  # noqa: E402
from tools.nn.train import checkpoint, evaluate, league, ppo, rollout  # noqa: E402

TINY = config.preset("v2", d=32, layers=2, heads=2, pointer=16, critic_d=32, critic_layers=1, critic_heads=2,
                     sector_d=16)
MIRROR = [("arena", "attack")]


def tiny(seed=0):
    torch.manual_seed(seed)
    return policy.Actor(TINY).eval(), mcritic.Critic(TINY).eval()


def test_the_rollout_holds_committed_units_and_restarts_their_commitment():
    import dataclasses
    actor, crit = tiny()
    env = rollout.Battles(league.layout(4, 1, {"self": 0.5, "nearest": 0.5}), MIRROR)
    steps = [env.step(actor, crit) for _ in range(4)]
    seen_held = False
    for s in steps:
        o, a = s["obs"], s["action"]
        held = o["ctrl"] & ~o["free"]
        seen_held |= bool(held.any())
        assert torch.all(a.kind[held] == KEEP)
        no_ability = ~o["abil_ok"].any(-1) if "abil_ok" in o else torch.ones_like(held)
        assert torch.all(s["lp"][held & no_ability] == 0)          # (a held lord still chooses its abilities)
        assert o["commit"].shape[-1] == 2 and o["geo"].shape[-1] == 8
    assert seen_held
    # a battle that starts again starts with no commitment
    done = torch.zeros(env.B, dtype=torch.bool)
    done[0] = True
    env.cstate["until"].fill_(50.0)
    env._reset(done)
    assert torch.all(env.cstate["until"][[0, env.B]] == -1) and torch.all(env.cstate["until"][1] == 50.0)
    # narrowing the batch keeps each battle's commitment
    env.cstate["until"][1] = 7.0
    env.narrow([1, 2])
    assert torch.all(env.cstate["until"][0] == 7.0) and env.cstate["until"].shape[0] == 4
    del dataclasses


def test_a_tiny_v2_training_step_trains_the_chain_and_the_commitment():
    import dataclasses
    actor, crit = tiny()
    before = {n: p.detach().clone() for n, p in actor.named_parameters()}
    env = rollout.Battles(league.layout(6, 1, {"self": 0.34, "past": 0.33, "hold_shoot": 0.33}), MIRROR)
    env.set_past(tiny(1)[0], untrained=True)
    batch = rollout.collect(env, actor, crit, 3)
    assert batch["action"].commit is not None and batch["obs"]["free"].shape == (3, env.R, env.N)
    ref = tiny(2)[0]
    opt = torch.optim.Adam(list(actor.parameters()) + list(crit.parameters()), lr=1e-3)
    st = ppo.update(actor, crit, opt, batch, dataclasses.replace(ppo.PPOConfig(), epochs=2, minibatch=6, anchor=0.1),
                    reference=ref)
    assert all(np.isfinite(v) for v in st.values() if isinstance(v, float))
    changed = {n for n, p in actor.named_parameters() if not torch.equal(before[n], p)}
    for part in ("heads.kind.", "heads.commit.", "heads.place_q.", "heads.fine.", "sector_enc.", "sector_att.",
                 "commit_in."):
        assert any(n.startswith(part) for n in changed), part
    assert ppo.distance(actor, actor, batch, 6) == pytest.approx(0.0, abs=1e-6)


def test_evaluation_plays_a_v2_network_against_a_v1_past_version():
    actor, _ = tiny()
    small = policy.Actor(config.preset("small", d=32, layers=2, heads=2, pointer=16)).eval()
    res = evaluate.play(actor, opponents=("hold", "past"), per_scene=2, past=small, limit_s=2.0, scene_list=MIRROR)
    assert res["by_opponent"]["hold"]["games"] == 2 and res["by_opponent"]["past"]["games"] == 2


def test_run_starts_v2_from_its_own_untrained_network_with_the_v2_reward(tmp_path, monkeypatch):
    import json
    from tools.nn.train import run
    monkeypatch.setattr(checkpoint, "DIR", tmp_path)
    monkeypatch.setattr(checkpoint, "RANDOM_V2", tmp_path / "random_v2.pt")
    monkeypatch.setitem(config.PRESETS, "v2", TINY)
    args = run.parser().parse_args(["--name", "v2", "--preset", "v2", "--reward", "v2", "--battles", "4", "--steps",
                                    "2", "--updates", "2", "--minutes", "5", "--device", "cpu", "--no-eval",
                                    "--mix", '{"self": 0.5, "nearest": 0.5}', "--bank", "8", "--max-units", "4"])
    trained, summary, out = run.train(args)
    assert isinstance(trained, policy.ActorV2) and (tmp_path / "random_v2.pt").exists()
    assert (args.order_cost, args.lord, args.idle, args.retarget) == (0.006, 0.0, 0.0, 0.0)
    rows = [json.loads(x) for x in (out / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2 and all(r["reward_parts"] for r in rows)
    for role in rows[-1]["reward_parts"].values():
        assert role.get("lord", 0) == 0 and role.get("idle", 0) == 0

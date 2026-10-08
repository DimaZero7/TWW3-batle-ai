"""v2 (tools/nn/model config sectors > 0): the map's sector tokens, the chained heads, the commitment (torch).

torch is not in the project's .venv: skipped there, run in the training container (snake-ai-trainer, pytest on
PYTHONPATH).
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn.model import chain, commit, config, decide, eyes, heads, policy, sectors, sources  # noqa: E402
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
    st["melee"], threat, st["rout"], st["lord"] = commit._flags(o)
    st["threat"] = threat                                          # (a threat already on counted, since long ago)
    st["threat_t"] = torch.where(threat, torch.zeros_like(st["threat_t"]), st["threat_t"])
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
    assert torch.all(~commit.inputs(st, o, torch.full((B,), 10.0))["free"][:, 0])
    if event == "melee":
        o = _obs_with(obs, melee=(0, 1.0))
    elif event == "threat":
        o = _obs_with(obs, threat_rear=(0, 1.0))
        st["threat_t"][:, 0] = 8.0                                 # on since 2 s before (t 10)
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
    assert torch.all(commit.inputs(st, o, torch.full((B,), 10.0))["free"][:, 0])
    # the same flag on at the last decision too: no new event, the commitment holds (except the target's, the lord's
    # and the rally's, which are states, not edges)
    if event in ("melee", "threat"):
        st2 = dict(st, melee=st["melee"] | (event == "melee"), threat=st["threat"] | (event == "threat"))
        assert torch.all(~commit.inputs(st2, o, torch.full((B,), 10.0))["free"][:, 0])


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


def test_a_tiny_v2_training_step_trains_the_chain_and_the_commitment(monkeypatch):
    import dataclasses
    monkeypatch.setattr(eyes, "HORIZONS_S", (1.0, 2.0))         # windows inside the 3-decision chunk
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
    assert batch["eyes"]["own"].shape == (3, env.R, env.N, 2)
    for h in eyes.HEADS:
        assert np.isfinite(st[f"eyes_{h}"]) and f"eyes_ev_{h}" in st
    for part in ("heads.kind.", "heads.commit.", "heads.place_q.", "heads.fine.", "sector_enc.", "sector_att.",
                 "commit_in.", "eyes.own.", "eyes.threat.", "eyes.danger.", "eyes.own_in."):
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
                                    "--mix", '{"self": 0.5, "nearest": 0.5}', "--bank", "8", "--max-units", "4",
                                    "--keep-every", "1e-6"])
    trained, summary, out = run.train(args)
    assert isinstance(trained, policy.ActorV2) and (tmp_path / "random_v2.pt").exists()
    assert (args.order_cost, args.lord, args.idle, args.retarget) == (0.006, 0.0, 0.0, 0.0)
    rows = [json.loads(x) for x in (out / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2 and all(r["reward_parts"] for r in rows)
    assert isinstance(checkpoint.load_policy(out / "m0.pt"), policy.ActorV2)          # --keep-every: kept copies
    assert all("eyes_ev_own10" in r for r in rows[1:])                                # the eyes in the log
    for role in rows[-1]["reward_parts"].values():
        assert role.get("lord", 0) == 0 and role.get("idle", 0) == 0


# --- the eyes ---

def test_the_eyes_targets_are_the_change_over_the_window_cut_at_the_battles_end_and_masked_past_the_chunk():
    T, R, N = 6, 2, 1
    before = torch.zeros(T, R, N)
    after = torch.zeros(T, R, N)
    hp = torch.tensor([1.0, 0.9, 0.7, 0.6, 0.6, 0.5])          # row 0: health share at each decision's observation
    before[:, 0, 0] = hp
    after[:-1, 0, 0] = hp[1:]
    after[-1, 0, 0] = 0.4
    done = torch.zeros(T, R, dtype=torch.bool)
    done[2, 1] = True                                            # row 1's battle ends after decision 2
    before[:, 1, 0] = torch.tensor([1.0, 0.8, 0.5, 1.0, 1.0, 1.0])     # a new battle from decision 3
    after[:, 1, 0] = torch.tensor([0.8, 0.5, 0.2, 1.0, 1.0, 0.9])
    ch, ok = eyes.change(before, after, done, 3)                 # 3 decisions ahead
    assert torch.allclose(ch[:4, 0, 0], torch.tensor([0.6, 0.6, 0.5, 0.4]) - hp[:4])
    assert ok[:, 0].tolist() == [True, True, True, True, False, False]
    assert torch.allclose(ch[:3, 1, 0], torch.tensor([0.2, 0.2, 0.2]) - before[:3, 1, 0])   # cut at the end
    assert ok[:, 1].tolist() == [True, True, True, True, False, False]
    tg = eyes.targets({"hp": before, "gold": torch.zeros(T, R, N)}, {"hp": after, "gold": torch.ones(T, R, N)},
                      done, torch.full((T, R), 2.0), 10.0)       # decisions of 10 s: 10 s = 1, 30 s = 3 decisions
    assert torch.allclose(tg["own"][0, 0, 0], torch.tensor([0.1, 0.4]))
    assert torch.allclose(tg["threat"][0, 0, 0], torch.tensor(1.0 / 2.0 * eyes.THREAT_SCALE))


def test_the_eyes_feed_back_detached_and_learn_only_from_their_loss():
    setup, state, obs = setup_obs()
    o = v2_obs(obs, setup)
    net = model()
    logits, a, _ = net.act(o)
    lp, _ = heads.log_prob(logits, a, o["ctrl"])
    net.zero_grad()
    (-lp.sum()).backward(retain_graph=True)
    for name in ("own", "threat", "danger"):
        g = getattr(net.eyes, name)[-1].weight.grad
        assert g is None or float(g.abs().sum()) == 0.0, name        # the policy does not train the eyes
    assert net.eyes.own_in.weight.grad is not None                   # ... it trains how it reads them
    net.zero_grad()
    B, N = o["own"].shape
    tg = {"own": torch.full((B, N, 2), 0.2), "threat": torch.ones(B, N), "ok": torch.ones(B, 2, dtype=torch.bool)}
    total, parts = eyes.loss(CFG, logits, tg, o)
    own = o["own"] & o["attend"]
    assert torch.allclose(parts["own10"][0], ((logits["eyes_own"][..., 0] - 0.2) ** 2)[own].mean())
    assert all(float(v[0]) >= 0 for v in parts.values())
    total.backward()
    assert set(parts) == set(eyes.HEADS)
    for name in ("own", "threat", "danger"):
        assert float(getattr(net.eyes, name)[-1].weight.grad.abs().sum()) > 0, name
    assert logits["eyes_own"].shape == (B, N, 2) and logits["eyes_danger"].shape == (B, CFG.sectors ** 2)


def test_the_simulators_gold_out_is_the_gold_of_health_the_other_side_lost():
    # (melee only: an own archer's arrows into our own men are lost health no enemy's gold_out counts)
    from tools.nn.sim import battle, scenario
    from tools.nn.train import opponents
    army = {"attacker": 1, "sides": {
        1: {"faction": "wh_main_emp_empire", "units": [{"key": "wh_main_emp_inf_spearmen_0", "x": -30, "z": 0, "b": 90},
                                                       {"key": "wh_main_emp_inf_spearmen_0", "x": -30, "z": 40, "b": 90}]},
        2: {"faction": "wh_main_emp_empire", "units": [{"key": "wh_main_emp_inf_spearmen_0", "x": 30, "z": 0, "b": 270}]}}}
    st = scenario.build([army])
    for _ in range(60):
        battle.step(st, opponents.nearest(st))
    u = st.u
    lost = u["cost"] * (1 - u["hp_abs"] / u["hp0"].clamp(min=1e-6))
    for s in (1, 2):
        dealt = float((u["gold_out"] * (u["side"] == s)).sum())
        took = float((lost * (u["side"] == 3 - s)).sum())
        assert took > 0 and dealt == pytest.approx(took, rel=1e-3), s


# --- the companion and the simulator see the same v2 inputs on the same state ---

def _game_doc(st, ours, move):
    """The bridge's state document of battle 0 (src/entries/nn_arena.lua): our side is the document's side 1."""
    from tools.nn.companion import exchange
    u = {k: v[0].tolist() for k, v in st.u.items()}
    name = lambda i: f"{'own' if u['side'][i] == ours else 'enemy'}_{i}"
    units = []
    for i, key in enumerate(st.keys[0]):
        if not key:
            continue
        tgt = u["target"][i]
        units.append({"n": name(i), "side": 1 if u["side"][i] == ours else 2, "key": key, "x": u["x"][i],
                      "z": u["z"][i], "b": u["b"][i], "men": u["men"][i], "hp": u["hp"][i], "mp": u["mp"][i],
                      "ms": u["ms"][i], "r": u["r"][i], "s": u["s"][i], "w": u["w"][i], "m": u["m"][i],
                      "mv": u["mv"][i], "f": False, "a": u["a"][i], "fire": u["fire"][i],
                      "t": name(tgt) if tgt >= 0 else "", "fat": exchange.FATIGUE_LEVELS[int(u["fat"][i])],
                      "k": u["k"][i], "ox": u["ox"][i], "oz": u["oz"][i], "lf": u["lf"][i], "rf": u["rf"][i],
                      "bf": u["bf"][i], "v": True})
    attacker = int(st.attacker[0])
    return {"batch": "twin", "move": move, "t": round(float(st.t[0]) * 1000), "done": False,
            "attacker": 1 if attacker == ours else 2, "decide_ms": 1000,
            "factions": {"own": "wh_main_emp_empire", "enemy": "wh_main_emp_empire"}, "units": units}


@pytest.mark.parametrize("ours", [1, 2])
def test_the_companion_sees_the_v2_inputs_as_the_simulator_on_one_state(ours, monkeypatch):
    from tools.nn.companion import loop
    from tools.nn.model import sources as msources
    from tools.nn.sim import battle, scenario
    from tools.nn.train import opponents, scenes as mscenes
    left = [{"key": "wh_main_emp_inf_spearmen_0", "x": -60, "z": z, "b": 90} for z in (-80, 0, 80)]
    right = [{"key": "wh_main_emp_inf_swordsmen", "x": 60, "z": z, "b": 270} for z in (-40, 40)]
    left.append({"key": "wh_main_emp_cha_general_0", "general": True, "x": -140, "z": 0, "b": 90})
    st = scenario.build([{"attacker": 1, "sides": {1: {"faction": "wh_main_emp_empire", "units": left},
                                                   2: {"faction": "wh_main_emp_empire", "units": right}}}])
    for _ in range(50):                                          # 25 s: they meet, fight, lose men
        battle.step(st, opponents.nearest(st))
    assert bool(st.u["m"][0].any())
    setup, _ = msources.from_sim(st)
    setup = ob.Setup(keys=setup.keys, side=setup.side, bounds=np.tile(np.array(msources.CROSSROADS, np.float32),
                     (st.B, 1)), factions=setup.factions, attacker=setup.attacker)
    live = mscenes.LiveSetup.of(setup, "cpu")
    net = model()
    # the simulator's decision (rollout.Battles: observe, then the v2 inputs)
    obs, _ = ob.observe(st.observation(), live, ours, None)
    o_sim = policy.to_torch(obs)
    o_sim = rollout._v2_inputs(commit.start(1, st.N), o_sim, st.t, decide.frame_to(obs.frame, "cpu"), live.bounds)
    # the companion's (loop.Brain on the bridge's document), the actor's input caught
    seen = {}
    real = net.act

    def spy(o, *a, **k):
        seen["o"] = o
        return real(o, *a, **k)
    monkeypatch.setattr(net, "act", spy)
    brain = loop.Brain(net, greedy=True)
    brain.decide(_game_doc(st, ours, 1))
    o_game = seen["o"]
    names = brain.battle.names
    idx = [int(n.split("_")[1]) for n in names]                   # the document's units -> the simulator's slots
    for k in ("own", "attend", "ctrl", "free"):
        assert torch.equal(o_game[k][0], o_sim[k][0, idx]), k
    assert torch.allclose(o_game["pos"][0], o_sim["pos"][0, idx], atol=1e-4)
    assert torch.equal(o_game["target_ok"][0], o_sim["target_ok"][0, idx])
    cols = [ob.INDEX[n] for n in ("hp", "visible", "seen", "state_routing", "state_shattered", "melee", "threat_left",
                                  "threat_right", "threat_rear")] + [sectors.COST]
    assert torch.allclose(o_game["tokens"][0][:, cols], o_sim["tokens"][0, idx][:, cols], atol=1e-5)
    assert torch.allclose(o_game["geo"], o_sim["geo"], atol=1e-4)
    f_game = sectors.features(CFG, o_game, sectors.inside(CFG, o_game["geo"]))
    f_sim = sectors.features(CFG, o_sim, sectors.inside(CFG, o_sim["geo"]))
    assert torch.allclose(f_game, f_sim, atol=1e-5) and float(f_sim[..., 0].sum()) > 0
    # the frame: forward points from our army to the enemy's whichever side we are
    fwd_enemy = o_sim["pos"][0][~o_sim["own"][0] & o_sim["attend"][0], 0].mean()
    fwd_own = o_sim["pos"][0][o_sim["own"][0] & o_sim["attend"][0], 0].mean()
    assert fwd_enemy > fwd_own


def test_a_flank_threat_ends_a_commitment_only_once_it_has_lasted_2_s():
    setup, state, obs = calm()
    o = policy.to_torch(obs)
    B, N = o["own"].shape
    on = _obs_with(obs, threat_left=(0, 1.0))
    st = settled(commit.start(B, N), o)
    st["until"][:, 0] = 100.0
    a = heads.Action(torch.full((B, N), KEEP), torch.zeros((B, N), dtype=torch.long), torch.full((B, N), -1),
                     torch.zeros((B, N), dtype=torch.bool), None, torch.zeros((B, N), dtype=torch.long))
    free = []
    # flickers (on 1 s, off, on 1 s): never 2 s on - held throughout; then on for good: freed once, at 2 s
    for t, obs_now in enumerate([on, o, on, o, on, on, on, on, on]):
        x = commit.inputs(st, obs_now, torch.full((B,), float(t)))
        free.append(bool(x["free"][0, 0]))
        st = commit.apply(st, x, torch.full((B,), float(t)), a)
        st["until"][:, 0] = 100.0                                   # (kept committed, to see every event)
    assert free == [False, False, False, False, False, False, True, False, False]


def test_the_eyes_targets_and_what_they_feed_back_are_bounded_and_finite():
    T, R, N = 3, 1, 2
    before = {"hp": torch.ones(T, R, N), "gold": torch.zeros(T, R, N)}
    after = {"hp": torch.tensor([[[0.5, float("nan")]]] * T), "gold": torch.full((T, R, N), 1e6)}
    tg = eyes.targets(before, after, torch.zeros(T, R, dtype=torch.bool), torch.zeros(T, R), 1.0)
    assert torch.isfinite(tg["own"]).all() and torch.isfinite(tg["threat"]).all()
    assert float(tg["threat"].max()) == eyes.THREAT_CAP * eyes.THREAT_SCALE           # cost 0 and a huge damage
    assert float(tg["own"][0, 0, 1, 0]) == 0.0                                        # a NaN health: 0
    # an extreme prediction (the threat head's last layer pushed to +1e6) changes the tokens no more than the cap
    setup, state, obs = setup_obs()
    o = v2_obs(obs, setup)
    net = model()
    x = torch.randn(2, 1 + o["own"].shape[1], CFG.d)
    s = torch.randn(2, CFG.sectors ** 2, CFG.sector_d)
    with torch.no_grad():
        net.eyes.threat[-1].bias.fill_(1e6)
        net.eyes.danger[-1].bias.fill_(float("nan"))
        x2, s2, seen = net.eyes(x, s, o)
    assert float(seen["eyes_threat"].max()) > 1e5
    assert torch.isfinite(x2).all() and torch.isfinite(s2).all()
    w = net.eyes.threat_in.weight.abs().sum() * eyes.THREAT_CAP * eyes.THREAT_SCALE + net.eyes.threat_in.bias.abs().sum()
    assert float((x2 - x).abs().max()) <= float(w + net.eyes.own_in.weight.abs().sum() + net.eyes.own_in.bias.abs().sum())


def test_a_non_finite_gradient_skips_the_step():
    import dataclasses
    actor, crit = tiny()
    env = rollout.Battles(league.layout(4, 1, {"self": 0.5, "nearest": 0.5}), MIRROR)
    batch = rollout.collect(env, actor, crit, 2)
    batch["value"] = batch["value"] * float("nan")              # a NaN return: every gradient NaN
    before = [p.detach().clone() for p in actor.parameters()]
    opt = torch.optim.Adam(list(actor.parameters()) + list(crit.parameters()), lr=1e-3)
    st = ppo.update(actor, crit, opt, batch, dataclasses.replace(ppo.PPOConfig(), epochs=1, minibatch=4))
    assert st["skipped"] >= 1
    assert all(torch.equal(a, b) for a, b in zip(before, actor.parameters()))


def test_the_gumbel_sampler_draws_the_softmax_never_a_masked_choice_and_survives_bad_logits():
    torch.manual_seed(0)
    logits = torch.tensor([[1.0, 0.0, -1.0, heads.NEG, 2.0]]).expand(200000, -1)
    picks = chain._pick(logits, False, 1.0)
    freq = torch.bincount(picks, minlength=5).float() / len(picks)
    assert torch.allclose(freq, torch.softmax(logits[0], -1), atol=0.005) and int(freq[3] * len(picks)) == 0
    hot = chain._pick(logits[:20000], False, 0.5)                 # the temperature sharpens it as softmax(x / T)
    f2 = torch.bincount(hot, minlength=5).float() / len(hot)
    assert torch.allclose(f2, torch.softmax(logits[0] / 0.5, -1), atol=0.01)
    bad = torch.tensor([[float("nan")] * 4, [float("inf"), 0, 0, 0], [3e38, -3e38, heads.NEG, 0],
                        [heads.NEG] * 4, [float("-inf"), 1.0, float("nan"), 0.0]])
    for _ in range(50):
        a = chain._pick(bad, False, 1.0)
        assert ((a >= 0) & (a < 4)).all() and a[1] == 0 and a[2] == 0 and a[4] in (1, 3)
    assert chain._pick(bad, True, 1.0)[2] == 0


def test_the_heads_logits_are_soft_capped_and_a_non_finite_one_is_counted_and_masked():
    setup, state, obs = setup_obs()
    o = v2_obs(obs, setup)
    net = model()
    with torch.no_grad():
        net.heads.place_near.bias.fill_(1e4)                        # the near term blown up: sector logits ~ -1e6
        net.heads.commit.weight.fill_(float("nan"))                 # a head gone NaN
        logits, a, _ = net.act(o)
    for k in ("kind", "target", "sector", "fine", "run", "ability"):
        v = logits[k]
        allowed = v > heads.NEG / 2
        assert float(v[allowed].abs().max()) <= chain.CAP + 1e-4, k
    assert float(logits["nan_fixed"].sum()) > 0                     # the NaN commit logits counted ...
    assert ((a.commit >= 0) & (a.commit < len(commit.DURATIONS))).all()        # ... and a valid choice made
    lp, _ = heads.log_prob(logits, a, o["ctrl"])
    assert torch.isfinite(lp).all()

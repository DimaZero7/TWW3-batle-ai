"""tools/nn/general/runtime.py, units.py, tiles.py, learn.py: the general in the battles (S28 skeleton) - the deployment
stays in the zone, the plan is locked between re-plans, re-plans come on their events, the task input, the update's
cards are the rollout's (docs/en/training/general.md "Checks"). torch: the training container, on the CPU."""
import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn.general import army_input as ai                  # noqa: E402
from tools.nn.general import config as gconfig                 # noqa: E402
from tools.nn.general import executor, learn, tiles, units     # noqa: E402
from tools.nn.model import config, critic as mcritic, heads as hd, policy  # noqa: E402
from tools.nn.train import league, ppo, randomise, rollout, scenes  # noqa: E402

TINY = config.preset("v2", d=32, layers=2, heads=2, pointer=16, critic_d=32, critic_layers=1, critic_heads=2,
                     sector_d=16)
GCFG = gconfig.GeneralConfig(card=16, reader_hidden=16, council_heads=2)
MIRROR = [("arena", "attack")]


def nets(seed=0):
    torch.manual_seed(seed)
    return executor.wrap(policy.Actor(TINY).eval(), GCFG).eval(), mcritic.Critic(TINY).eval()


def battles(B=4, mix=None, seed=1):
    params = rollout.params_with_limit(600.0)
    src = scenes.Generated(range(500, 500 + 2 * B), 5, params, "cpu", seed=seed)
    lay = league.layout(B, 1, mix or {"self": 0.5, "nearest": 0.5}, scene_attacker=[1])
    return rollout.Battles(lay, MIRROR, params=params, source=src, spread=randomise.NONE, compile=False)


def test_the_unit_table_is_the_army_input():
    keys = ["wh_main_emp_inf_spearmen_0", "wh_main_emp_cha_general_0", "wh2_main_skv_inf_skavenslaves_0", ""]
    t = units.UnitTable()
    num, lab, full = t.battle([keys])
    a = ai.build({1: keys[:2], 2: keys[2:3]}, 1)
    own = torch.tensor([[True, True, False, False]])
    u = {"men": full[..., 0], "hp": torch.ones(1, 4), "hp0": full[..., 1]}
    got = units.dynamic(num, full, u, own)
    torch.testing.assert_close(got[0, :2], torch.as_tensor(a.numbers[:2]))
    torch.testing.assert_close(got[0, 2], torch.as_tensor(a.numbers[40]))
    assert torch.equal(lab[0, :2], torch.as_tensor(a.labels[:2])) and torch.all(got[0, 3] == 0)
    half = units.dynamic(num, full, dict(u, men=full[..., 0] / 2, hp=torch.full((1, 4), 0.25)), own)
    assert float(half[0, 0, units.MEN_COME]) == pytest.approx(0.5) and float(half[0, 0, units.HP_COME]) == pytest.approx(0.25)


def test_the_tiles_of_the_state_are_the_sectors_of_the_observation():
    env = battles()
    actor, _ = nets()
    a, frame, _ = env.observe(False)
    rows = torch.arange(2 * env.B)
    b, s = rows % env.B, rows // env.B + 1
    u = {k: env.st.u[k][b] for k in ("x", "z", "side", "men", "hp", "ms", "gone")}
    from tools.nn.model import sectors as ms
    fr = rollout.frame_rows(frame, rows)
    cells = ms.inside(TINY, ms.geo(fr, env.bounds2[rows]))
    st_t = tiles.from_state(TINY, u, env.st.u["cost"][b], s, fr, cells)
    ob_t = tiles.from_obs(TINY, {k: a[k][rows] for k in ("tokens", "own", "attend", "pos")}, cells)
    torch.testing.assert_close(st_t, ob_t, rtol=1e-4, atol=1e-4)
    f2 = tiles.frame_at(u, s, env.bounds2[rows])
    for k in ("cx", "cz", "ux", "uz"):
        torch.testing.assert_close(getattr(f2, k), getattr(fr, k).float(), rtol=0, atol=1e-5)


def test_the_units_places_of_the_state_are_the_observations():
    """The general's unit places (net.POS_FEATURES) at the deployment (from the state) are what the side's
    observation's tokens show at the battle's start (where it stands, its facing; no velocity yet)."""
    from tools.nn.general.runtime import Runtime
    env = battles()
    a, frame, _ = env.observe(False)
    rows = torch.arange(2 * env.B)
    b = rows % env.B
    u = {k: env.st.u[k][b] for k in ("x", "z", "b", "side")}
    fr = rollout.frame_rows(frame, rows)
    o = {k: a[k][rows] for k in ("tokens", "own", "attend", "pos")}
    present = u["side"] > 0
    got = Runtime.pos_of_state(u, fr, present & o["attend"])
    want = Runtime.pos_of_obs(o)
    known = want[..., -1] == 1                                        # (own units and the enemies seen)
    assert float(known.float().sum()) > 0.9 * float((present & o["attend"]).float().sum())
    torch.testing.assert_close(got[known], want[known], rtol=1e-4, atol=1e-4)
    assert float(want[..., :2].abs().sum()) > 0


def test_the_deployment_is_the_generals_and_stays_in_the_zone():
    env = battles(B=6)
    actor, crit = nets()
    before = {k: env.st.u[k].clone() for k in ("x", "z")}
    env.step(actor, crit)
    rt = env.general_rt
    r = rt.r
    ctrl = torch.cat([env.ctrl[:, 0], env.ctrl[:, 1]])
    learner = ctrl == league.LEARNER
    assert torch.all(r["has"] == learner) and torch.all(r["owner"][learner] == 0) and torch.all(r["owner"][~learner] == -1)
    dep = r["in/deploy"]
    assert bool(dep.any())
    rule = tiles.ZoneRule.of_arena()
    moved = False
    for row in torch.nonzero(dep).flatten().tolist():
        b, s = row % env.B, row // env.B + 1
        own = env.st.u["side"][b] == s
        # one second of the battle has passed: a few metres at most beyond the zone
        assert bool(rule.inside(env.st.u["x"][b][own], env.st.u["z"][b][own], torch.full_like(env.st.u["x"][b][own], s),
                                slack=10.0).all())
        moved |= bool((env.st.u["x"][b][own] - before["x"][b][own]).abs().max() > 20)
    assert moved                                                      # the general's places, not place.py's
    # the scripted side keeps place.py's placement
    for row in torch.nonzero(~learner).flatten().tolist():
        b, s = row % env.B, row // env.B + 1
        own = env.st.u["side"][b] == s
        assert float((env.st.u["x"][b][own] - before["x"][b][own]).abs().max()) < 10
    st = rt.take_stats()
    assert st["deploy_plans"] == int(learner.sum()) and st["laid_units"] > 0


def test_the_plan_is_locked_between_replans_and_replans_on_its_events():
    env = battles(B=4, mix={"self": 1.0})
    actor, crit = nets()
    env.step(actor, crit)
    rt = env.general_rt
    r = rt.r
    keep = {k: r[k].clone() for k in ("main", "kind", "target")}
    for _ in range(3):
        env.step(actor, crit)
    for k, v in keep.items():
        assert torch.equal(r[k], v), k                               # nothing due: locked
    assert torch.all(r["n_replan"] == 0)
    # the timer: 30 s since the plan
    r["t_plan"][0] -= 31.0
    env.step(actor, crit)
    assert int(r["n_replan"][0]) == 1 and int(r["n_replan"][1:].sum()) == 0
    assert rt.stats["replan_timer"] == 1
    # (the re-plan's numbering is counted: our units in a group in both plans, those that changed number)
    assert rt.stats["churn_units"] > 0 and 0 <= rt.stats["churn_moved"] <= rt.stats["churn_units"]
    assert 0 <= rt.stats["kind_switched"] <= rt.stats["kind_kept_groups"]
    # a broken group: more than half of its units rout - but not sooner than replan_min_s after the plan (row 0's is
    # a second old)
    row = 0
    b, s = row % env.B, row // env.B + 1
    g = int(r["main"][row][r["main"][row] >= 0][0])
    members = torch.nonzero((r["main"][row] == g) & (env.st.u["side"][b] == s)).flatten()
    env.st.u["ms"][b, members] = 6.0
    n0 = rt.stats["replan_broken"]
    env.step(actor, crit)
    assert rt.stats["replan_broken"] == n0                           # 1-2 s after the plan: not yet
    r["t_plan"][row] -= 5.0
    env.st.u["ms"][b, members] = 6.0
    env.step(actor, crit)
    assert rt.stats["replan_broken"] == n0 + 1
    # a lord fallen (ours or theirs)
    row = 2
    b, s = row % env.B, row // env.B + 1
    r["t_plan"][row] -= 5.0
    n0 = rt.stats["replan_lord"]
    r["t_plan"][row + env.B] -= 5.0
    env.st.lord_dead_s[b, 2 - s] = 0.5
    env.step(actor, crit)
    assert rt.stats["replan_lord"] == n0 + 2                         # both sides: their lord, our lord


def test_the_task_is_for_own_units_in_a_group_only():
    env = battles(B=4)
    actor, crit = nets()
    s = env.step(actor, crit)
    o = s["obs"]
    on, geo = o["task_on"], o["task_geo"]
    assert "task" not in o                                            # the cards are not stored
    assert torch.all(~on | (o["own"] & o["ctrl"]))
    assert torch.all(geo[~on] == 0) and bool(on.any())
    assert geo.shape[-1] == 4                                         # one target: cos, sin, distance, none
    cs = geo[..., 0:2].norm(dim=-1)                                   # (cos, sin) a unit vector, or the "none" flag
    none = geo[..., 3] > 0.5
    assert torch.allclose(cs[on & ~none], torch.ones_like(cs[on & ~none]), atol=1e-5)
    assert torch.all(geo[..., 2][on & ~none] > 0)
    assert torch.all(o["task_plan"][on.any(-1)] >= -1)


def test_the_task_points_at_the_target_from_the_units_facing():
    env = battles(B=2, mix={"self": 1.0})
    actor, crit = nets()
    env.step(actor, crit)
    rt = env.general_rt
    a, frame, _ = env.cur
    row = 0
    rows = torch.tensor([row])
    G = GCFG.places
    g = int(rt.r["main"][row][rt.r["main"][row] >= 0][0])
    # the group's target: a tile 100 m ahead of the unit in the frame
    obs_r = rollout.rows_of(a, rows)
    fr = rollout.frame_rows(frame, rows)
    n = int(torch.nonzero(rt.r["main"][row] == g)[0])
    from tools.nn.model import observation as ob
    pos = obs_r["pos"][0, n] * ob.POS
    x, z = fr.world(pos[0:1] + 100.0, pos[1:2])
    rt.r["target"][row, g] = 2 * G                                    # coarse tile 0 ...
    rt.r["tgt"][row, g] = torch.stack([x[0], z[0]])                   # ... at that point
    out = rt.task(rows, dict(obs_r, ctrl=torch.ones_like(obs_r["ctrl"])), fr)
    tok = obs_r["tokens"][0, n]
    fc, fs = float(tok[ob.INDEX["face_cos"]]), float(tok[ob.INDEX["face_sin"]])
    geo = out["task_geo"][0, n]
    assert float(geo[0]) == pytest.approx(fc, abs=1e-4) and float(geo[1]) == pytest.approx(-fs, abs=1e-4)
    assert float(geo[2]) == pytest.approx(1.0, abs=1e-4) and float(geo[3]) == 0.0
    # a group target: its units' centre now
    e = int(torch.nonzero(rt.r["active"][row, G:]).flatten()[0]) + G
    rt.r["target"][row, g] = e
    out = rt.task(rows, dict(obs_r, ctrl=torch.ones_like(obs_r["ctrl"])), fr)
    seen = obs_r["tokens"][0][:, ob.INDEX["seen"]] > 0.5
    w = (rt.r["main"][row] == e) & obs_r["attend"][0] & (obs_r["own"][0] | seen)
    assert bool(w.any())
    cent = (obs_r["pos"][0][w] * ob.POS).mean(0)
    d = cent - pos
    assert float(out["task_geo"][0, n, 2]) == pytest.approx(float(d.norm()) / 100.0, abs=1e-3)


def test_the_updates_cards_are_the_rollouts_and_a_tiny_update_trains_the_general():
    env = battles(B=4, mix={"self": 0.5, "nearest": 0.5})
    actor, crit = nets()
    env.step(actor, crit)
    env.general_rt.r["t_plan"] -= 28.0                                # re-plans inside the chunk too
    batch = rollout.collect(env, actor, crit, 4)
    g = batch["general"]
    assert g["table"] is not None and batch["obs"]["task_plan"].shape == (4, env.R)
    with torch.no_grad():
        obs = rollout.full_obs(batch["obs"], batch["abil_static"])
        obs = ppo.with_task(actor, batch, obs)
        logits, _ = actor.sequence(obs, batch["h0"], batch["reset"], batch["action"])
        lp, _ = hd.log_prob(logits, batch["action"], obs["ctrl"])
    torch.testing.assert_close(lp, batch["lp"], atol=1e-4, rtol=0)
    assert g["ready"] is not None and g["ready"]["ret"].shape[0] > 0     # the re-planned plans ended
    before = {n: p.detach().clone() for n, p in actor.named_parameters()}
    opt = torch.optim.Adam(list(actor.parameters()) + list(crit.parameters()), lr=1e-3)
    with torch.no_grad():
        actor.task_in.weight.normal_(0, 0.1)                          # (at zero the executor's loss reaches no card)
    st = ppo.update(actor, crit, opt, batch, dataclasses.replace(ppo.PPOConfig(), minibatch=8),
                    general=learn.LearnConfig(epochs=1))
    assert all(np.isfinite(v) for v in st.values() if isinstance(v, float))
    assert st["general_plans"] > 0 and np.isfinite(st["general_value_mse"])
    for k in ("general_ent_kind", "general_ent_target", "general_ent_place", "general_kl_plan"):
        assert k in st and np.isfinite(st[k]), k
    assert 0 < st["general_ent_kind"] <= np.log(GCFG.kinds) + 1e-6
    changed = {n for n, p in actor.named_parameters() if not torch.equal(before[n], p)}
    for part in ("general.reader.", "general.council.", "general.groups.", "general.listen.", "general.kinds",
                 "general.target_q.", "general.pos_in.", "general.value.", "task_in."):
        assert any(n.startswith(part) for n in changed), part


def test_a_plan_keeps_the_log_probability_it_was_made_with():
    """begin_chunk computes the plans' cards again with the weights now (the executor's input), but their choices keep
    the log-probability they were chosen with: PPO's ratio for a plan made before an update is honest."""
    env = battles(B=4, mix={"self": 1.0})
    actor, crit = nets()
    env.step(actor, crit)
    rt = env.general_rt
    lp0, mix0 = rt.r["lp"].clone(), rt.r["mix"].clone()
    with torch.no_grad():
        for p in actor.general.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    rt.learner = actor
    rt.begin_chunk(env)
    assert torch.equal(rt.r["lp"], lp0)
    assert not torch.allclose(rt.r["mix"], mix0)
    t = rt.end_chunk()["table"]
    mine = (rt.r["has"] & (rt.r["owner"] == 0)).nonzero().squeeze(1)
    torch.testing.assert_close(t["lp"], lp0[mine], rtol=0, atol=0)


def test_the_logs_targets_and_numbering():
    env = battles(B=4, mix={"self": 1.0})
    actor, crit = nets()
    env.step(actor, crit)
    env.general_rt.r["t_plan"] -= 31.0
    env.step(actor, crit)
    st = env.general_rt.take_stats()
    assert st["target_own"] + st["target_enemy"] + st["target_tile"] == pytest.approx(1.0, abs=1e-3)
    assert st["target_group"] == pytest.approx(st["target_own"] + st["target_enemy"], abs=1e-3)
    assert 0 <= st["group_churn"] <= 1 and 0 <= st["kind_switch"] <= 1


def test_narrowing_and_a_branch_keep_the_plans():
    env = battles(B=4, mix={"self": 1.0})
    actor, crit = nets()
    env.step(actor, crit)
    rt = env.general_rt
    keep = torch.tensor([1, 3])
    mix = rt.r["mix"].clone()
    snap = rt.snap(keep)
    br = rt.branch(snap, 2)
    assert br.frozen and torch.equal(br.r["mix"], torch.cat([mix[keep], mix[keep + 4]]))
    env.narrow(keep)
    assert rt.B == 2 and torch.equal(rt.r["mix"], torch.cat([mix[keep], mix[keep + 4]]))
    env.step(actor, crit)                                             # goes on


def test_a_replan_passes_the_last_kinds_and_a_strong_inertia_keeps_them():
    """Runtime.inputs: a re-plan's prev_kind is the last plan's kind of each of our active group numbers (-1 else);
    with GeneralConfig.kind_stick 50 every group that kept some of its number's units keeps its kind (kind_switch)."""
    torch.manual_seed(0)
    actor = executor.wrap(policy.Actor(TINY).eval(), dataclasses.replace(GCFG, kind_stick=50.0)).eval()
    crit = mcritic.Critic(TINY).eval()
    env = battles(B=4, mix={"self": 1.0})
    env.step(actor, crit)
    rt = env.general_rt
    G = GCFG.places
    old_kind, old_main, old_act = rt.r["kind"].clone(), rt.r["main"].clone(), rt.r["active"][:, :G].clone()
    rt.r["t_plan"] -= 31.0
    env.step(actor, crit)                                             # every row re-plans
    rows = rt.r["has"] & (rt.r["owner"] >= 0)
    pk = rt.r["in/prev_kind"]
    assert torch.equal(pk[rows], torch.where(old_act, old_kind, torch.full_like(old_kind, -1))[rows])
    ar = torch.arange(G)
    kept = ((rt.r["main"][..., None] == ar) & (old_main[..., None] == ar)).any(1) & old_act & rows[:, None]
    assert bool(kept.any())
    assert torch.equal(rt.r["kind"][kept], old_kind[kept])


def test_a_decision_with_a_general_refuses_a_cuda_graph_before_any_read_back(monkeypatch):
    """evaluate.Graphed captures a decision as a CUDA graph: with a general the step refuses at its first call
    (Runtime.deploy_pending raises before any read-back), so the capture fails cleanly; outside a capture it plays."""
    from tools.nn.general import runtime
    env = battles(B=2)
    actor, crit = nets()
    env.step(actor, crit)
    monkeypatch.setattr(runtime, "capturing", lambda: True)
    with pytest.raises(RuntimeError, match="no CUDA graph"):
        env.step(actor, crit)
    monkeypatch.setattr(runtime, "capturing", lambda: False)
    env.step(actor, crit)

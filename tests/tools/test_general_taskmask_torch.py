"""tools/nn/general/taskmask.py: the mission cuts the executor's choices (docs/en/training/general.md "Task masks") -
each kind's mask on hand-made observations, the fall-back, the copy of the chained heads, the rollout's
log-probabilities reproduced by the update under the masks, the switch off to the bit; the general's "shoot" only for a
group with a shooter; the bonus's kind weights. torch: the training container, on the CPU."""
import math

import pytest

torch = pytest.importorskip("torch")

from tools.nn.general import config as gconfig, executor, learn, taskmask    # noqa: E402
from tools.nn.general import army_input as ai                                 # noqa: E402
from tools.nn.model import config, heads as hd, observation as ob, policy    # noqa: E402
from tools.nn.model import sectors as sc                                      # noqa: E402
from tools.nn.sim.orders import ATTACK, HOLD, KEEP, MOVE, WITHDRAW            # noqa: E402

TINY = config.preset("v2", d=32, layers=2, heads=2, pointer=16, critic_d=32, critic_layers=1, critic_heads=2,
                     sector_d=16)
K = taskmask.KIND
G = gconfig.GeneralConfig().places
N = 6
OWN = [True, True, True, False, False, False]
H = sc.step(TINY) * math.sqrt(0.5)
CENTRES = sc.centres(TINY)


def obs_of(pos, missions, shooter=(), attacking=()):
    """One row: units at pos [(forward, lateral) m] (0-2 ours, 3-5 theirs: e0 = 3, 4; e1 = 5), missions {unit: (kind,
    point (f, l) or None, reach, target)}; shooter / attacking: our units with missiles / an attack order in force."""
    tok = torch.zeros(1, N, ob.TOKEN)
    for i in shooter:
        tok[0, i, ob.INDEX["ammo"]] = 1.0
    for i in attacking:
        tok[0, i, ob.INDEX["order_attack"]] = 1.0
    own = torch.tensor([OWN])
    o = {"pos": torch.tensor([pos], dtype=torch.float32) / ob.POS, "tokens": tok, "own": own, "ctrl": own.clone(),
         "free": own.clone(), "target_ok": ~own}
    kind = torch.zeros(1, N, dtype=torch.long)
    point = torch.zeros(1, N, 2)
    reach = torch.zeros(1, N)
    target = torch.full((1, N), -1, dtype=torch.long)
    on = torch.zeros(1, N, dtype=torch.bool)
    for i, (k, p, r, t) in missions.items():
        kind[0, i], reach[0, i], target[0, i], on[0, i] = K[k], (r if p is not None else 0.0), t, True
        if p is not None:
            point[0, i] = torch.tensor(p, dtype=torch.float32)
    main = torch.tensor([[0, 0, 1, G, G, G + 1]])
    return dict(o, **taskmask.fields(kind, point, reach, target, main, on))


def run(o, sec_ok=None):
    sec_ok = torch.ones(1, CENTRES.shape[0], dtype=torch.bool) if sec_ok is None else sec_ok
    m = taskmask.masks(TINY, o, sec_ok)
    kind, tgt, sec, fb = taskmask.apply(m, o["ctrl"], o["free"], o["target_ok"], sec_ok)
    return m, kind[0], tgt[0], sec[0], fb[0]


def dist(c, p):
    return (c - torch.tensor(p, dtype=torch.float32)).norm(dim=-1)


POS = [(-300.0, 0.0), (-300.0, 50.0), (-500.0, 0.0), (0.0, 0.0), (0.0, 40.0), (100.0, 300.0)]


def test_the_copy_of_the_chained_heads_follows_its_source():
    assert taskmask.source_sha() == taskmask.SOURCE_SHA, "chain.ChainHeads.forward changed: copy it to taskmask.py"


def test_withdraw_closes_the_attack_and_the_sectors_towards_the_enemy():
    o = obs_of(POS, {0: ("withdraw", (-700.0, 0.0), 100.0, 2)}, attacking=[0])
    m, kind, tgt, sec, fb = run(o)
    assert not kind[0, ATTACK] and not kind[0, KEEP]                 # nor keep of the attack in force
    assert kind[0, HOLD] and kind[0, MOVE] and kind[0, WITHDRAW]
    assert kind[1, ATTACK] and kind[1, KEEP]                          # no mission: as the heads' own masks
    to_e = dist(CENTRES, (0.0, 0.0))
    assert bool((to_e[sec[0]] >= 300.0 - H - 1e-3).all())            # never nearer the enemy than now
    near = dist(CENTRES, (-150.0, 0.0)).argmin()                      # a sector half way to the enemy: closed
    back = dist(CENTRES, (-650.0, 0.0)).argmin()                      # behind: open
    assert not sec[0, near] and sec[0, back] and bool(sec[1].all())
    assert not bool(fb.any()) and bool(m["on"][0, 0])


def test_shoot_closes_the_melee_of_a_unit_without_missiles_and_aims_a_shooter_at_the_target_group():
    o = obs_of(POS, {0: ("shoot", None, 0.0, G), 2: ("shoot", None, 0.0, G)}, shooter=[2])
    _, kind, tgt, sec, _ = run(o)
    assert not kind[0, ATTACK] and kind[0, MOVE] and kind[0, HOLD]    # no missiles: no attack, moving free
    assert kind[2, ATTACK]
    assert tgt[2].tolist() == [False, False, False, True, True, False]   # only e0's units
    assert tgt[1].tolist() == [False, False, False, True, True, True]
    assert bool(sec[0].all()) and bool(sec[2].all())


def test_attack_aims_only_at_the_target_group_and_never_moves_away_from_it():
    o = obs_of(POS, {0: ("attack", (100.0, 300.0), 100.0, G + 1)})
    _, kind, tgt, sec, fb = run(o)
    assert tgt[0].tolist() == [False] * 5 + [True] and kind[0, ATTACK]
    d_now = float(dist(torch.tensor([POS[0]]), (100.0, 300.0))[0])
    assert bool((dist(CENTRES, (100.0, 300.0))[sec[0]] <= d_now + H + 1e-3).all())
    assert sec[0, dist(CENTRES, (100.0, 300.0)).argmin()]
    # the target group not seen: the targets left free (a fall-back)
    o["target_ok"][0, 5] = False
    _, kind, tgt, _, fb = run(o)
    assert tgt[0].tolist() == [False, False, False, True, True, False] and bool(fb[0])


def test_hold_keeps_the_unit_near_the_point_and_fights_only_near_it():
    o = obs_of(POS, {2: ("hold", (-500.0, 0.0), 100.0, 2 * G + 1)})
    _, kind, tgt, sec, _ = run(o)
    assert bool((dist(CENTRES, (-500.0, 0.0))[sec[2]] <= 100.0 + H + 1e-3).all()) and bool(sec[2].any())
    assert not bool(tgt[2].any()) and not kind[2, ATTACK]             # no enemy within 300 m of the point
    o = obs_of(POS, {2: ("hold", (-150.0, 0.0), 100.0, 2 * G + 1)})
    _, kind, tgt, _, _ = run(o)
    assert tgt[2].tolist() == [False, False, False, True, True, False] and kind[2, ATTACK]


def test_screen_keeps_the_unit_on_the_segment_to_the_enemy():
    # our g1 (unit 2) at (-500, 0): unit 0 screens it from its nearest enemy (unit 3 at (0, 0))
    o = obs_of(POS, {0: ("screen", (-500.0, 0.0), 100.0, 1)})
    _, _, _, sec, _ = run(o)
    c = CENTRES[sec[0]]
    assert bool((c[:, 1].abs() <= 50.0 + H + 1e-3).all())            # near the line z = 0
    assert bool((c[:, 0] >= -500.0 - 2 * H).all() & (c[:, 0] <= 2 * H).all())
    assert sec[0, dist(CENTRES, (-250.0, 0.0)).argmin()] and not sec[0, dist(CENTRES, (-250.0, 300.0)).argmin()]


def test_a_mask_that_closes_every_sector_falls_back():
    o = obs_of(POS, {2: ("hold", (-500.0, 0.0), 100.0, 2 * G + 1)})
    sec_ok = dist(CENTRES, (-500.0, 0.0))[None] > 400.0               # every sector near the point off the map
    m, _, _, sec, fb = run(o, sec_ok)
    assert bool(fb[2]) and torch.equal(sec[2], sec_ok[0])


def real():
    from tools.nn.train import league, rollout
    from tools.nn.model import critic as mcritic
    params = rollout.params_with_limit(600.0)
    from tools.nn.train import randomise, scenes
    src = scenes.Generated(range(500, 508), 5, params, "cpu", seed=1)
    lay = league.layout(4, 1, {"self": 1.0}, scene_attacker=[1])
    env = rollout.Battles(lay, [("arena", "attack")], params=params, source=src, spread=randomise.NONE, compile=False)
    torch.manual_seed(0)
    gcfg = gconfig.GeneralConfig(card=16, reader_hidden=16, council_heads=2, task_mask_parts="all")
    return env, executor.wrap(policy.Actor(TINY).eval(), gcfg).eval(), mcritic.Critic(TINY).eval()


def test_the_rollouts_masked_choices_are_the_updates_and_obey_the_masks():
    from tools.nn.train import ppo, rollout
    env, actor, crit = real()
    env.step(actor, crit)
    rt = env.general_rt
    G_ = rt.G
    rt.r["kind"][:, :] = K["withdraw"]                                # every group withdraws: no attack at all
    batch = rollout.collect(env, actor, crit, 3)
    o = batch["obs"]
    assert "task_mk" in o and bool((o["task_mk"] >= 0).any())
    a = batch["action"]
    under = (o["task_mk"].long() == K["withdraw"]) & o["free"] & o["ctrl"]
    assert bool(under.any()) and not bool(((a.kind == ATTACK) & under).any())
    with torch.no_grad():
        obs = rollout.full_obs(batch["obs"], batch["abil_static"])
        obs = ppo.with_task(actor, batch, obs)
        logits, _ = actor.sequence(obs, batch["h0"], batch["reset"], batch["action"])
        lp, _ = hd.log_prob(logits, batch["action"], obs["ctrl"])
    torch.testing.assert_close(lp, batch["lp"], atol=1e-4, rtol=0)
    st = rt.take_stats()
    assert 0 < st["mask_share"] <= 1 and "mask_fallback" in st and G_ > 0


def test_switched_off_the_task_carries_no_mask_and_the_executor_is_as_before():
    import dataclasses
    from tools.nn.train import rollout
    env, actor, crit = real()
    actor.set_rules(dataclasses.replace(actor.gcfg, task_mask=False))
    s = env.step(actor, crit)
    assert "task_mk" not in s["obs"]
    a, frame, _ = env.cur
    obs_r, *_ = env._act(actor, a, frame, env.rows_learn, None, True)
    plain = policy.Actor(TINY).eval()
    plain.load_state_dict(actor.state_dict(), strict=False)
    with torch.no_grad():
        la, aa, _ = plain.act({k: v for k, v in obs_r.items() if not k.startswith("task")}, None, greedy=True)
        lb, ab, _ = actor.act({k: v for k, v in obs_r.items() if k != "task"}, None, greedy=True)
    for k in la:
        assert torch.equal(la[k], lb[k]), k


def test_the_general_gives_shoot_only_to_a_group_with_a_shooter():
    from tools.nn.general.net import General
    gcfg = gconfig.GeneralConfig(card=16, reader_hidden=16, council_heads=2, task_mask_parts="all")
    gen = General(gcfg, TINY.sectors, TINY.grid_half)
    P, Nn = 1, 7
    num = torch.zeros(P, Nn, ai.SIZE)
    num[0, [1, 3], ai.INDEX["range"]] = 0.3                           # units 1 and 3 have a missile weapon
    num[0, [1, 3], ai.INDEX["ammo_come"]] = 1.0                       # ... and their ammunition
    inp = {"num": num, "on": torch.ones(P, Nn, dtype=torch.bool)}
    # g0: unit 0 (no shooter); g1: 1, 2 (one of two: half); g2: 3, 4, 5 (one of three); g3: empty
    c = {"main": torch.tensor([[0, 1, 1, 2, 2, 2, G]])}
    ok = gen.shoot_ok(inp, c)
    s = K["shoot"]
    assert not ok[0, 0, s] and ok[0, 1, s] and not ok[0, 2, s] and not ok[0, 3, s]
    spent = num.clone()
    spent[0, 1, ai.INDEX["ammo_come"]] = 0.0                          # unit 1 out of ammunition: no shooter
    assert not gen.shoot_ok(dict(inp, num=spent), c)[0, 1, s]
    assert bool(ok[0, :, K["attack"]].all())
    third = General(gconfig.GeneralConfig(card=16, reader_hidden=16, council_heads=2, shoot_share=0.3,
                                          task_mask_parts="shoot_group"), TINY.sectors, TINY.grid_half)
    assert bool(third.shoot_ok(inp, c)[0, 2, s])                     # one of three passes at 0.3
    off = General(gconfig.GeneralConfig(card=16, reader_hidden=16, council_heads=2, task_mask=False), TINY.sectors,
                  TINY.grid_half)
    assert bool(off.shoot_ok(inp, c).all())
    part_off = General(gconfig.GeneralConfig(card=16, reader_hidden=16, council_heads=2, task_mask_parts="attack_target"),
                       TINY.sectors, TINY.grid_half)
    assert bool(part_off.shoot_ok(inp, c).all())                     # the part shoot_group not in force


def test_the_parts_switch_each_mask_alone():
    assert taskmask.parse_parts("all") == frozenset(taskmask.PARTS)
    default = taskmask.parts_of(gconfig.GeneralConfig())                # the harmless ones (docs: the ablation)
    assert default == taskmask.parse_parts(gconfig.MASK_PARTS) and "attack_target" not in default
    assert not default & {"withdraw_attack", "shoot_melee", "shoot_target"} and "withdraw_move" in default
    assert taskmask.parse_parts("") == frozenset() == taskmask.parse_parts("none")
    with pytest.raises(ValueError):
        taskmask.parse_parts("withdraw")
    assert taskmask.parts_of(gconfig.GeneralConfig(task_mask=False, task_mask_parts="all")) == frozenset()
    o = obs_of(POS, {0: ("withdraw", (-700.0, 0.0), 100.0, 2), 2: ("hold", (-500.0, 0.0), 100.0, 2 * G + 1)},
               attacking=[0])
    sec_ok = torch.ones(1, CENTRES.shape[0], dtype=torch.bool)
    m = taskmask.masks(TINY, o, sec_ok, frozenset({"withdraw_move"}))
    assert bool(m["kind"][0, 0].all()) and not bool(m["sector"][0, 0].all())     # attack open, sectors cut
    assert bool(m["on"][0, 0]) and not bool(m["on"][0, 2])                      # hold: no part in force
    assert bool(m["sector"][0, 2].all()) and bool(m["target"][0, 2].all())
    m = taskmask.masks(TINY, o, sec_ok, frozenset({"withdraw_attack", "hold_target"}))
    assert not m["kind"][0, 0, ATTACK] and bool(m["sector"][0, 0].all())
    assert bool(m["sector"][0, 2].all()) and not bool(m["target"][0, 2].any())
    m = taskmask.masks(TINY, o, sec_ok, frozenset())
    assert not bool(m["on"].any()) and bool(m["kind"].all() and m["target"].all() and m["sector"].all())


def test_the_executor_heads_follow_the_rules_parts():
    import dataclasses
    actor = executor.wrap(policy.Actor(TINY).eval(), gconfig.GeneralConfig(card=16, reader_hidden=16, council_heads=2,
                                                                           task_mask_parts="hold_move,screen_move"))
    assert actor.heads.parts == frozenset({"hold_move", "screen_move"})
    actor.set_rules(dataclasses.replace(actor.gcfg, task_mask=False))
    assert actor.heads.parts == frozenset()


def test_the_bonus_kind_weights():
    assert learn.exec_kinds("withdraw=2,screen=2") == [1.0, 1.0, 1.0, 2.0, 1.0, 2.0]
    assert learn.exec_kinds("") == [1.0] * 6 and learn.exec_kinds("attack=0.5", 3) == [0.5, 1.0, 1.0]
    with pytest.raises(ValueError):
        learn.exec_kinds("charge=2")


def shooting(firing=True, volley_ready=1.0, enemy_at=None):
    """Our unit 2 (a shooter: reload 10 s, aim 3.3 s, silence after a new target 8.5 s) under "shoot" at e0, firing
    (the token's flag) or not; volley_ready its VOLLEY column; enemy_at: unit 3 moved there (a near enemy)."""
    pos = list(POS)
    if enemy_at is not None:
        pos[3] = enemy_at
    o = obs_of(pos, {2: ("shoot", None, 0.0, G)}, shooter=[2])
    tok = o["tokens"]
    tok[0, 2, taskmask.RELOAD_COL] = 10.0 / taskmask.RELOAD_SCALE
    tok[0, 2, ob.INDEX["aim_first"]], tok[0, 2, ob.INDEX["aim_retarget"]] = 0.33, 0.85
    tok[0, 2, ob.INDEX["firing"]] = float(firing)
    tok[0, 2, ob.INDEX["volley_ready"]] = volley_ready
    return o


def lock_at(o, t, t0):
    shoot = o["task_mk"].long() == K["shoot"]
    return taskmask.volley_lock(o, shoot, torch.tensor([float(t)]), t0)


def test_the_reload_column_is_the_passports_reload():
    from tools.nn.model import passport
    p = passport.load()
    key = next(k for k, v in p.items() if (v.get("missile") or {}).get("reload_s"))
    f = passport.features(p[key])
    assert abs(f[taskmask.RELOAD_COL - ob.INDEX["passport_0"]] * taskmask.RELOAD_SCALE
               - p[key]["missile"]["reload_s"]) < 1e-4


def test_the_volley_lock_holds_a_firing_shooter_for_its_cycle_then_frees_it():
    reload, longest = taskmask.cycle_s(shooting()["tokens"])
    N_ = float(longest[0, 2])
    assert float(reload[0, 2]) == pytest.approx(10.0) and N_ == pytest.approx(8.5 + 1.3 * 10.0 + 1.0, abs=1e-3)
    t0 = torch.full((1, N), -1.0)
    o = shooting()
    lock, t0 = lock_at(o, 10, t0)
    assert lock[0].tolist() == [False, False, True, False, False, False] and float(t0[0, 2]) == 10.0
    # under the lock: only keep (no move, no other target, no hold); the mask counts it
    m, kind, tgt, sec, fb = run(dict(o, task_lock=lock))
    assert kind[2].tolist() == [False, False, False, False, True] and bool(m["on"][0, 2]) and not bool(fb[2])
    assert bool(kind[1, MOVE])                                            # (the others as before)
    # no volley: locked N s, free at the first decision N s after it began, a new cycle from the next one
    t, seen = 10, []
    while t < 40:
        t += 1
        lock, t0 = lock_at(o, t, t0)
        seen.append((t, bool(lock[0, 2])))
    free = [t for t, k in seen if not k]
    assert free[0] == math.ceil(10 + N_) and all(k for t, k in seen if t < free[0])
    assert dict(seen)[free[0] + 1]
    _, kind, _, _, _ = run(dict(o, task_lock=torch.zeros_like(lock)))     # free: every kind open
    assert bool(kind[2, MOVE] and kind[2, ATTACK] and kind[2, HOLD])


def test_the_volley_lock_frees_the_decision_of_a_volley_and_locks_the_next_cycle():
    t0 = torch.full((1, N), -1.0)
    lock, t0 = lock_at(shooting(), 10, t0)
    lock, t0 = lock_at(shooting(volley_ready=1.0), 12, t0)
    assert bool(lock[0, 2])
    lock, t0 = lock_at(shooting(volley_ready=0.0), 13, t0)               # the ammo fell at this decision
    assert not bool(lock[0, 2]) and float(t0[0, 2]) == 13.0
    lock, t0 = lock_at(shooting(volley_ready=0.1), 14, t0)               # reloading: locked again
    assert bool(lock[0, 2])


def test_the_volley_lock_leaves_alone_a_shooter_not_firing_in_melee_or_near_an_enemy_and_other_missions():
    t0 = torch.full((1, N), -1.0)
    lock, t0b = lock_at(shooting(firing=False), 10, t0)                 # no target in range: not firing
    assert not bool(lock.any()) and float(t0b[0, 2]) == -1.0
    o = shooting()
    o["tokens"][0, 2, ob.INDEX["melee"]] = 1.0
    assert not bool(lock_at(o, 10, t0)[0].any())
    assert not bool(lock_at(shooting(enemy_at=(-470.0, 0.0)), 10, t0)[0].any())   # an enemy 30 m away
    o = shooting()
    o["tokens"][0, 2, ob.INDEX["ammo"]] = 0.0                              # out of ammunition
    assert not bool(lock_at(o, 10, t0)[0].any())
    o = shooting()
    lock, _ = taskmask.volley_lock(o, torch.zeros(1, N, dtype=torch.bool), torch.tensor([10.0]), t0)
    assert not bool(lock.any())                                           # not under "shoot"
    # a lock while the part is off: not applied
    lock, _ = lock_at(o, 10, t0)
    m = taskmask.masks(TINY, dict(o, task_lock=lock), torch.ones(1, CENTRES.shape[0], dtype=torch.bool),
                       frozenset({"shoot_group"}))
    assert bool(m["kind"][0, 2].all()) and not bool(m["on"][0, 2])


def test_the_runtimes_volley_clock_gives_the_same_lock_at_the_same_time_and_restarts_with_the_battle():
    from types import SimpleNamespace
    from tools.nn.general import runtime as gruntime
    rt = SimpleNamespace(dev=torch.device("cpu"), r={"vl_t0": torch.full((2, N), -1.0), "vl_at": torch.full((2,), -1.0),
                                                     "vl_lock": torch.zeros(2, N, dtype=torch.bool)})
    o = shooting()
    shoot = o["task_mk"].long() == K["shoot"]
    rows = torch.tensor([1])
    vol = (lambda o_, t: gruntime.Runtime._volley(rt, rows, o_, shoot, torch.tensor([float(t)])))
    assert bool(vol(o, 10)[0, 2])
    assert not bool(vol(shooting(volley_ready=0.0), 13)[0, 2])
    assert not bool(vol(shooting(volley_ready=0.0), 13)[0, 2])            # again at 13: the same (not a new cycle)
    assert bool(vol(shooting(volley_ready=0.1), 14)[0, 2])
    vol(o, 2)                                                             # the time went back: a new battle
    assert float(rt.r["vl_t0"][1, 2]) == 2.0 and float(rt.r["vl_t0"][0, 2]) == -1.0

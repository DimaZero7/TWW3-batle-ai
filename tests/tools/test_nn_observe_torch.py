"""Learning by observation (tools/nn/observe): the grid cell of a point, the imitation term, the demonstrations' store
and the PPO update with them (torch: run in the training container)."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn.model import config, critic as mcritic, decide, heads, policy, sectors, sources  # noqa: E402
from tools.nn.model import observation as ob  # noqa: E402
from tools.nn.observe import labels as lab, loss as obs_loss, store  # noqa: E402
from tools.nn.sim.orders import ATTACK, HOLD, MOVE, WITHDRAW  # noqa: E402

TINY = config.preset("v2", d=32, layers=2, heads=2, pointer=16, critic_d=32, critic_layers=1, critic_heads=2,
                     sector_d=16)


def tiny(seed=0):
    torch.manual_seed(seed)
    return policy.Actor(TINY).eval(), mcritic.Critic(TINY).eval()


def side_obs(T=8, own=4, enemy=5):
    """T observations of side 1 of a synthetic battle (B = 1), as convert.py stores them, and the frame and geo."""
    setup, state = sources.synthetic(batch=1, own=own, enemy=enemy, seed=0)
    mem, rows = None, []
    for k in range(T):
        st = dict(state, t=np.array([float(k)]))
        o, mem = ob.observe(st, setup, 1, mem)
        rows.append(o)
    cat = lambda k: np.concatenate([np.asarray(getattr(o, k)) for o in rows], 0)
    arrays = {k: cat(k) for k in store.OBS if getattr(rows[0], k) is not None}
    geo = sectors.geo(rows[0].frame, torch.as_tensor(setup.bounds, dtype=torch.float32))[0].numpy()
    return arrays, rows[0].frame, geo


def test_a_point_goes_to_the_nearest_grid_cell_inside_the_map():
    cfg = config.preset("v2")
    _, frame, geo = side_obs(2)
    cells = lab.grid_cells(cfg, geo)
    rng = np.random.default_rng(0)
    x, z = rng.uniform(-750, 750, 400), rng.uniform(-782, 718, 400)       # 18 m inside the map
    f, l = frame.point(x[None], z[None])
    on = (np.abs(f[0]) < 800) & (np.abs(l[0]) < 800)                       # on the side's grid (+-800 m)
    x, z = x[on], z[on]
    idx = lab.cell_of(frame, x, z, cells)
    centre = dict(zip(cells[0].tolist(), cells[1]))
    f, l = frame.point(x[None], z[None])
    d = np.array([np.hypot(*(centre[int(i)] - np.array([f[0, j], l[0, j]]))) for j, i in enumerate(idx)])
    assert d.max() <= 25 / np.sqrt(2) + 1e-3
    # the cell's centre in the world, as the network's order would put it (heads.point_world)
    a = heads.Action(torch.zeros(1, 1, dtype=torch.long), torch.as_tensor(idx[:1]).reshape(1, 1),
                     torch.full((1, 1), -1), torch.zeros(1, 1, dtype=torch.bool))
    pos = torch.zeros(1, 1, 2)
    wx, wz = heads.point_world(cfg, a, {"pos": pos}, decide.frame_to(frame, "cpu"),
                               torch.as_tensor([[-768.0, 768.0, -800.0, 736.0]]))[0, 0]
    assert np.hypot(float(wx) - x[0], float(wz) - z[0]) <= 25 / np.sqrt(2) + 1e-2


def demo_arrays(T=8):
    arrays, frame, geo = side_obs(T)
    N = arrays["own"].shape[1]
    own = np.nonzero(arrays["own"][0])[0]
    enemy = np.nonzero(arrays["target_ok"][0])[0]
    kind = np.full((T, N), -1, np.int8)
    target = np.full((T, N), -1, np.int16)
    point = np.full((T, N), -1, np.int32)
    run, known, alt = (np.zeros((T, N), bool) for _ in range(3))
    cells = lab.grid_cells(TINY, geo)
    kind[1, own[0]], target[1, own[0]] = ATTACK, enemy[0]
    kind[2, own[1]], point[2, own[1]] = MOVE, int(lab.cell_of(frame, [0.0], [0.0], cells)[0])
    run[2, own[1]] = known[2, own[1]] = alt[2, own[1]] = True
    kind[3, own[2]] = HOLD
    kind[5, own[0]], target[5, own[0]] = ATTACK, enemy[1]
    a = np.array([0.0, 0.5, 0.2, -0.3, 0.0, 1.0, 0.0, 0.0], np.float32)[:T]
    arrays.update(kind=kind, target=target, point=point, run=run, run_known=known, alt=alt, geo=geo,
                  t=np.arange(T, dtype=np.float32), adv_critic=a, adv_mc=a, w20_adv=-a, w20_trade=np.ones(T, np.float32))
    return arrays


def write_demo(path, T=8):
    arrays = demo_arrays(T)
    meta = {"run": path.stem, "sides": [1], "cfg": {"sectors": TINY.sectors, "fine": TINY.fine,
                                                    "grid_half": TINY.grid_half}}
    np.savez_compressed(path, meta=np.asarray(json.dumps(meta)), **{f"s1_{k}": v for k, v in arrays.items()})


def test_the_store_selects_by_the_advantage_and_samples_chunks_with_their_memory(tmp_path):
    write_demo(tmp_path / "b1.npz")
    small = store.Demos(tmp_path, T=4)
    # a battle with fewer units than the store's largest: its padded units carry no label
    arrays = demo_arrays()
    meta = {"run": "b0", "sides": [1], "cfg": {"sectors": TINY.sectors, "fine": TINY.fine, "grid_half": TINY.grid_half}}
    np.savez_compressed(tmp_path / "b0.npz", meta=np.asarray(json.dumps(meta)),
                        **{f"s1_{k}": (v[:, :-2] if v.ndim >= 2 and v.shape[1] == small.N else v)
                           for k, v in arrays.items()})
    pad = store.Demos(tmp_path, T=4)
    q = next(q for q in pad.seqs if q["name"] == "b0/s1")
    assert q["obs"]["tokens"].shape[1] == pad.N and bool((q["lab"]["kind"][:, -2:] == -1).all())
    assert float(q["c"][:, -2:].abs().sum()) == 0 and pad.info["labels"] == 8
    (tmp_path / "b0.npz").unlink()
    actor, _ = tiny()
    d = store.Demos(tmp_path, T=4, adv="critic", cfg=actor.cfg)
    assert d.info["labels"] == 4 and d.info["selected"] == 3                   # the hold's A -0.3: not selected
    assert d.info["agree_w20"]["only_w20"] == 1 and d.info["agree_w20"]["both"] == 0
    assert len(d.chunks) == 2
    d.refresh(actor, "cpu")
    b = d.sample(3, "cpu", torch.Generator().manual_seed(0))
    assert b["obs"]["tokens"].shape[:2] == (4, 3) and b["h0"].shape == (3, 1 + d.N, TINY.d)
    assert torch.all((b["c"] > 0) <= (b["kind"] >= 0))
    w = store.Demos(tmp_path, T=4, adv="window20", cfg=actor.cfg)
    assert w.info["selected"] == 1                                             # only the hold (w20_adv 0.3)
    with pytest.raises(SystemExit):
        store.Demos(tmp_path, T=4, cfg=config.preset("v2", fine=2))


def test_the_term_takes_the_running_move_as_the_set_with_withdraw_and_trains_the_actor(tmp_path):
    write_demo(tmp_path / "b1.npz")
    actor, _ = tiny()
    d = store.Demos(tmp_path, T=4, cfg=actor.cfg)
    d.refresh(actor, "cpu")
    b = d.sample(4, "cpu", torch.Generator().manual_seed(1))
    actor.train()
    loss, st = obs_loss.term(actor, b)
    sm = obs_loss.summary(st)
    assert torch.isfinite(loss) and sm["labels"] > 0 and 0 <= sm["kind_acc"] <= 1 and sm["mean_c"] >= 1
    loss.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in actor.parameters())
    # the set: log P = log(P(move, place, run) + P(withdraw, place))
    T, B, N = b["kind"].shape
    f = lambda v: v.reshape(T * B, N)
    kind, point, run, known, alt = map(f, (b["kind"], b["point"], b["run"], b["run_known"], b["alt"]))
    i = torch.nonzero(alt)[0]
    k1 = torch.where(kind >= 0, kind, torch.full_like(kind, HOLD))
    tg = torch.full_like(kind, -1)
    c0 = torch.zeros_like(k1)
    acts = [heads.Action(k1, point.clamp(min=0), tg, run, None, c0),
            heads.Action(torch.where(alt, torch.full_like(k1, WITHDRAW), k1), point.clamp(min=0), tg, run, None, c0)]
    with torch.no_grad():
        lg, _ = obs_loss.logits_under(actor, b["obs"], b["h0"], acts)
        move = obs_loss.label_logp(actor.cfg, lg[0], k1, tg, point, run, known)[i[0], i[1]]
        wd = obs_loss.label_logp(actor.cfg, lg[1], torch.full_like(k1, WITHDRAW), tg, point, run,
                                 torch.zeros_like(known))[i[0], i[1]]
    assert float(move) < 0 and float(wd) < 0
    sel = (f(b["kind"]) >= 0) & (f(b["c"]) > 0)
    assert bool(sel[i[0], i[1]])


def test_a_tiny_training_step_with_the_demonstrations(tmp_path, monkeypatch):
    import dataclasses
    from tools.nn.model import eyes
    from tools.nn.train import league, ppo, rollout
    monkeypatch.setattr(eyes, "HORIZONS_S", (1.0, 2.0))
    write_demo(tmp_path / "b1.npz")
    actor, crit = tiny()
    start = tiny()[0]
    for p in start.parameters():
        p.requires_grad_(False)
    env = rollout.Battles(league.layout(4, 1, {"self": 0.5, "nearest": 0.5}), [("arena", "attack")])
    batch = rollout.collect(env, actor, crit, 3)
    d = store.Demos(tmp_path, T=4, cfg=actor.cfg)
    o = store.Observe(d, 2, weight=0.1, decay=0.5, anchor_net=start, anchor=0.2, anchor_decay=0.5, refresh=1, updates=1)
    assert o.weight == pytest.approx(0.05) and o.anchor == pytest.approx(0.1)
    opt = torch.optim.Adam(list(actor.parameters()) + list(crit.parameters()), lr=1e-3)
    cfg = dataclasses.replace(ppo.PPOConfig(), epochs=1, minibatch=6)
    st = ppo.update(actor, crit, opt, batch, cfg, observe=o)
    s = st["observe"]
    assert all(np.isfinite(v) for v in s.values()) and s["labels"] > 0 and s["anchor_kl"] >= 0
    assert o.updates == 2 and o.weight == pytest.approx(0.025)
    # off: the update has no observation stats
    assert "observe" not in ppo.update(actor, crit, opt, batch, cfg)

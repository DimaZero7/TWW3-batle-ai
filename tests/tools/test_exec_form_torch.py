"""The executor's formation heads (tools/nn/model/chain.py cfg.form, tools/nn/general/form.py; docs/ru/training/general.md
"Тонкий исполнитель"): an old network given them computes what it did (sampled and greedy, to the bit) with "keep" all
but always; the masks (melee, attack / keep, a fixed width) leave only keep; the choices -> the simulator's Orders.face /
Orders.width; the rollout's log-probabilities reproduced by the update; a facing / width change pays the order cost;
the line's real start (g48_9_grafted.pt) loads. torch: the training container, on the CPU."""
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tools.nn.general import executor, form                    # noqa: E402
from tools.nn.model import chain, decide, heads as hd, policy  # noqa: E402
from tools.nn.model import observation as ob                   # noqa: E402
from tools.nn.model.frame import Frame                         # noqa: E402
from tools.nn.sim import orders as O                           # noqa: E402
from tools.nn.train import checkpoint, league, reward, rollout  # noqa: E402

from tests.tools.test_general_executor_torch import GCFG, MIRROR, TINY  # noqa: E402

START = Path("build/nn-train/line_g48/g48_9_grafted.pt")


def battle(seed=0):
    """(old executor, the same with the formation's heads, the learner rows' input with "wlim", the env)."""
    torch.manual_seed(seed)
    old = executor.wrap(policy.Actor(TINY), GCFG).eval()
    new = form.with_form(old).eval()
    env = rollout.Battles(league.layout(2, 1, {"self": 1.0}), MIRROR)
    env.step(new)
    a, frame, _ = env.cur
    obs, *_ = env._act(new, a, frame, env.rows_learn, None, True)
    return old, new, obs, env


def test_an_old_network_given_the_heads_computes_what_it_did():
    old, new, obs, _ = battle()
    assert "wlim" in obs and new.cfg.form and not old.cfg.form
    for greedy in (True, False):
        with torch.no_grad():
            torch.manual_seed(7)
            la, aa, ha = old.act(obs, None, greedy=greedy)
            torch.manual_seed(7)
            lb, ab, hb = new.act(obs, None, greedy=greedy)
        for k in la:
            assert torch.equal(la[k], lb[k]), k
        assert torch.equal(ha, hb)
        for f in aa.parts():
            assert torch.equal(getattr(aa, f), getattr(ab, f)), f
        assert aa.face is None and ab.face is not None
    keep_f = torch.softmax(lb["face"], -1)[..., 0]
    keep_w = torch.softmax(lb["width"], -1)[..., 0]
    assert float(keep_f.min()) > 0.99 and float(keep_w.min()) > 0.99
    with torch.no_grad():
        _, ag, _ = new.act(obs, None, greedy=True)
    assert int(ag.face.abs().sum()) == 0 and int(ag.width.abs().sum()) == 0
    fr = decide.frame_to(Frame(*(torch.zeros(obs["own"].shape[0]) for _ in range(2)),
                               torch.zeros(obs["own"].shape[0]), torch.ones(obs["own"].shape[0])), "cpu")
    bounds = torch.tensor([[-1000.0, 1000.0, -1000.0, 1000.0]]).expand(obs["own"].shape[0], 4)
    o = decide.to_orders(new.cfg, ag, obs, fr, bounds)
    assert bool((o.face == -1).all()) and bool((o.width == -1).all())     # keep: the old orders to the bit


def test_a_checkpoint_saved_with_the_heads_loads_with_them_and_the_new_names_come_fresh(tmp_path):
    old, new, _, _ = battle()
    names = {n for n, _ in new.named_parameters()} - {n for n, _ in old.named_parameters()}
    assert names == {"heads.face.weight", "heads.face.bias", "heads.width.weight", "heads.width.bias"}
    assert float(new.heads.face.weight.detach().abs().sum()) == 0 and float(new.heads.face.bias[0].detach()) == chain.FORM_KEEP_BIAS
    back = checkpoint.load_policy(checkpoint.save(tmp_path / "f.pt", new, None, "v2"))
    assert back.cfg.form and torch.equal(back.heads.width.bias, new.heads.width.bias)
    assert form.with_form(back) is back


def test_melee_attack_keep_and_a_fixed_width_leave_only_keep():
    _, new, obs, _ = battle()
    R, N = obs["own"].shape
    mine = obs["ctrl"] & obs.get("free", obs["ctrl"])
    assert bool(mine.any())
    kind = torch.full((R, N), O.HOLD, dtype=torch.long)
    able = chain.form_able(obs, kind, obs["ctrl"], obs.get("free", obs["ctrl"]))
    assert torch.equal(able, mine & ~(obs["tokens"][..., ob.INDEX["melee"]] > 0.5))
    tok = obs["tokens"].clone()
    tok[..., ob.INDEX["melee"]] = 1.0
    melee = dict(obs, tokens=tok)
    assert not bool(chain.form_able(melee, kind, obs["ctrl"], obs["ctrl"]).any())
    for k in (O.ATTACK, O.KEEP):
        assert not bool(chain.form_able(obs, torch.full_like(kind, k), obs["ctrl"], obs["ctrl"]).any())
    # the heads' logits under a melee: only keep allowed
    with torch.no_grad():
        lg, a, _ = new.act(melee, None)
    assert bool((lg["face"][..., 1:] < hd.NEG / 2).all()) and bool((lg["width"][..., 1:] < hd.NEG / 2).all())
    assert int(a.face.abs().sum()) == 0 and int(a.width.abs().sum()) == 0
    fixed = dict(obs, wlim=torch.zeros_like(obs["wlim"]))
    assert not bool(chain.width_free(fixed, mine).any())
    assert not bool(chain.width_free({k: v for k, v in obs.items() if k != "wlim"}, mine).any())


def test_width_limits_follow_the_simulators_rule():
    _, _, obs, env = battle()
    w = obs["wlim"]
    u = env.st.u
    lone_or_fixed = (u["men0"] <= 1) | (u["wmin_k"] <= 0)
    rows = env.rows_learn % env.B
    assert bool((w[..., 1] >= w[..., 0]).all()) and bool((w[..., 1] <= 80.0 + 1e-4).all())
    assert bool((w[lone_or_fixed[rows]] == 0).all())
    some = ~lone_or_fixed[rows] & (u["men0"][rows] > 20)
    assert bool(some.any()) and bool((w[some][:, 1] > w[some][:, 0]).all())
    # the companion's way (the passports by key) gives the same numbers as the simulator's state
    keys = env.setup.keys[0] if hasattr(env.setup, "keys") else None
    if keys is not None:
        k0 = [k for k in keys if k]
        stat = form.limits_static(keys, env.params)
        assert stat.shape == (len(keys), 2) and len(k0) > 0


def test_the_choices_become_the_simulators_facing_and_width():
    B, N = 1, 3
    fr = Frame(torch.zeros(B), torch.zeros(B), torch.zeros(B), torch.ones(B))   # forward +z: bearing 0
    face = torch.tensor([[0, 1, 5]])
    assert decide.face_world(face, fr).tolist() == [[-1.0, 0.0, 90.0]]
    fr2 = Frame(torch.zeros(B), torch.zeros(B), torch.ones(B), torch.zeros(B))  # forward +x: bearing 90
    assert decide.face_world(torch.tensor([[9, 13, 1]]), fr2).tolist() == [[270.0, 0.0, 90.0]]
    wlim = torch.tensor([[[10.0, 70.0], [10.0, 70.0], [10.0, 70.0]]])
    assert decide.width_metres(torch.tensor([[0, 1, 7]]), wlim).tolist() == [[-1.0, 10.0, 70.0]]
    assert decide.width_metres(torch.tensor([[4, 4, 4]]), None).tolist() == [[-1.0, -1.0, -1.0]]


def test_the_rollouts_log_probability_is_the_updates():
    _, new, obs, _ = battle()
    with torch.no_grad():
        # a sharper start so the formation heads choose something besides keep
        new.heads.face.bias.zero_()
        new.heads.width.bias.zero_()
        torch.manual_seed(3)
        lg, a, _ = new.act(obs, None)
        lg2, _, _ = new.act(obs, None, action=a)
    assert bool((a.face > 0).any()) and bool((a.width > 0).any())
    for k in ("face", "width", "kind"):
        assert torch.allclose(lg[k], lg2[k]), k
    lp, ent = hd.log_prob(lg2, a, obs["ctrl"])
    base, _ = hd.log_prob(lg2, hd.Action(**{f: getattr(a, f) for f in a.parts() if f not in ("face", "width")}),
                          obs["ctrl"])
    pf = torch.log_softmax(lg2["face"], -1).gather(-1, a.face[..., None])[..., 0]
    pw = torch.log_softmax(lg2["width"], -1).gather(-1, a.width[..., None])[..., 0]
    want = torch.where(obs["ctrl"], base + pf + pw, torch.zeros_like(base))
    assert torch.allclose(lp, want, atol=1e-5)
    assert bool(torch.isfinite(ent).all())


def test_a_facing_or_width_change_pays_the_order_cost():
    B, N = 1, 2
    u = {"order_kind": torch.full((B, N), O.HOLD), "order_target": torch.full((B, N), -1),
         "ox": torch.zeros(B, N), "oz": torch.zeros(B, N), "oface": torch.full((B, N), -1.0),
         "ow": torch.full((B, N), -1.0), "men": torch.ones(B, N) * 50, "r": torch.zeros(B, N, dtype=torch.bool),
         "s": torch.zeros(B, N, dtype=torch.bool), "gone": torch.zeros(B, N, dtype=torch.bool),
         "side": torch.ones(B, N, dtype=torch.long)}
    o = O.hold(B, N)
    assert not bool(reward.order_changes(u, o).any())                    # the same hold, no formation: none
    o.face[0, 0] = 92.0
    o.width[0, 1] = 40.0
    assert reward.order_changes(u, o).tolist() == [[True, True]]
    u["oface"][0, 0] = float(O.snap_face(torch.tensor(92.0)))
    u["ow"][0, 1] = 40.4
    assert reward.order_changes(u, o).tolist() == [[False, False]]      # re-given: no change


@pytest.mark.skipif(not START.exists(), reason="the line's start point is not on this machine")
def test_the_line_g48_start_loads_and_plays_as_before_with_keep():
    data = checkpoint.read(START)
    old = checkpoint.load_policy(START).eval()
    assert not old.cfg.form and data.get("general")
    new = form.with_form(old).eval()
    torch.manual_seed(0)
    env = rollout.Battles(league.layout(2, 1, {"self": 1.0}), MIRROR)
    env.step(new)
    a, frame, _ = env.cur
    obs, *_ = env._act(new, a, frame, env.rows_learn, None, True)
    with torch.no_grad():
        torch.manual_seed(1)
        la, aa, _ = old.act(obs, None)
        torch.manual_seed(1)
        lb, ab, _ = new.act(obs, None)
    for k in la:
        assert torch.equal(la[k], lb[k]), k
    for f in aa.parts():
        assert torch.equal(getattr(aa, f), getattr(ab, f)), f
    assert math.isclose(float(torch.softmax(lb["face"], -1)[..., 0].min()), 1.0, abs_tol=0.01)


def test_the_companion_sends_the_facing_and_width_to_the_bridge():
    """The game's way (tools/nn/companion/loop.py on a bridge state document): keep -> the old lines; heads that choose
    -> 'face' / 'width' on the hold / move / withdraw lines, a width only for a unit whose width can change."""
    from tools.nn.companion import exchange, loop
    from tools.nn.companion import general as cg
    from tests.tools.test_companion_general_torch import doc, general_actor
    actor = form.with_form(general_actor()).eval()
    brain = loop.Brain(actor, greedy=False, general=cg.config(actor))
    out, _, _ = brain.decide(doc(0, 1))
    assert not any("face" in o or "width" in o for o in out)                 # keep all but always
    lim = brain.wlim(brain.battle)[0]
    names = list(brain.battle.names)
    assert lim[names.index("own_lord")].tolist() == [0.0, 0.0]               # a lone man: no width
    lo, hi = lim[names.index("own_spear_1")].tolist()
    assert 0 < lo < hi <= 80.0
    with torch.no_grad():
        actor.heads.face.bias.copy_(torch.tensor([-20.0] + [0.0] * chain.FACES))
        actor.heads.width.bias.copy_(torch.tensor([-20.0] + [0.0] * (chain.WIDTHS - 1)))
    brain = loop.Brain(actor, greedy=False, general=cg.config(actor))
    torch.manual_seed(0)
    out, _, _ = brain.decide(doc(0, 1))
    formed = [o for o in out if o["kind"] in ("hold", "move", "withdraw") and not o.get("out")]
    assert formed and all("face" in o and 0 <= o["face"] < 360 for o in formed)
    assert all(("width" in o) == (o["unit"] != "own_lord") for o in formed)
    assert all(lo - 1e-3 <= o["width"] <= hi + 1e-3 for o in formed if "width" in o)
    assert not any("face" in o for o in out if o["kind"] in ("attack", "keep"))
    back = exchange.parse_orders(exchange.orders_text("b-1", 1, out))["orders"]
    for o in formed:
        assert back[o["unit"]].get("face") == o["face"] and back[o["unit"]].get("width") == o.get("width")


def test_the_fighters_hook_takes_every_argument_of_the_functions_it_wraps():
    """(the g48 line runs --fighters: melee.strikes got `late` on 29.10 and the hook's copy did not - every battle of the
    lab failed at its first step)"""
    import inspect
    from tools.nn.fighters.hook import Hook
    from tools.nn.sim import geometry, magic, melee, missile, morale, scenario
    for name, fn in (("strikes", melee.strikes), ("contact_limit", geometry.contact_limit), ("volley", missile.volley),
                     ("casualty_windows", morale.casualty_windows), ("bombard", magic.bombard),
                     ("magic_strikes", magic.strikes), ("build", scenario.build)):
        mine = set(inspect.signature(getattr(Hook, name)).parameters) - {"self"}
        assert set(inspect.signature(fn).parameters) <= mine, name


def test_an_update_trains_the_heads_and_logs_their_keep_as_numbers():
    import dataclasses
    from tools.nn.model import critic as mcritic
    from tools.nn.train import ppo
    torch.manual_seed(0)
    actor = policy.Actor(dataclasses.replace(TINY, form=True)).eval()
    crit = mcritic.Critic(TINY).eval()
    with torch.no_grad():
        actor.heads.face.bias.zero_()           # (choices besides keep, so the heads have rows to learn from)
        actor.heads.width.bias.zero_()
    env = rollout.Battles(league.layout(6, 1, {"self": 0.5, "hold_shoot": 0.5}), MIRROR)
    batch = rollout.collect(env, actor, crit, 3)
    assert "wlim" in batch["obs"] and batch["action"].face is not None
    w0 = actor.heads.face.weight.detach().clone()
    opt = torch.optim.Adam(list(actor.parameters()) + list(crit.parameters()), lr=1e-2)
    st = ppo.update(actor, crit, opt, batch, dataclasses.replace(ppo.PPOConfig(), epochs=2, minibatch=6, target_kl=0.0))
    bad = {k: (type(v).__name__, getattr(v, "shape", None), getattr(v, "dtype", None)) for k, v in st.items()
           if torch.is_tensor(v)}
    assert not bad, bad
    assert 0.0 < st["head_keep_face"] < 1.0 and st["head_rows_face"] > 0 and "head_share_width" in st
    assert not torch.equal(w0, actor.heads.face.weight.detach())

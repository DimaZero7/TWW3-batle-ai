"""The ability branches (tools/nn/train/ability_branch.py): presses priced by replaying them in the simulator.

torch is not in the project's .venv: skipped there, run in the training container.
"""
import dataclasses
import json

import pytest

torch = pytest.importorskip("torch")

from tools.nn.model import config, policy  # noqa: E402
from tools.nn.model import critic as mcritic  # noqa: E402
from tools.nn.model.heads import Action  # noqa: E402
from tools.nn.train import ability_branch as ab  # noqa: E402
from tools.nn.train import checkpoint, head_entropy, head_reset, league, ppo, rollout  # noqa: E402

TINY = config.preset("v2", d=32, layers=2, heads=2, pointer=16, critic_d=32, critic_layers=1, critic_heads=2,
                     sector_d=16)
MIRROR = [("arena", "attack")]


def tiny(seed=0):
    torch.manual_seed(seed)
    return policy.Actor(TINY).eval(), mcritic.Critic(TINY).eval()


def env_of(n=4, mix=None):
    return rollout.Battles(league.layout(n, 1, mix or {"self": 0.5, "hold_shoot": 0.5}), MIRROR)


class Grab:
    """A capture hook that snapshots every battle at the first decision it sees."""

    def __init__(self):
        self.got = None

    def capture(self, env, info):
        if self.got is None:
            self.got = (ab.snapshot(env, torch.arange(env.B), info), info, env.rows_learn.clone(), env.B)


def ready_units(grab):
    """[(battle, side, unit, slot)] of the learner's units with a usable ability at the grabbed decision."""
    snap, info, rows, B = grab.got
    ok = info["obs"]["abil_ok"] & info["obs"]["ctrl"][..., None]
    out = []
    for r, n, k in ok.nonzero().tolist():
        out.append((int(rows[r]) % B, int(rows[r]) // B + 1, n, k))
    return out


def grabbed():
    actor, crit = tiny()
    env = env_of()
    env.brancher = Grab()
    env.step(actor, crit)
    cand = ready_units(env.brancher)
    assert cand, "the arena's lords have a ready self-cast ability at the start"
    return actor, env, cand


def pick(cand, k=None):
    t = lambda i: torch.tensor([c[i] for c in cand])
    return t(0), t(1), t(2), t(3)


def test_the_branches_repeat_exactly_and_a_same_press_on_both_sides_prices_zero():
    actor, env, cand = grabbed()
    snap = env.brancher.got[0]
    b, side, unit, slot = pick(cand)
    one = ab.play_pair(env, actor, ab.select(snap, b), unit, side, slot, 2, 4, seed=5)
    two = ab.play_pair(env, actor, ab.select(snap, b), unit, side, slot, 2, 4, seed=5)
    assert torch.equal(one[0], two[0]) and torch.equal(one[1], two[1])
    # the "pressed" branch not pressing either: the two branches are the same battle (common random numbers)
    same = ab.play_pair(env, actor, ab.select(snap, b), unit, side, torch.full_like(slot, -1), 2, 4, seed=5)
    assert torch.count_nonzero(same[0]) == 0
    # the global random state is the one before
    s = torch.get_rng_state()
    ab.play_pair(env, actor, ab.select(snap, b), unit, side, slot, 1, 2, seed=9)
    assert torch.equal(s, torch.get_rng_state())


def test_not_pressed_fires_nothing_and_spends_no_recharge():
    actor, env, cand = grabbed()
    snap = env.brancher.got[0]
    b, side, unit, slot = pick(cand)
    cd0 = torch.stack([snap[f"u/ab{k}_cd"][b, unit] for k in range(3)], 1).gather(1, slot[:, None])[:, 0]
    seen = {}
    for name, press in (("pressed", True), ("not", False)):
        e, parts, scripted = ab.branch_env(env, ab.select(snap, b))
        r_of = {int(r): i for i, r in enumerate(e.rows_learn.tolist())}
        r = torch.tensor([r_of[(int(s) - 1) * e.B + j] for j, s in enumerate(side.tolist())])
        parts[0][1].ability[r, unit] = slot if press else -1
        torch.manual_seed(1)
        e.replay(parts, scripted)
        j = torch.arange(e.B)
        on = torch.stack([e.st.u[f"ab{k}_on"][j, unit] for k in range(3)], 1).gather(1, slot[:, None])[:, 0]
        cd = torch.stack([e.st.u[f"ab{k}_cd"][j, unit] for k in range(3)], 1).gather(1, slot[:, None])[:, 0]
        seen[name] = (on, cd)
    assert (seen["not"][0] == 0).all() and (seen["not"][1] <= cd0).all()          # no effect, no recharge
    assert (seen["pressed"][1] > cd0).any() and (seen["pressed"][0] > 0).any()    # pressed: it fires


def test_the_hold_keeps_the_slot_unused_in_not_while_the_network_would_press_it():
    actor, env, cand = grabbed()
    snap = env.brancher.got[0]
    b, side, unit, slot = pick(cand)
    fired = {}
    for hold in (0.0, 100.0):
        e, parts, scripted = ab.branch_env(env, ab.select(snap, b))
        r_of = {int(r): i for i, r in enumerate(e.rows_learn.tolist())}
        r = torch.tensor([r_of[(int(s) - 1) * e.B + j] for j, s in enumerate(side.tolist())])
        parts[0][1].ability[r, unit] = -1
        j = torch.arange(e.B)
        if hold:
            until = torch.full((e.B, e.N), -1.0)
            held = torch.full((e.B, e.N), -1, dtype=torch.long)
            until[j, unit] = e.st.t + hold
            held[j, unit] = slot
            e.ability_hold = (until, held)
        torch.manual_seed(2)
        e.replay(parts, scripted)
        for i in range(6):
            torch.manual_seed(3 + i)
            e.step(actor)
        on = torch.stack([e.st.u[f"ab{k}_cd"][j, unit] for k in range(3)], 1).gather(1, slot[:, None])[:, 0]
        fired[hold] = int((on > 0).sum())
    assert fired[100.0] == 0 and fired[0.0] > 0, fired      # the untrained head presses a ready ability ~0.4 a decision


def test_a_chunk_prices_its_presses_and_missed_moments_into_the_ability_rows():
    actor, crit = tiny()
    env = env_of()
    env.brancher = ab.Brancher(ab.Config(share=1.0, missed=1.0, copies=2, horizon_s=3.0, batch=4))
    batch = rollout.collect(env, actor, crit, 3)
    br = batch["abil_branch"]
    st = env.brancher.stats
    assert br is not None and br["mask"].shape == batch["action"].ability.shape
    n = int(br["mask"].sum())
    assert n == st["replayed_presses"] + st["replayed_missed"] > 0
    # the rows priced are those of a press (its slot) or of a ready ability left (none): the logged choice
    a = batch["action"].ability[br["mask"]]
    assert int((a >= 0).sum()) == st["replayed_presses"]
    assert br["adv"][~br["mask"]].abs().sum() == 0 and br["adv"].abs().max() <= ab.ADV_CLIP
    for k in ("presses", "missed_moments", "branch_battles", "seconds", "price_rms", "price_se"):
        assert k in st
    # a batch with the term: the update runs and logs it
    cfg = dataclasses.replace(ppo.PPOConfig(), minibatch=12, ability=0.1)
    opt = torch.optim.Adam(list(actor.parameters()) + list(crit.parameters()), lr=1e-3)
    out = ppo.update(actor, crit, opt, batch, cfg)
    assert "ability_branch_loss" in out and out["ability_branch_rows"] > 0


def test_the_term_reaches_the_ability_head_and_not_the_other_heads_nor_the_critic():
    actor, crit = tiny()
    env = env_of()
    env.brancher = ab.Brancher(ab.Config(share=1.0, missed=1.0, copies=1, horizon_s=2.0, batch=4))
    batch = rollout.collect(env, actor, crit, 2)
    br = batch["abil_branch"]
    assert br is not None and bool(br["mask"].any())
    br = dict(br, adv=torch.where(br["mask"], torch.ones_like(br["adv"]), br["adv"]))
    obs = rollout.full_obs(batch["obs"], batch.get("abil_static"))
    actor.zero_grad()
    crit.zero_grad()
    logits, _ = actor.sequence(obs, batch["h0"], batch["reset"], batch["action"])
    logits = {k: v.reshape(-1, *v.shape[2:]) for k, v in logits.items()}
    act = Action(**{k: getattr(batch["action"], k).reshape(-1, *getattr(batch["action"], k).shape[2:])
                    for k in batch["action"].parts()})
    ctrl = obs["ctrl"].reshape(-1, obs["ctrl"].shape[-1])
    term, rows = ab_term = ppo.ability_term(logits, act, {k: v.reshape(-1, v.shape[-1]) for k, v in br.items()},
                                            ctrl, 0.2)
    term.backward()
    grads = {n: (p.grad is not None and float(p.grad.abs().sum()) > 0) for n, p in actor.named_parameters()}
    assert any(g for n, g in grads.items() if n.startswith("heads.ability_"))
    others = ("heads.kind", "heads.place_", "heads.fine", "heads.commit", "heads.run", "heads.q.", "heads.k.")
    assert not any(g for n, g in grads.items() if n.startswith(others)), [n for n, g in grads.items()
                                                                            if g and n.startswith(others)]
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0 for p in crit.parameters())
    assert float(rows) == float(br["mask"].sum())


def test_the_branch_batch_composition_follows_the_learner_rows_and_fills_its_slots():
    L, P, A = league.LEARNER, league.CODE["past"], league.CODE["ai_like"]
    ctrl = torch.tensor([[L, L]] * 2 + [[L, P]] * 4 + [[A, L]] * 2)       # learner rows: self 4, past 4, script 2
    assert ab.composition(ctrl, 10) == {10: 4, 9: 4, 2: 2}
    assert [ab.learner_rows(k) for k in (10, 9, 6, 8, 2)] == [2, 1, 1, 1, 1]
    for n in (1, 3, 7, 128):
        c = ab.composition(ctrl, n)
        assert sum(c.values()) == max(n, 3) and min(c.values()) >= 1


def test_mixed_kinds_play_in_one_batch_of_the_composition_padded_and_frozen():
    actor, crit = tiny()
    env = env_of(8, {"self": 0.5, "hold_shoot": 0.5})
    env.brancher = ab.Brancher(ab.Config(share=1.0, missed=1.0, copies=2, horizon_s=2.0, cap=6, batch=6))
    calls = []
    real = ab.play_pair

    def spy(template, actor_, snap, unit, side, slot, *a, **k):
        calls.append((ab.kind_of(snap).tolist(), snap["st/done"].tolist(), slot.tolist()))
        return real(template, actor_, snap, unit, side, slot, *a, **k)
    ab.play_pair = spy
    try:
        batch = rollout.collect(env, actor, crit, 3)
    finally:
        ab.play_pair = real
    comp = env.brancher.comp
    assert sum(comp.values()) == 6 and 10 in comp and set(comp) <= {10, 8, 2}
    st = env.brancher.stats
    n = st["replayed_presses"] + st["replayed_missed"]
    assert 0 < n <= 6 and len(calls) == 1                      # the cap within the composition: one batch
    kinds, done, slot = calls[0]
    assert sorted(kinds) == sorted(k for k, v in comp.items() for _ in range(v))
    assert all(done[n:]) and all(s == -1 for s in slot[n:])    # the padding: ended, nothing pressed
    assert st["branch_battles"] == 2 * 2 * 6
    assert int(batch["abil_branch"]["mask"].sum()) == n


def test_the_weight_decays_to_zero_and_the_reserved_penalty_is_refused():
    assert ab.weight(0.1, 100, 0) == pytest.approx(0.1)
    assert ab.weight(0.1, 100, 50) == pytest.approx(0.05)
    assert ab.weight(0.1, 100, 150) == 0.0
    assert ab.weight(0.1, 0, 10 ** 6) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        ab.Config(unused_penalty=0.1)


def test_the_ability_head_resets_alone_and_has_a_spread_of_its_own():
    actor, crit = tiny()
    before = {n: p.detach().clone() for n, p in actor.named_parameters()}
    names = head_reset.reset(actor, head_reset.parse("ability"), seed=3)
    assert names and all(n.startswith("heads.ability_") for n in names)
    for n, p in actor.named_parameters():
        if n not in names:
            assert torch.equal(p, before[n]), n
    assert any(not torch.equal(dict(actor.named_parameters())[n], before[n]) for n in names)
    assert head_entropy.parse("ability=0.5") == {"ability": 0.5}
    batch = rollout.collect(env_of(), actor, crit, 2)
    obs = rollout.full_obs(batch["obs"], batch.get("abil_static"))
    with torch.no_grad():
        logits, _ = actor.sequence(obs, batch["h0"], batch["reset"], batch["action"])
    logits = {k: v.reshape(-1, *v.shape[2:]) for k, v in logits.items()}
    act = Action(**{k: getattr(batch["action"], k).reshape(-1, *getattr(batch["action"], k).shape[2:])
                    for k in batch["action"].parts()})
    ctrl = obs["ctrl"].reshape(-1, obs["ctrl"].shape[-1])
    m = head_entropy.measure(logits, act, obs["free"].reshape(ctrl.shape), ctrl)
    share, nats, rows = m["ability"]
    assert float(rows) > 0 and 0.0 < float(share) <= 1.0


def test_run_logs_the_branches_and_carries_the_decay_in_the_chain(tmp_path, monkeypatch):
    from tools.nn.train import run
    monkeypatch.setattr(checkpoint, "DIR", tmp_path)
    monkeypatch.setitem(config.PRESETS, "v2", TINY)
    actor, crit = tiny()
    init = checkpoint.save(tmp_path / "init.pt", actor, crit, "v2")
    args = run.parser().parse_args(["--name", "abr", "--init", str(init), "--reward", "v2", "--battles", "4",
                                    "--steps", "2", "--updates", "2", "--minutes", "5", "--device", "cpu", "--no-eval",
                                    "--mix", '{"self": 0.5, "nearest": 0.5}', "--bank", "8", "--max-units", "4",
                                    "--keep-every", "1e-6", "--ability-branch", "1.0", "--ability-branch-missed",
                                    "1.0", "--ability-branch-copies", "1", "--ability-branch-horizon", "2",
                                    "--ability-branch-batch", "4", "--ability-branch-weight", "0.1",
                                    "--ability-branch-decay", "4", "--reset-heads", "ability",
                                    "--head-entropy", "ability=0.5"])
    _, _, out = run.train(args)
    rows = [json.loads(x) for x in (out / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all("ability_branch" in r and "presses" in r["ability_branch"] and "head_share_ability" in r for r in rows)
    assert [r["ability_branch"]["weight"] for r in rows] == [0.1, 0.075]
    assert checkpoint.train_state(out / "latest.pt")["state"]["ability_updates"] == 2



def test_the_reliability_is_the_real_share_of_the_prices_spread():
    g = torch.Generator().manual_seed(0)
    signal = torch.randn(4000, 1, generator=g)
    assert abs(ab.reliability(signal + torch.randn(4000, 4, generator=g) * 2.0) - 0.5) < 0.05   # var 1 vs 4 / 4
    assert ab.reliability(torch.randn(4000, 4, generator=g)) < 0.05
    assert ab.reliability(signal.expand(-1, 4)) == 1.0
    assert ab.reliability(signal[:4].expand(-1, 4)) is None and ab.reliability(signal) is None


def test_a_chunk_keeps_its_prices_per_copy_and_logs_their_real_share():
    actor, crit = tiny()
    env = env_of(8)
    env.brancher = ab.Brancher(ab.Config(share=1.0, missed=1.0, copies=2, horizon_s=3.0, batch=8))
    rollout.collect(env, actor, crit, 4)
    br = env.brancher
    n = br.stats["replayed_presses"] + br.stats["replayed_missed"]
    assert br.copies.shape == (n, 2) and torch.allclose(br.last["price"], br.copies.mean(1))
    assert n < 8 or br.stats["price_rel"] == round(ab.reliability(br.copies), 3)


def test_open_loop_not_replays_the_pressed_orders_exactly_and_never_presses():
    actor, env, cand = grabbed()
    snap = env.brancher.got[0]
    b, side, unit, slot = pick(cand)
    # nothing pressed in either: the open-loop "not" replays "pressed"'s orders, so the two are the same battle
    for greedy in (False, True):
        same = ab.play_pair(env, actor, ab.select(snap, b), unit, side, torch.full_like(slot, -1), 2, 6, seed=5,
                            openloop=True, greedy=greedy)
        assert torch.count_nonzero(same[0]) == 0
    # pressed: the unit's slot stays unused in "not" though "pressed"'s orders are replayed there
    fired = {}

    def each(i, pe, ne):
        j = torch.arange(pe.B)
        n = unit.repeat_interleave(2)
        s = slot.repeat_interleave(2)
        for name, e in (("pressed", pe), ("not", ne)):
            cd = torch.stack([e.st.u[f"ab{k}_cd"][j, n] for k in range(3)], 1).gather(1, s[:, None])[:, 0]
            fired[name] = int((cd > 0).sum())
    ab.play_pair(env, actor, ab.select(snap, b), unit, side, slot, 2, 6, seed=5, openloop=True, each=each)
    assert fired["pressed"] > 0 and fired["not"] == 0


def test_greedy_branches_repeat_and_take_the_most_likely_action():
    actor, env, cand = grabbed()
    snap = env.brancher.got[0]
    b, side, unit, slot = pick(cand)
    one = ab.play_pair(env, actor, ab.select(snap, b), unit, side, slot, 2, 4, seed=5, greedy=True)
    two = ab.play_pair(env, actor, ab.select(snap, b), unit, side, slot, 2, 4, seed=5, greedy=True)
    assert torch.equal(one[0], two[0])
    seen = []
    e, _, _ = ab.branch_env(env, ab.select(snap, b))
    real = type(e)._act
    type(e)._act = lambda self, *a: seen.append(a[-1]) or real(self, *a)
    try:
        e._act = ab._greedy_act(e)
        e.step(actor)
    finally:
        type(e)._act = real
    assert seen and all(seen)


def test_the_open_loop_and_greedy_flags_reach_the_branches():
    from tools.nn.train import run
    args = run.parser().parse_args(["--ability-branch", "0.2", "--ability-branch-openloop", "--ability-branch-greedy"])
    assert args.ability_branch_openloop and args.ability_branch_greedy
    actor, crit = tiny()
    env = env_of()
    env.brancher = ab.Brancher(ab.Config(share=1.0, missed=1.0, copies=2, horizon_s=3.0, batch=4, openloop=True,
                                         greedy=True))
    calls = []
    real = ab.play_pair

    def spy(*a, **k):
        calls.append((k.get("openloop"), k.get("greedy")))
        return real(*a, **k)
    ab.play_pair = spy
    try:
        rollout.collect(env, actor, crit, 3)
    finally:
        ab.play_pair = real
    assert calls and all(c == (True, True) for c in calls)
    assert env.brancher.stats["replayed_presses"] + env.brancher.stats["replayed_missed"] > 0

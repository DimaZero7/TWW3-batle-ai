"""Domain randomisation of the simulator's uncertain numbers (tools/nn/train/dr.py, tools/nn/sim/knobs.py): the draws stay
inside their ranges with the centre their median, only in the battles asked; the simulator reads every scalar knob and
the draws change the battle; OFF nothing is drawn, and a knob drawn at its centre plays the same battle bit for bit as
no knob (the simulator's code without the knobs). Needs torch: run in the container."""
import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import battle, knobs, replay, scenario  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402
from tools.nn.train import dr, league, randomise, rollout  # noqa: E402

P = load()
EMP, SKV = "wh_main_emp_empire", "wh2_main_skv_skaven"


def mixed(B=2):
    """Empire (general, spearmen, swordsmen, archers) west against Skaven (warlord, clanrats, slaves, slingers with the
    skirmish mode: side 2 plays as the game's AI) east, ~160 m apart: melee, missiles, lords, routs and skirmish runs."""
    def side(sign, faction, units, faces):
        return {"faction": faction, "units": [
            {"key": k, "general": i == 0, "x": sign * (80 + (20 if k.endswith("archers_0") or "slingers" in k else 0)),
             "z": -45 + 30 * i, "b": faces} for i, k in enumerate(units)]}
    army = {"attacker": 1, "sides": {
        1: side(-1, EMP, ["wh_main_emp_cha_general_0", "wh_main_emp_inf_spearmen_0", "wh_main_emp_inf_swordsmen",
                          "wh2_dlc13_emp_inf_archers_0"], 90),
        2: side(1, SKV, ["wh2_main_skv_cha_warlord_0", "wh2_main_skv_inf_clanrats_0", "wh2_main_skv_inf_skavenslaves_0",
                         "wh2_main_skv_inf_skavenslave_slingers_0"], 270)}}
    return scenario.build([army] * B, P)


def play(st, seconds=90.0):
    return battle.run(st, replay.nearest_attack, P, until_s=seconds, compile=False)


def with_knobs(st, which):
    """Every scalar knob of the config in the state at its centre / lo / hi (as a draw that landed there)."""
    B, N = st.u["side"].shape
    for s in dr.load(params=P).scalars:
        st.u[knobs.PREFIX + s.name] = torch.full((B, N), getattr(s, which))
    return st


def test_a_knob_at_its_centre_plays_the_same_battle_as_no_knob():
    a = play(mixed())
    b = play(with_knobs(mixed(), "centre"))
    assert float(a.u["hp"].sum()) < float(mixed().u["hp"].sum())       # the battle was fought
    for k, v in a.u.items():
        assert torch.equal(v, b.u[k]), k
    assert torch.equal(a.t, b.t) and torch.equal(a.done, b.done) and torch.equal(a.winner, b.winner)


def test_the_simulator_reads_every_scalar_knob(monkeypatch):
    seen = set()
    real = knobs.get

    def spy(u, name, default, dims=2):
        seen.add(name)
        return real(u, name, default, dims)
    monkeypatch.setattr(knobs, "get", spy)
    play(with_knobs(mixed(1), "centre"), 30.0)
    assert seen >= set(knobs.SCALARS), set(knobs.SCALARS) - seen


def test_the_draws_change_the_battle():
    a, b = play(with_knobs(mixed(1), "lo")), play(with_knobs(mixed(1), "hi"))
    assert not torch.equal(a.u["hp"], b.u["hp"])


def test_draws_stay_in_range_with_the_centre_their_median_and_only_where_asked():
    cfg = dr.load(params=P)
    st = mixed(2000)
    before = {k: v.clone() for k, v in st.u.items()}
    rows = torch.arange(2000) % 2 == 0
    out = dr.apply(st, rows, cfg, torch.Generator().manual_seed(0))
    for s in cfg.scalars:
        v = st.u[knobs.PREFIX + s.name]
        assert torch.equal(v, v[:, :1].expand_as(v))                  # one value a battle
        x = v[rows, 0]
        assert float(x.min()) >= s.lo - 1e-5 and float(x.max()) <= s.hi + 1e-5
        assert abs(float(x.median()) - s.centre) <= 0.06 * (s.hi - s.lo) + 1e-6
        assert torch.allclose(v[~rows], torch.full_like(v[~rows], s.centre))   # not asked: the centre
        assert torch.equal(out[s.name], v[:, 0])
    for f in cfg.fields:
        x0, x1 = before[f.field], st.u[f.field]
        assert torch.equal(x1[~rows], x0[~rows])                       # not asked: unchanged
        for c, lo, hi in f.entries:
            m = ((x0 - c).abs() < dr.TOL) & rows[:, None]
            if m.any():
                assert float(x1[m].min()) >= lo - 1e-5 and float(x1[m].max()) <= hi + 1e-5
                assert not torch.equal(x1[m], x0[m])
        other = ~torch.stack([(x0 - c).abs() < dr.TOL for c, _, _ in f.entries]).any(0)
        assert torch.equal(x1[other], x0[other])                       # units of no listed centre keep theirs
    assert torch.equal(st.u["damage"], before["damage"])               # randomise.py's numbers are not touched


def test_battles_draw_at_start_and_restart_and_nothing_when_off():
    lay = league.layout(4, 1, opponent="nearest")
    env = rollout.Battles(lay, [("arena", "attack")], params=rollout.params_with_limit(1.0), spread=randomise.NONE)
    assert not any(k.startswith(knobs.PREFIX) for k in env.st.u)
    env = rollout.Battles(lay, [("arena", "attack")], params=rollout.params_with_limit(1.0), spread=randomise.NONE,
                          dr=dr.load(params=P))
    k = knobs.PREFIX + "hold_rate"
    first = env.st.u[k][:, 0].clone()
    assert len(set(first.tolist())) == 4                                # each battle its own draw
    env._reset(torch.tensor([True, False, False, False]))
    again = env.st.u[k][:, 0]
    assert again[0] != first[0] and torch.equal(again[1:], first[1:])


def test_off_by_default_in_run_py():
    from tools.nn.train import run
    assert run.parser().parse_args([]).dr is None
    assert run.parser().parse_args(["--dr"]).dr == str(dr.DEFAULT)
    assert run.parser().parse_args(["--dr", "x.json"]).dr == "x.json"


def test_a_training_run_with_dr_trains(tmp_path, monkeypatch, capsys):
    from tools.nn.model import config, critic, policy
    from tools.nn.train import checkpoint, run
    torch.manual_seed(0)
    cfg = config.preset("small", d=32, layers=2, heads=2, pointer=16, critic_d=32, critic_layers=1, critic_heads=2)
    init = checkpoint.save(tmp_path / "init.pt", policy.Actor(cfg).eval(), critic.Critic(cfg).eval())
    monkeypatch.setattr(checkpoint, "DIR", tmp_path)
    args = run.parser().parse_args(["--name", "dr", "--init", str(init), "--battles", "4", "--steps", "2", "--updates",
                                    "2", "--minutes", "5", "--device", "cpu", "--no-eval", "--mix", '{"nearest": 1.0}',
                                    "--dr", "--bank", "8", "--max-units", "4"])
    _, summary, _ = run.train(args)
    assert summary["updates"] == 2
    assert "domain randomisation" in capsys.readouterr().out

"""The operative's cases (net/operative/cases.py: gather / split / go and the armies' ones): the formula
groups each case as the plan says, the armies' cases have a lord first, a short batch of the formula
plays without errors and gives every case's measure (env.measures). Needs torch (the container, CPU)."""
import pytest

torch = pytest.importorskip("torch")

from net.operative import cases as CS  # noqa: E402
from net.operative import cfg  # noqa: E402
from net.operative import env as EN  # noqa: E402
from net.operative import frame as FR  # noqa: E402
from net.operative import groups as GR  # noqa: E402
from net.operative import inputs as IN  # noqa: E402
from net.operative import lower as LW  # noqa: E402
from net.operative import net as NN  # noqa: E402
from net.operative import step as SP  # noqa: E402
from net.operative import target as TG  # noqa: E402
from net.operative import teacher as TC  # noqa: E402
from sim.params import load  # noqa: E402
from net.tactic import decide as DC  # noqa: E402
from net.tactic import parcel as PC  # noqa: E402
from net.tactic import raw as TR  # noqa: E402
from net.tactic import train as TT  # noqa: E402

P = load()


def setup_of(n=6, seed=0, cases=CS.OLD):
    sp = CS.specs(n, torch.Generator().manual_seed(seed), fixed=True, cases=cases)
    s = CS.Setup(sp, P, "cpu")
    return s, TR.Seer(P).raw(s.st)


def decide(net, setup, raw, groups, act=None):
    axes = FR.Axes.of(raw, setup.side, setup.st.bounds)
    inp = IN.build(raw, setup.card, setup.side, axes, groups, setup.command, bounds=setup.st.bounds,
                   K=EN.rows_of(raw.shape[1]))
    ctx = NN.ctx_of(inp, raw, axes, groups)
    out, _ = net(inp, ctx, act=act)
    return out, inp, ctx, axes


def test_the_formula_groups_the_cases_as_the_plan_says():
    setup, raw = setup_of(6, seed=6)
    B, N = raw.shape[:2]
    axes = FR.Axes.of(raw, setup.side, setup.st.bounds)
    groups = GR.Groups.empty(B, N)
    inp = IN.build(raw, setup.card, setup.side, axes, groups, setup.command, bounds=setup.st.bounds, K=N)
    ctx = NN.ctx_of(inp, raw, axes, groups)
    for mode, want in (("formula", {"gather": 1, "split": 2, "go": 1}), ("each", {"gather": 3, "split": 6, "go": 3}),
                       ("blob", {"gather": 1, "split": 1, "go": 1})):
        out = TC.play_formula(inp, ctx, setup.destroy, mode)
        for b in range(B):
            assert int(out["new"].on[b].sum()) == want[CS.CASES[int(setup.case[b])]], (mode, b)
    out = TC.play_formula(inp, ctx, setup.destroy, "formula")
    new = out["new"]
    for b in range(B):
        if CS.CASES[int(setup.case[b])] == "split":
            # each clump its own group, its own enemy three
            for k in (0, 1):
                g = int(new.member[b, setup.clump[b] == k][0])
                assert bool((new.member[b, setup.clump[b] == k] == g).all())
                xz = raw[b][..., [TR.I["x"], TR.I["z"]]]
                t = xz[new.target[b, g]].mean(0)
                m = xz[setup.clump[b] == k].mean(0)
                o = xz[setup.clump[b] == 1 - k].mean(0)
                assert float((t - m).norm()) < float((t - o).norm())
    tac = LW.Tacticians(B, N, int(cfg()["groups"]), "cpu")
    p, idx = tac.apply(new, out["task"], out["moved"], raw)
    assert not any(PC.check(raw, p, EN.field_dict(axes, None)))


def test_the_armies_cases_have_a_lord_first_and_the_formula_groups_them():
    sp = CS.specs(12, torch.Generator().manual_seed(9), fixed=True, cases=CS.ARMY)
    setup = CS.Setup(sp, P, "cpu")
    raw = TR.Seer(P).raw(setup.st)
    B, N = raw.shape[:2]
    lords = cfg()["cases"]["army"]["lords"]
    for b, s in enumerate(sp):
        n = int(setup.own[b].sum())
        assert 3 <= n <= 8
        assert setup.st.keys[b][0] == lords[s["ours"]] and setup.st.keys[b][CS.H] == lords[s["theirs"]]
        assert bool(setup.st.u["lord"][b, 0]) and bool(setup.st.u["lord"][b, CS.H])
        assert float(setup.st.u["range"][b, 0]) == 0.0                   # the lord: a melee unit of its group
    axes = FR.Axes.of(raw, setup.side, setup.st.bounds)
    groups = GR.Groups.empty(B, N)
    inp = IN.build(raw, setup.card, setup.side, axes, groups, setup.command, bounds=setup.st.bounds, K=N)
    ctx = NN.ctx_of(inp, raw, axes, groups)
    out = TC.play_formula(inp, ctx, setup.destroy, "formula")
    for b, s in enumerate(sp):
        mem = out["new"].members()[b] & out["new"].on[b][:, None]
        assert bool((mem.sum(0)[setup.own[b]] == 1).all())               # every unit of mine in one group
        if s["case"] == "split_a":
            assert int(out["new"].on[b].sum()) >= 2
    tac = LW.Tacticians(B, N, int(cfg()["groups"]), "cpu")
    p, idx = tac.apply(out["new"], out["task"], out["moved"], raw)
    assert not any(PC.check(raw, p, EN.field_dict(axes, None)))


@pytest.fixture(scope="module")
def unit_net():
    return TT.unit_network("cpu")


def short(n=3, limit=40.0, seed=0):
    sp = CS.specs(n, torch.Generator().manual_seed(seed), fixed=True)
    s = CS.Setup(sp, P, "cpu")
    s.limit = torch.full_like(s.limit, limit)
    return s


def test_the_formula_plays_without_errors(unit_net):
    s = short(limit=20.0, seed=1)
    _, rs, rd = EN.play(s, unit_net, mode="formula", keep=False)
    assert rs["parcel_errors"] == 0
    assert rs["groups_mean"] >= 1.0
    m = EN.measures(s, rd)
    assert set(m) == set(CS.CASES)

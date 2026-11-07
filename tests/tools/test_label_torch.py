"""Разметка (tools/label): the session pauses exactly on the person's level's period, the person's decision goes through
the level's own bans and tactic/parcel.check (an error is not written), 'keep' keeps the parcels in force, the written
samples rebuild the level's input exactly, and the operative's sample gives the network's imitation term. Needs torch
(the container; the simulator on the CPU)."""
import copy

import pytest

torch = pytest.importorskip("torch")

from tools.label import human as HU  # noqa: E402
from tools.label import record as RC  # noqa: E402
from tools.label import session as SE  # noqa: E402
from tools.nn.operative import net as NN  # noqa: E402
from tools.nn.tactic import train as TT  # noqa: E402

MATCH = {"name": "t", "map": "flat", "limit_s": 60, "command": {"verb": "hold", "pace": "any"}, "enemy": "attack",
         "sides": {"1": {"faction": "wh_main_emp_empire", "units": [
             {"key": "wh_main_emp_inf_spearmen_0", "x": -60, "z": -300, "b": 0},
             {"key": "wh_main_emp_inf_swordsmen", "x": 60, "z": -300, "b": 0},
             {"key": "wh_main_emp_inf_crossbowmen", "x": 0, "z": -350, "b": 0}]},
             "2": {"faction": "wh2_main_skv_skaven", "units": [
                 {"key": "wh2_main_skv_inf_clanrat_spearmen_0", "x": -70, "z": 300, "b": 180},
                 {"key": "wh2_main_skv_inf_clanrats_0", "x": 70, "z": 300, "b": 180},
                 {"key": "wh2_main_skv_inf_clanrats_1", "x": 0, "z": 340, "b": 180}]}}}


@pytest.fixture(scope="module")
def unit_net():
    return TT.unit_network("cpu")


def session(unit_net, tmp_path, level="operative", period=None, m=None):
    return SE.Session(copy.deepcopy(m or MATCH), level=level, period=period, nets={"unit": unit_net}, out_root=tmp_path)


def play_out(s, edit_of=lambda p: {"keep": True}):
    secs = []
    while not s.over:
        s.advance()
        if s.pending is not None:
            secs.append(s.pending["sec"])
            r = s.decide(edit_of(s.pending))
            assert r["ok"], r
    return secs


def test_operative_pauses_on_its_period_and_writes(unit_net, tmp_path):
    s = session(unit_net, tmp_path)
    s.advance()
    assert s.pending["level"] == "operative" and s.pending["sec"] == 0
    assert not s.decide({"keep": True})["ok"]               # no groups yet: nothing to keep
    assert s.n_decisions == 0
    assert s.decide(s.pending["hint"])["ok"]
    secs = [0] + play_out(s)
    assert secs == [0, 15, 30, 45]                     # (the battle ends at its limit, 60 s)
    S = RC.load(s.folder)
    assert len(S) == 4 and [x["sec"] for x in S] == secs
    assert [x["source"] for x in S] == ["hint"] + ["keep"] * 3
    assert (s.folder / "replay.json").exists() and (s.folder / "result.json").exists()


def test_own_period(unit_net, tmp_path):
    s = session(unit_net, tmp_path, period=20)
    s.advance()
    s.decide(s.pending["hint"])
    assert [0] + play_out(s) == [0, 20, 40]


def test_keep_is_the_parcels_in_force(unit_net, tmp_path):
    s = session(unit_net, tmp_path)
    s.advance()
    s.decide(s.pending["hint"])
    s.advance()
    before = s.groups.clone()
    mine, cx = s.tac.mine.clone(), s.tac.cx.clone()
    sent = s.parcels
    assert s.decide({"keep": True})["ok"]
    g = s.groups
    for f in ("member", "on", "target", "seed", "cx", "cz", "heading", "line", "verb", "pace", "n0"):
        assert torch.equal(getattr(g, f), getattr(before, f)), f
    assert torch.equal(s.tac.mine, mine) and torch.equal(s.tac.cx, cx) and s.parcels == sent   # no parcel sent


def test_a_banned_decision_is_refused_and_not_written(unit_net, tmp_path):
    s = session(unit_net, tmp_path)
    s.advance()
    s.decide(s.pending["hint"])
    s.advance()
    n = len(RC.load(s.folder))
    # a zone goal far outside the field: the walk's cell is the nearest cell, the step by code; a unit in no group
    bad = {"groups": [{"g": 0, "members": [0, 2], "task": None}]}
    r = s.decide(bad)
    assert not r["ok"] and any("есть группа" in e for e in r["errors"])
    # two groups whose zones would stand 240 m apart or less with different targets: zone_sep_m's ban
    sep = {"groups": [{"g": 0, "members": [0, 2], "task": {"seed": -1, "goal": [0, -200], "form": "line", "turn": 0}},
                      {"g": 1, "members": [1], "task": None}]}
    r = s.decide(sep)
    assert not r["ok"] and any("240 м" in e for e in r["errors"])
    # an enemy unit as a member
    r = s.decide({"groups": [{"g": 0, "members": [0, 1, 2, 3], "task": None}]})
    assert not r["ok"]
    assert len(RC.load(s.folder)) == n and s.pending is not None
    # a correct change: two groups far enough apart
    ok = {"groups": [{"g": 0, "members": [0], "task": {"seed": 3, "goal": [-200, -300], "turn": 0}},
                     {"g": 1, "members": [1, 2], "task": {"seed": 4, "goal": [200, -320], "turn": 0}}]}
    pv = s.preview(ok)
    assert pv["ok"], pv
    r = s.decide(ok)
    assert r["ok"] and r["source"] == "human"
    assert s.groups.on[0].sum() == 2 and s.parcel_errors == []


def test_picks_are_watched_and_bans_found(unit_net, tmp_path):
    s = session(unit_net, tmp_path)
    s.advance()
    inp, ctx = s.op
    act, plan, _ = HU.op_act(s.pending["hint"], ctx, s.axes)
    act["verb"][:] = 0                                      # 'destroy': closed (verbs_open: hold)
    _, errs = HU.op_walk(inp, ctx, act, plan)
    assert any("приказ" in e for e in errs)


def test_operative_sample_rebuilds_exactly_and_teaches(unit_net, tmp_path):
    s = session(unit_net, tmp_path)
    s.advance()
    live = s.op[0]
    s.decide(s.pending["hint"])
    smp = RC.load(s.folder)[0]
    inp, ctx, act = RC.rebuild(smp)
    for k, v in live.items():
        if torch.is_tensor(v):
            assert torch.equal(inp[k], v), k
    net = NN.OperativeNet()
    out, _ = net(inp, ctx, act=act, labels=RC.labels(smp))
    t = out["teach"]["term"]
    t = t.detach()
    assert torch.isfinite(t) and float(t) > 0 and out["teach"]["n"] > 0


def test_tactic_level(unit_net, tmp_path):
    s = session(unit_net, tmp_path, level="tactic")
    s.advance()
    p = s.pending
    assert p["level"] == "tactic" and p["sec"] == 0 and not p["can_keep"]
    assert not s.decide({"keep": True})["ok"]               # no frames yet: every unit must change
    live = s.pending["_T"]["inp"]
    assert s.decide(p["edit"])["ok"]
    smp = RC.load(s.folder)[0]
    inp, order, parcel, tiles, act, prev = RC.rebuild(smp)
    for k, v in live.items():
        if torch.is_tensor(v):
            assert torch.equal(inp[k], v), k
    s.advance()
    assert s.pending["sec"] == 5
    # move one unit's frame: the code nudges it into a free cell; a frame far out of the zone is refused
    e = copy.deepcopy(s.pending["edit"])
    u = e["rows"][0]["units"][0]
    far = copy.deepcopy(e)
    far["rows"][0]["units"][0].update(keep=False, x=u["x"] + 900.0)
    r = s.decide(far)
    assert not r["ok"]
    u.update(keep=False, w=48.0, d=48.0)
    pv = s.preview(e)
    assert pv["ok"], pv
    assert s.decide(e)["ok"]
    secs = play_out(s)
    assert secs == list(range(10, 60, 5))
    S = RC.load(s.folder)
    assert [x["sec"] for x in S] == list(range(0, 60, 5))


def test_strategist_level(unit_net, tmp_path):
    from tools.label import strategist as ST
    s = session(unit_net, tmp_path, level="strategist")
    p = s.pending
    assert p["level"] == "strategist" and p["sec"] == 0 and not p["can_keep"]
    live = s.strat.inp
    # a zone far out of my deployment zone is refused, nothing written
    bad = copy.deepcopy(p["hint"])
    bad["groups"][0]["goal"] = [0.0, 300.0]
    r = s.decide(bad)
    assert not r["ok"] and any("зоне расстановки" in e or "зона" in e for e in r["errors"])
    # the person: every unit one group, the zone 100 m left of the formula's
    e = copy.deepcopy(p["hint"])
    g = e["groups"][0]
    g["members"] = sorted(sum((x["members"] for x in e["groups"]), []))
    e["groups"] = [g]
    g["goal"] = [g["goal"][0] - 100.0, g["goal"][1]]
    pv = s.preview(e)
    assert pv["ok"], pv
    assert s.decide(e)["ok"]
    smp = RC.load(s.folder)[0]
    inp, ctx, act = ST.rebuild(smp)
    for k, v in live.items():
        if torch.is_tensor(v):
            assert torch.equal(inp[k], v), k
    u = s.b.st.u
    xs = [float(u["x"][0, i]) for i in range(s.b.st.N) if bool(s.b.own[0, i])]
    assert max(abs(x - pv["groups"][0]["cx"]) for x in xs) < 200.0           # my units stand in the group's tiles
    while not s.over:
        s.advance()
        assert s.pending is None                                                # the battle plays itself after
    assert s.parcels >= 1 and s.parcel_errors == []


def test_men_of_the_fighters_lab_and_their_patches_end_with_the_battle(unit_net, tmp_path):
    from tools.nn.sim import battle
    plain = battle.step
    s = SE.Session(copy.deepcopy(MATCH), level="operative", nets={"unit": unit_net}, out_root=tmp_path, men=True)
    assert battle.step is not plain                                       # the lab's step while the battle lives
    s.advance()
    s.decide(s.pending["hint"])
    snaps = s.advance(3)
    men = snaps[-1]["men"]
    own = [u for u in snaps[-1]["units"] if u[1] == 1]
    assert men and all(len(men[u[0]]) == 2 * u[7] for u in own)          # a point a living man
    s.stop()
    assert battle.step is plain                                           # and gone with it
    assert RC.load(s.folder)[0]["level"] == "operative"
    info = s.state()["units"]
    assert info[0]["name"] and info[0]["kind"] in ("infantry", "shooters", "lord", "cavalry", "monster", "artillery")

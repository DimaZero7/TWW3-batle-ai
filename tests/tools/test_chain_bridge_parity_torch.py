"""The chain in the game one to one with the simulator's chain (CLAUDE.md "Вход сети один в один";
build/diag_g49/bridge_parity.md, section "цепочка"; tools/nn/companion/chain.py):

* the companion's chain (Chain.step) on the simulator's raw states of a strategist battle gives, second by second, the
  same orders of the unit network as the simulator's chain (strategist/battle.play: the strategist's groups at second 0,
  the operative every 15 s, the tacticians every 5 s, the units every second) - the same clocks, the same order of the
  levels, the same areas, arrival and views (the unit network greedy, so a choice is a function of its input);
* the strategist's plan in the companion (deploy.Strategist.plans) gives the operative the same starting groups as the
  simulator's deployment (strategist/battle.placed);
* the whole path on the bridge's documents of a battle (the tactician's parity test's doc_of_sim): orders for own units
  only, a valid orders file, the leash and the areas in the log.
Needs torch (the container)."""
import pytest

torch = pytest.importorskip("torch")

from tests.tools import test_strategist_bridge_parity_torch as SP  # noqa: E402
from tests.tools import test_tactic_bridge_parity_torch as TP  # noqa: E402
from tools.nn.companion import chain as CH  # noqa: E402
from tools.nn.companion import deploy as CD  # noqa: E402
from tools.nn.companion import exchange  # noqa: E402
from tools.nn.operative import groups as GR  # noqa: E402
from tools.nn.operative import lower as LW  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim import scenario  # noqa: E402
from tools.nn.strategist import battle as BT  # noqa: E402
from tools.nn.strategist import cases as CS  # noqa: E402
from tools.nn.tactic import decide as DC  # noqa: E402
from tools.nn.tactic import raw as TR  # noqa: E402
from tools.nn.unit import net as Nt  # noqa: E402
from tools.nn.unit import train as UT  # noqa: E402

P = TP.P
SECONDS = 40           # the operative at 0, 15, 30; the tacticians every 5 s


def fresh_unit_net(seed=0):
    torch.manual_seed(seed)
    return Nt.UnitNet().eval()


def plan_of(pre, out, inp, b=0):
    """The plan dict the companion's strategist keeps (deploy.Strategist.chain_plan) of the simulator's deployment."""
    _, g0, pl, _ = BT.placed(pre, out, inp)
    plan = out["plan"]
    own = pre.own[b]
    return {"group": {f"u{i}": int(pl["group"][b, i]) for i in range(own.shape[0]) if bool(own[i])},
            **{k: plan[k][b].tolist() for k in ("on", "cx", "cz", "heading", "line", "n")},
            "verb": DC.VERBS[int(plan["verb"][b])], "pace": DC.PACES[int(plan["pace"][b])]}, g0


def record_sim(monkeypatch, pre, out, inp, unit_net):
    """Play the simulator's chain (strategist/battle.play) SECONDS long; record each second's raw state, the unit level's
    raw (its leash column) and the units' intended orders."""
    rec = {"raw": [], "raw_u": [], "intended": []}
    seer_raw, second = TR.Seer.raw, LW.Units.second

    def raw(self, st, dt=1.0):
        r = seer_raw(self, st, dt)
        rec["raw"].append(r.clone())
        return r

    def units_second(self, st):
        lands = second(self, st)
        rec["raw_u"].append(self.ubuf[-1].clone())
        rec["intended"].append(self.intended.clone())
        return lands

    monkeypatch.setattr(TR.Seer, "raw", raw)
    monkeypatch.setattr(LW.Units, "second", units_second)
    pre.limit = torch.full_like(pre.limit, float(SECONDS))
    BT.play(pre, out, inp, unit_net)
    monkeypatch.setattr(TR.Seer, "raw", seer_raw)
    monkeypatch.setattr(LW.Units, "second", second)
    return rec


@pytest.mark.parametrize("task", ["attack", "hold"])
def test_the_companions_chain_gives_the_simulators_orders_second_by_second(monkeypatch, task):
    sample = Nt.sample
    monkeypatch.setattr(Nt, "sample", lambda logits, greedy=False: sample(logits, True))
    sp = [s for s in CS.exam_specs(n=8, seed=3, arena=True) if s["case"] == task][:1]
    pre = CS.Pre(sp, P, "cpu")
    out, inp, _ = BT.decide(pre)
    plan, _ = plan_of(pre, out, inp)
    unit_net = fresh_unit_net()
    rec = record_sim(monkeypatch, pre, out, inp, unit_net)
    D = len(rec["raw"])                  # seconds 0 .. SECONDS - 1 (the battle ends at the limit)
    assert D == SECONDS and len(rec["intended"]) == D
    st, _, _, _ = BT.placed(pre, out, inp)
    N = st.N
    # (strategist/battle.play: the plain chain, no translator rules)
    chain = CH.Chain(unit_net, plans={"sim": plan}, params=P, targets=False, still=False, aim=False)
    chain.start("sim", [f"u{i}" for i in range(N)], st.keys[0], UT.cards(st, "cpu").float()[:1], rec["raw"][0])
    own = pre.own[0]
    moved = 0
    for s in range(D):
        _, intended, _ = chain.step(rec["raw"][s], s, raw_u=rec["raw_u"][s])
        want = rec["intended"][s]
        for f in O.FIELDS:
            a, b = getattr(intended, f)[0, own], getattr(want, f)[0, own]
            assert torch.allclose(a.float(), b.float(), atol=1e-4), (s, f, a.tolist(), b.tolist())
        moved += int((want.kind[0, own] == O.MOVE).sum())
    assert chain.stats["op_decisions"] == 3 and chain.stats["tactic_ticks"] == 8
    assert chain.stats["parcel_errors"] == 0 and chain.stats["from_strategist"]
    assert moved > 0                     # the units went somewhere: the comparison is not of empty orders


def test_the_strategists_plan_in_the_companion_gives_the_simulators_starting_groups():
    pre = CS.Pre(CS.exam_specs(n=4, seed=7, arena=True), P, "cpu")
    out, inp, _ = BT.decide(pre)
    _, g0, _, _ = BT.placed(pre, out, inp)
    for b, s in enumerate(pre.sp):
        doc = SP.doc_of_pre(pre, b)
        st = CD.Strategist("formula", s["task"], params=P)
        st.decide(doc)
        plan = st.plans[doc["batch"]]
        names = [u["n"] for u in doc["units"]]
        g = CH.groups_of_plan(plan, names, len(names), g0.G)
        for i, n in enumerate(names):
            assert int(g.member[0, i]) == int(g0.member[b, int(n[1:])]), n
        for k in ("on", "cx", "cz", "heading", "line", "n0", "verb", "pace"):
            a, w = getattr(g, k)[0], getattr(g0, k)[b]
            assert torch.allclose(a.float(), w.float(), atol=1e-3), k
        assert plan["verb"] == DC.VERBS[int(out["plan"]["verb"][b])] and plan["pace"] == DC.PACES[int(out["plan"]["pace"][b])]


def test_the_chain_answers_the_bridges_documents_with_orders_for_its_own_units():
    torch.manual_seed(5)
    st = scenario.build([TP.skirmish(gap=160.0)], P)
    chain = CH.Chain(fresh_unit_net(), task="destroy", params=P)       # no strategist: the operative makes the groups
    kinds, areas = set(), 0
    for s in range(20):
        doc = TP.doc_of_sim(st, 0, 1000 * s, own=1)
        orders, think_ms, uses = chain.decide(doc)
        side = {u["n"]: u["side"] for u in doc["units"]}
        assert orders and all(side[o["unit"]] == 1 for o in orders) and uses == []
        text = exchange.orders_text(doc["batch"], doc["move"], orders, think_ms)
        parsed = exchange.parse_orders(text)
        assert parsed is not None and parsed["move"] == doc["move"]
        for o in orders:
            kinds.add(o["kind"])
            if o["kind"] == "attack":
                assert side[o["target"]] == 2
        extra = chain.log_extra()
        assert set(extra["areas"]) == set(extra["inside"]) == {n for n, k in side.items() if k == 1}
        assert set(extra["ms"]) == {"raw_ms", "op_ms", "tac_ms", "unit_ms"}
        areas += len(extra["areas"])
        if s == 0:
            assert extra.get("operative") and all(g["task"] for g in extra["operative"])
        for _ in range(round(1.0 / P.dt)):
            from tools.nn.sim import battle
            from tools.nn.train import opponents
            battle.step(st, opponents.nearest(st), P)
    assert kinds <= {"move", "attack", "hold", "keep"} and kinds & {"move", "attack"}
    assert chain.stats["op_decisions"] == 2 and chain.stats["tactic_ticks"] == 4 and chain.stats["parcel_errors"] == 0
    assert isinstance(chain.groups, GR.Groups) and bool(chain.groups.on.any())


def test_the_leash_reads_the_raw_state_as_the_simulators_state():
    """decode.Leash on the companion's raw (chain.leash_state) sees what it sees on the simulator's state."""
    from tools.nn.unit import decode as Dc
    from tools.nn.unit import view as V
    torch.manual_seed(2)
    st = scenario.build([TP.skirmish(gap=40.0)], P)
    from tools.nn.sim import battle
    from tools.nn.train import opponents
    for _ in range(30):
        battle.step(st, opponents.nearest(st), P)
    raw_u = V.raw(st, P.sim["formation"]["spacing_m"], missile=P.sim["missile"])
    sh = CH.leash_state(raw_u)
    u = st.u
    for k in ("m", "r", "x", "z", "vx", "vz", "range"):
        assert torch.allclose(sh.u[k].float(), u[k].float(), atol=1e-5), k
    assert torch.equal(sh.u["men"] > 0, u["men"] > 0)
    own = u["side"] == 1
    o = O.hold(st.B, st.N)
    o.kind[own] = O.ATTACK
    o.target[own] = st.N - 1
    area = {"x": u["x"].clone(), "z": u["z"].clone(), "b": u["b"].clone(), "front": torch.full_like(u["x"], 20.0),
            "depth": torch.full_like(u["x"], 20.0)}
    a, b = Dc.Leash(st.B, st.N), Dc.Leash(st.B, st.N)
    oa, ob = a.apply(st, o, area, own), b.apply(sh, o, area, own)
    assert torch.equal(a.on, b.on)
    for f in O.FIELDS:
        assert torch.allclose(getattr(oa, f).float(), getattr(ob, f).float()), f


MATCHUP_SECONDS = 90     # the operative at 0, 15, .., 75; the tacticians every 5 s


def test_the_companions_chain_gives_the_matchup_chains_orders_second_by_second(monkeypatch):
    """The game's chain (companion --chain: Chain with the translator's rules targets / still / aim, the army's
    formula of the operative from 9 units, zone_stick_m) gives, second by second, the units' intended orders of the
    simulator's matchup chain (matchup/duel.Chain, as the night's exam plays it) on M3 (Skaven with guns and
    jezzails on ksl_plains_01): the same raw states, the shooter's own target (the document's 't': the simulator's
    aim_tgt) and its melee contact given as the game gives them."""
    from tools.nn import arena_map
    from tools.nn.matchup import armies as AR
    from tools.nn.matchup import duel as DU
    from tools.nn.operative import cfg as ocfg
    from tools.nn.operative import env as OEN
    from tools.nn.sim import battle
    from tools.nn.tactic import cfg as tcfg
    if not (arena_map.MAPS / "ksl_plains_01" / "meta.json").exists():
        pytest.skip("build/maps/ksl_plains_01 (the captured map) is not here")
    unit_net = fresh_unit_net(1)
    st, card = DU.build("m3", [AR.base("m3")], "cpu", P)
    chain = DU.Chain(1, "formula", command="hold")
    chain.start(st, P, "cpu", card, unit_net)
    rec = {"raw": [], "raw_u": [], "intended": [], "aim": [], "contact": [], "order": []}
    second = LW.Units.second

    def units_second(self, s):
        rec["aim"].append(s.u["aim_tgt"].clone())
        rec["contact"].append(s.u["contact_s"] > 0)
        rec["order"].append((s.u["order_kind"].clone(), s.u["order_target"].clone()))
        lands = second(self, s)
        rec["raw_u"].append(self.ubuf[-1].clone())
        rec["intended"].append(self.intended.clone())
        return lands

    monkeypatch.setattr(LW.Units, "second", units_second)
    seer = TR.Seer(P)
    every, t_every = int(ocfg()["decide_s"]), int(tcfg()["decide_s"])
    K = OEN.rows_of(st.N)
    side = st.u["side"]
    from tools.nn.train import opponents
    for sec in range(MATCHUP_SECONDS):
        raw = seer.raw(st)
        rec["raw"].append(raw.clone())
        if sec % every == 0:
            chain.decide(st, raw, K)
        if sec % t_every == 0:
            chain.tick(st, raw)
        chain.second(st)
        for j in range(round(1.0 / P.dt)):
            o = O.merge(chain.orders(st, j), opponents.nearest(st), side == 2)
            battle.step(st, o, P)
        chain.end_second()
    monkeypatch.setattr(LW.Units, "second", second)
    game = CH.Chain(unit_net, unit_greedy=True, params=P, task="hold")
    game.start("sim", [f"u{i}" for i in range(st.N)], st.keys[0], card[:1], rec["raw"][0], lords=st.u["lord"][:1])
    own = side[0] == 1
    shooter = own & (st.u["range"][0] > 0)
    attacks = 0
    for s in range(MATCHUP_SECONDS):
        _, intended, _ = game.step(rec["raw"][s], s, raw_u=rec["raw_u"][s], aim_tgt=rec["aim"][s],
                                   contact=rec["contact"][s], order=rec["order"][s])
        want = rec["intended"][s]
        for f in O.FIELDS:
            a, b = getattr(intended, f)[0, own], getattr(want, f)[0, own]
            assert torch.allclose(a.float(), b.float(), atol=1e-4), (s, f, a.tolist(), b.tolist())
        attacks += int((want.kind[0, shooter] == O.ATTACK).sum())
    assert game.stats["op_decisions"] == 6 and game.stats["parcel_errors"] == 0
    # the translator's rules fired, so the comparison covers them: the shooters' attacks (aim_rule) or their refusal
    # on a fast side-stepping target (fast_rule; 08.11 the M3 Skaven shoot at Empire cavalry), "stay in the area"
    # (stay_rule); "keep the target" (sticky_rule) keeps an attack in force - none here once fast_rule refuses them
    assert attacks + getattr(game.units, "n_fast", 0) > 0
    assert getattr(game.units, "n_fast", 0) > 0
    assert getattr(game.units, "n_stay", 0) > 0

"""The operative's memory one to one between the simulator and the game, and honest (CLAUDE.md "Вход сети один в
один"; net/operative/memory.py, tmp/memory/design.md): a simulator battle with units that hide (Gutter Runners and
Night Runners: stalk) and the bridge's documents of the same battle (test_tactic_bridge_parity_torch.doc_of_sim)
through bridge/companion/operative_input.py give the same store for either side - the last seen rows (but the known
columns), the counters, the records every 15 s, the summary, the ghosts - and a store fed the simulator's raw state with
every unseen enemy's columns scrambled (all but the public ones: side, alive, vis, the map's handle) is the same bit for
bit. Needs torch."""
import pytest

torch = pytest.importorskip("torch")

from contract import test_tactic_bridge_parity_torch as TP  # noqa: E402
from bridge.companion import operative_input as OPI  # noqa: E402
from net.operative import frame as FR  # noqa: E402
from net.operative import memory as MM  # noqa: E402
from net.tactic import raw as TR  # noqa: E402
from sim import battle, scenario  # noqa: E402
from sim import orders as O  # noqa: E402
from sim import scripts as opponents  # noqa: E402

P = TP.P
RUNNERS = ["wh2_main_skv_cha_warlord_0", "wh2_main_skv_inf_gutter_runners_0", "wh2_main_skv_inf_clanrat_spearmen_0",
           "wh2_main_skv_inf_night_runners_1", "wh2_main_skv_inf_gutter_runners_0"]
PUBLIC = ("side", "alive", "vis", "terr", "tpx", "tpz", "tpr")
SKIP = [i for i, k in enumerate(TR.RAW) if k in TP.NOT_YET or k.startswith(("a_on", "a_cd"))]


def battle_spec():
    s = TP.skirmish(gap=300.0)
    s["sides"][2]["units"] = [{**u, "key": k} for u, k in zip(s["sides"][2]["units"], RUNNERS)]
    return s


def scramble(raw, side, gen):
    """raw with every unseen enemy's columns (but PUBLIC) replaced by noise."""
    I = TR.I
    enemy = (raw[..., I["side"]] > 0) & (raw[..., I["side"]] != side)
    unseen = enemy & (raw[..., I["vis"]] <= 0)
    noise = torch.rand(raw.shape, generator=gen) * 400 - 200
    keep = torch.zeros(raw.shape[-1], dtype=torch.bool)
    keep[[I[k] for k in PUBLIC]] = True
    out = torch.where(unseen[..., None] & ~keep, noise, raw)
    return out, int(unseen.sum())


def same(a, b, skip_last=True):
    for k in a:
        x, y = a[k], b[k]
        if k == "last" and skip_last:
            m = torch.ones(x.shape[-1], dtype=torch.bool)
            m[SKIP] = False
            x, y = x[..., m], y[..., m]
        if x.dtype == torch.bool or x.dtype == torch.long:
            assert torch.equal(x, y), k
        else:
            assert torch.allclose(x, y, atol=2e-3), (k, float((x - y).abs().max()))


def test_the_memory_is_the_same_in_the_game_and_holds_nothing_unseen():
    torch.manual_seed(6)
    st = scenario.build([battle_spec()], P)
    seers = {s: OPI.OperativeSeer(allow_flat=True, params=P, own=s) for s in (1, 2)}
    sim_seer = TR.Seer(P)
    gen = torch.Generator().manual_seed(1)
    stores = {(s, k): MM.Store(1, st.N, s, T=8) for s in (1, 2) for k in ("sim", "game", "noise")}
    axes, unseen, vanished, ghosts = {}, 0, 0.0, 0
    steps = round(1.0 / P.dt)
    two = st.u["side"] == 2
    for sec in range(1, 121):
        for _ in range(steps):
            battle.step(st, O.merge(opponents.nearest(st), opponents.hold_shoot(st), two), P)
        rs = sim_seer.raw(st)[:1]
        for s in (1, 2):
            rg = seers[s].raw(TP.doc_of_sim(st, 0, 1000 * sec, own=s))
            sd = torch.full((1,), s, dtype=torch.long)
            if s not in axes:
                axes[s] = FR.Axes.of(rs, sd, st.bounds)
            noisy, n = scramble(rs, s, gen)
            unseen += n
            for k, r in (("sim", rs), ("game", rg), ("noise", noisy)):
                stores[(s, k)].see(r, float(sec))
            if sec % 15 == 0:
                out = {}
                for k in ("sim", "game", "noise"):
                    m = stores[(s, k)]
                    rec = m.tick(float(sec), axes[s])
                    g_rows, g_mask, g_age = m.ghosts(float(sec))
                    out[k] = {"rec": rec, "sum": m.summary(float(sec), axes[s]), "g_mask": g_mask, "g_age": g_age}
                same(out["sim"], out["game"])
                assert all(torch.equal(out["sim"][k], out["noise"][k]) for k in out["sim"]), "an unseen enemy leaked"
                vanished += float(out["sim"]["rec"][..., MM.RI["vanished"]].sum())
                ghosts += int(out["sim"]["g_mask"].sum())
    for s in (1, 2):
        same(stores[(s, "sim")].state(), stores[(s, "game")].state())
        a, b = stores[(s, "sim")].state(), stores[(s, "noise")].state()
        assert all(torch.equal(a[k], b[k]) for k in a), "an unseen enemy leaked"
    assert unseen > 0 and vanished > 0 and ghosts > 0   # (the battle had hidden enemies, a vanishing, a ghost)

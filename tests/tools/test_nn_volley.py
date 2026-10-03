"""The volley input (observation VOLLEY): the simulator and the companion compute the same value from the
same projectiles-left trajectory, sampled at the network's decisions (1 s), and in the kiting drill it
cycles 0 -> 1 across the volleys (torch: skipped in the project's .venv, runs in the training container)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn.companion import exchange, loop  # noqa: E402
from tools.nn.model import config, policy  # noqa: E402
from tools.nn.model import observation as ob  # noqa: E402
from tools.nn.model import sources  # noqa: E402
from tools.nn.sim import battle, scenario  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402
from tools.nn.train import cadence, scenes  # noqa: E402
from tools.nn.train import drills as D  # noqa: E402

V = ob.INDEX["volley_ready"]
P = load()
K = cadence.Cadence().steps(P.dt)          # simulator steps a decision spans (2 at dt 0.5 s)


def kiting(seeds=range(8), side=None):
    """Kiting drill battles (Night Runners against slower Empire infantry): (State, our side [B], descs)."""
    drill = D.load(["kiting"])["kiting"]
    pairs = [p for p in D.battles(drill, seeds) if side is None or p[1] == side]
    st = scenario.build([p[0] for p in pairs])
    return st, torch.tensor([p[1] for p in pairs]), drill


def live_setup(st):
    """The setup as training has it (scenes.Bank: tensors of LiveSetup.FIELDS, the game's map frame)."""
    setup, _ = sources.from_sim(st)
    setup = ob.Setup(keys=setup.keys, side=setup.side, bounds=np.tile(np.array(sources.CROSSROADS, np.float32),
                     (st.B, 1)), factions=setup.factions, attacker=setup.attacker)
    return scenes.LiveSetup.of(setup, "cpu")


def doc_of(st, move, ours=1):
    """The bridge's state document (src/entries/nn_arena.lua) of battle 0 of the simulator's state: our
    side is the document's side 1 (own_*), the time in ms, `a` the projectiles left."""
    u = {k: v[0].tolist() for k, v in st.u.items()}
    units = []
    for i, key in enumerate(st.keys[0]):
        if not key:
            continue
        own = u["side"][i] == ours
        units.append({"n": f"{'own' if own else 'enemy'}_{i}", "side": 1 if own else 2, "key": key,
                      "x": u["x"][i], "z": u["z"][i], "b": u["b"][i], "men": u["men"][i], "hp": 1.0, "mp": 1.0,
                      "ms": 3, "r": False, "s": False, "w": False, "m": False, "mv": False, "f": False,
                      "a": u["a"][i], "fire": False, "t": "", "fat": "threshold_fresh", "k": 0,
                      "ox": u["x"][i], "oz": u["z"][i], "lf": False, "rf": False, "bf": False, "v": True})
    return {"batch": "vol-1", "move": move, "t": round(float(st.t[0]) * 1000), "done": False,
            "attacker": int(st.attacker[0]) if ours == 1 else 3 - int(st.attacker[0]), "decide_ms": 1000,
            "factions": {"own": "wh2_main_skv_skaven", "enemy": "wh_main_emp_empire"}, "units": units}


def test_the_simulator_and_the_companion_see_the_same_value_on_one_trajectory(monkeypatch):
    pairs = [p for p in D.battles(D.load(["kiting"])["kiting"], range(16)) if p[1] == 1]
    runners = [sum(u["key"].endswith("night_runners_1") for u in p[0]["sides"][1]["units"]) for p in pairs]
    st = scenario.build([pairs[int(np.argmax(runners))][0]])          # one battle, ours side 1, two units if any
    shooters = [i for i, k in enumerate(st.keys[0]) if k and st.u["side"][0, i] == 1]
    assert shooters and all(st.u["range"][0, i] > 0 for i in shooters)
    live = live_setup(st)
    # Projectiles left per simulator step (0.5 s): a volley at 2.5 s (between two decisions), a straggler
    # at 4 s, then 14 s and 20.5 s; the second unit (if any) shoots once at 6 s. At the decisions (whole
    # seconds) the companion reads the same numbers, as the bridge's ammo_left() each second.
    steps = 2 * 26
    a0 = float(st.u["a"][0, shooters[0]])
    falls = {5: 120, 8: 3, 28: 110, 41: 115}
    traj, a = [], a0
    for j in range(steps):
        a -= falls.get(j, 0)
        traj.append(a)
    second = [a0 if j < 12 else a0 - 100 for j in range(steps)]
    # the companion: its own decision path (loop.Brain), the observation taken by a spy
    seen = []
    observe = ob.observe

    def spy(*args, **kw):
        obs, mem = observe(*args, **kw)
        seen.append(obs)
        return obs, mem
    monkeypatch.setattr(ob, "observe", spy)
    torch.manual_seed(0)
    brain = loop.Brain(policy.Actor(config.preset("small", d=32, layers=1, heads=2, critic_d=32, critic_layers=1,
                                                  critic_heads=2)).eval(), greedy=True)
    sim_vals, game_vals = [], []
    mem = None
    for j in range(steps):
        st.t[:] = j * P.dt
        st.u["a"][0, shooters[0]] = traj[j]
        if len(shooters) > 1:
            st.u["a"][0, shooters[1]] = second[j]
        if j % K:
            continue                                   # a physics step between decisions: nobody looks
        obs, mem = observe(st.observation(), live, 1, mem)                         # the simulator (rollout)
        sim_vals.append(obs.tokens[0, shooters, V].tolist())
        doc = doc_of(st, move=j // K)
        brain.decide(doc)                                                           # the companion
        names = brain.battle.names
        game_vals.append([float(seen[-1].tokens[0, names.index(f"own_{i}"), V]) for i in shooters])
    assert np.allclose(np.array(sim_vals), np.array(game_vals), atol=1e-6, rtol=0)
    want = []                                      # by hand: the passport's reload 8 s, falls seen at 3, 4, 14, 21 s
    last = None
    for d in range(steps // K):
        if d in (3, 4, 14, 21):
            last = d
        want.append(1.0 if last is None else min(1.0, (d - last) / 8))
    assert [v[0] for v in sim_vals] == pytest.approx(want, abs=1e-6)
    if len(shooters) > 1:
        assert [v[1] for v in sim_vals] == pytest.approx([1.0] * 6 + [min(1.0, (d - 6) / 8)
                                                                      for d in range(6, steps // K)], abs=1e-6)


def test_in_the_kiting_drill_the_input_cycles_across_the_volleys():
    """The skilled script (shoot, run back, halt, shoot) in the simulator at the network's cadence: our
    Night Runners' input is 1 before the first volley, 0 at each decision their projectiles fell since the
    previous one, then rises by 1 / reload a decision while they run; it goes 0 -> 1 again and again."""
    st, ours, drill = kiting(seeds=range(4))
    live = live_setup(st)
    mem = {1: None, 2: None}
    vals, ammo, times = [], [], []
    for _ in range(200):
        state = st.observation()
        rows = []
        for s in (1, 2):
            obs, mem[s] = ob.observe(state, live, s, mem[s])
            rows.append(obs.tokens[..., V])
        mine = torch.where((st.u["side"] == ours[:, None]), torch.where(ours[:, None] == 1, rows[0], rows[1]),
                           torch.zeros_like(rows[0]))
        vals.append(mine.clone())
        ammo.append(st.u["a"].clone())
        times.append(st.t.clone())
        if bool(st.done.all()):
            break
        for _ in range(K):
            orders = D.merged(st, ours, drill.skilled(st), drill.enemy(st))
            battle.step(st, orders, P, P.dt)
    vals, ammo, times = torch.stack(vals), torch.stack(ammo), torch.stack(times)   # [T, B, N], [T, B]
    shooter = (st.u["side"] == ours[:, None]) & (st.u["range"] > 0)
    assert shooter.any()
    reload = live.arrays.reload
    cycles = 0
    for b, n in shooter.nonzero().tolist():
        v, a, t = vals[:, b, n], ammo[:, b, n], times[:, b]
        assert v[0] == 1
        fell = torch.cat([torch.zeros(1, dtype=torch.bool), a[1:] < a[:-1]])
        assert torch.all(v[fell] == 0)
        rising = ~fell[1:] & (t[1:] > t[:-1])           # (an ended battle's clock stands)
        step = (v[1:] - v[:-1])[rising & (v[:-1] < 1)]
        want = torch.minimum(1 - v[:-1], torch.full_like(v[:-1], 1.0 / float(reload[b, n])))[rising & (v[:-1] < 1)]
        assert torch.allclose(step, want, atol=1e-5)
        # a cycle: from 0 back up to 1
        low = False
        for x in v.tolist():
            if x == 0:
                low = True
            elif x == 1 and low:
                cycles, low = cycles + 1, False
    assert cycles >= 2 * int(shooter.sum())          # every unit reloads fully between volleys, twice or more
    assert torch.all(vals[:, ~shooter] == 0)          # no missile, or not ours: 0

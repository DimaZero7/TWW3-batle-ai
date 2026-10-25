"""tools/nn/companion/general.py: the general (S28) in the game's companion - the same plans and tasks as the
simulator's Runtime on the same battle, re-plans on the battle's time and their events, the log, and a companion
without a general as it was (docs/en/training/general.md "In the game"). torch: the training container, on the CPU."""
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn.companion import exchange, loop                      # noqa: E402
from tools.nn.companion import general as cg                        # noqa: E402
from tools.nn.general import config as gconfig, executor            # noqa: E402
from tools.nn.general import army_input as ai                  # noqa: E402
from tools.nn.general import runtime as gruntime                    # noqa: E402
from tools.nn.model import config, policy                           # noqa: E402
from tools.nn.model import observation as ob                        # noqa: E402
from tools.nn.model.decide import frame_to                          # noqa: E402
from tools.nn.model.sources import CROSSROADS                       # noqa: E402
from tools.nn.sim import battle, scenario                           # noqa: E402
from tools.nn.sim.params import load                                # noqa: E402
from tools.nn.sim.state import FATIGUE_LEVELS                       # noqa: E402
from tools.nn.train import opponents                                # noqa: E402

TINY = config.preset("v2", d=32, layers=2, heads=2, pointer=16, critic_d=32, critic_layers=1, critic_heads=2,
                     sector_d=16)
GCFG = gconfig.GeneralConfig(card=16, reader_hidden=16, council_heads=2)
LORD, SPEAR = "wh_main_emp_cha_general_0", "wh_main_emp_inf_spearmen_0"


def general_actor(seed=0, task_w=0.0, sharp=1.0, cards_w=0.0):
    """A tiny executor with a general. task_w: the task input's weights (0: the unit network's to the bit); sharp: the
    group collection's keys and queries x this (a fresh general splits every unit ~evenly over the places, 0.1 each:
    its main place is then a tie that float noise flips); cards_w: the weights of the group cards' fight and tiles
    layers (net.General fight_in, gtile_in; 0: as they start)."""
    torch.manual_seed(seed)
    actor = executor.wrap(policy.Actor(TINY).eval(), GCFG).eval()
    with torch.no_grad():
        if task_w:
            actor.task_in.weight.normal_(0, task_w)
        if cards_w:
            actor.general.fight_in.weight.normal_(0, cards_w)
            actor.general.gtile_in.weight.normal_(0, cards_w)
        actor.general.groups.k.weight.mul_(sharp)
        actor.general.groups.q.weight.mul_(sharp)
    return actor


# --- the simulator's side: Runtime on a rollout-like view of the simulator's own state ------------------------------
class SimEnv:
    """What rollout.Battles gives Runtime, for one simulator battle (B = 1, the simulator's slots): the State itself,
    the bounds, the side's observation memory, side 1 the learner."""

    def __init__(self, st, setup):
        self.st, self.B, self.N, self.device = st, 1, st.N, torch.device("cpu")
        self.setup = SimpleNamespace(bounds=torch.as_tensor(setup.bounds, dtype=torch.float32))
        self.ctrl = torch.tensor([[gruntime.LEARNER, cg.SCRIPT]])
        self.past, self.battle_slot, self.rows_learn = [], torch.zeros(1, dtype=torch.long), torch.tensor([0])
        self.mem = {1: SimpleNamespace(frame=None), 2: SimpleNamespace(frame=None)}


def army_of(seed):
    from tools.nn.armies import generate
    army = scenario.from_arena(generate.battle(seed), own_ai="defend")
    army["sides"][1]["ai"] = False
    return army


def doc_of(st, names, slots, keys, sides, move, factions, batch="b-1"):
    """The state document the bridge writes, from the simulator's state: a unit gone off the map has no position."""
    u = {k: v[0] for k, v in st.u.items()}
    name_of = {int(s): n for n, s in zip(names, slots)}
    rows = []
    for n, s, k, sd in zip(names, slots, keys, sides):
        f = (lambda key: float(u[key][s]))
        b = (lambda key: bool(u[key][s]))
        tg = int(u["target"][s])
        gone = b("gone")
        rows.append({"n": n, "side": int(sd), "key": k, "x": None if gone else f("x"), "z": None if gone else f("z"),
                     "b": f("b"), "men": f("men"), "hp": f("hp"), "mp": f("mp"), "ms": f("ms"),
                     "r": b("r") and not b("s"), "s": b("s"), "w": b("w"), "m": b("m"), "mv": b("mv"), "f": b("f"),
                     "a": f("a"), "fire": b("fire"), "t": name_of.get(tg, "") if tg >= 0 else "", "k": f("k"),
                     "ox": f("ox"), "oz": f("oz"), "fat": FATIGUE_LEVELS[int(u["fat"][s])], "lf": b("lf"),
                     "rf": b("rf"), "bf": b("bf"), "v": b("vis")})
    return {"batch": batch, "move": move, "t": float(st.t[0]) * 1000, "done": False, "attacker": int(st.attacker[0]),
            "factions": factions, "units": rows, "abilities_used": {}}


SEED = 1000900014


def parity(actor, seconds=240.0, seed=SEED, greedy=True, events=True):
    """A simulator battle (ai_like on both sides), at every decision: the simulator's Runtime on its own state (rollout's
    order: the deployment's plan at the start, then the observation, the re-plans due on it, the task) and the
    companion's general on the bridge's document of the same moment have the same plan (every unit's group, our
    groups' kinds and targets, the plan's time) and the same task of every unit (sampling: the same random draws).
    events: the enemy's lord falls at 75 s, our biggest group routs at 140 s (in the simulator's state: both views see
    them). -> the counts."""
    params = load()
    army = army_of(seed)
    st = scenario.build([army], params)
    slot = scenario.slots(army, st.N // 2)
    units = army["sides"][1]["units"] + army["sides"][2]["units"]
    names = [u["name"] for u in units]
    slots = np.array([slot[n] for n in names])
    keys = [u["key"] for u in units]
    sides = [1] * len(army["sides"][1]["units"]) + [2] * len(army["sides"][2]["units"])
    factions = {"own": army["sides"][1]["faction"], "enemy": army["sides"][2]["faction"]}
    gcfg = cg.config(actor)
    assert gcfg.deploy == "script"
    G = gcfg.places
    setup_sim = ob.Setup(keys=[list(st.keys[0])], side=st.u["side"].numpy(), bounds=np.asarray([CROSSROADS]),
                         factions=[(factions["own"], factions["enemy"])], attacker=st.attacker.numpy())
    env = SimEnv(st, setup_sim)
    rt = gruntime.Runtime(env, actor.cfg, gcfg)
    rt.learner, rt.greedy, rt.pending = actor, greedy, None
    gen = cg.GameGeneral(actor, gcfg, greedy=greedy, side=1, params=params)
    seen = {"move": 0, "mem_s": None, "mem_g": None, "battle": None, "plans": [], "plans_s": 0, "tasks": 0,
            "reasons": set(), "max_err": 0.0, "kinds": set(), "fight": torch.zeros(2), "state": torch.zeros(4)}
    sl = torch.as_tensor(slots)
    lord2 = int(slots[[i for i, n in enumerate(names) if sides[i] == 2 and n.endswith("lord")][0]])

    def check(s):
        seen["move"] += 1
        if events and seen["move"] == 75:
            s.u["men"][0, lord2] = 0.0
            s.u["hp"][0, lord2] = 0.0
            s.lord_dead_s[0, 1] = 0.0               # (as the simulator's step that kills him marks it)
        if events and seen["move"] == 140:
            main = rt.r["main"][0]
            big = int(torch.bincount(main[(main >= 0) & (main < G)]).argmax())
            for j in (main == big).nonzero().flatten().tolist():
                s.u["ms"][0, j], s.u["r"][0, j] = 6.0, True
        # the simulator's runtime
        obs_s, seen["mem_s"] = ob.observe(s.observation(), setup_sim, 1, seen["mem_s"])
        env.mem[1].frame = env.mem[2].frame = seen["mem_s"].frame
        n0 = rt.stats["plans"]
        torch.manual_seed(1000 + seen["move"])
        if seen["move"] == 1:
            rt.start_battles(env, torch.ones(1, dtype=torch.bool))
        ot = policy.to_torch(obs_s)
        rt.replan(env, {k: torch.cat([ot[k], ot[k]]) for k in ("tokens", "own", "attend", "pos")})
        task_s = rt.task(env.rows_learn, ot, frame_to(obs_s.frame, "cpu"))
        seen["plans_s"] += int(rt.stats["plans"] - n0)
        # the companion's general on the bridge's document of the same moment
        doc = doc_of(s, names, slots, keys, sides, seen["move"], factions)
        if seen["battle"] is None:
            seen["battle"] = exchange.battle(doc)
        b = seen["battle"]
        state = exchange.arrays(doc, b.names, b.slots)
        obs_g, seen["mem_g"] = ob.observe(state, b.setup, 1, seen["mem_g"])
        torch.manual_seed(1000 + seen["move"])
        task_g, recs = gen.decide(b, state, obs_g, seen["mem_g"])
        seen["plans"] += recs
        seen["reasons"] |= {r["reason"] for r in recs}
        seen["kinds"] |= {g["kind"] for r in recs for g in r["groups"]}
        # the same plan: groups of every unit, kinds and targets of our groups
        rg, rs = gen.rt.r, rt.r
        assert torch.equal(rg["main"][0], rs["main"][0][sl]), seen["move"]
        act = rs["active"][0][:G]
        assert torch.equal(rg["active"][0], rs["active"][0]), seen["move"]
        for k in ("kind", "target"):
            assert torch.equal(rg[k][0][act], rs[k][0][act]), (k, seen["move"])
        assert float(rg["t_plan"][0]) == pytest.approx(float(rs["t_plan"][0]))
        # the same fight flags of every unit at the plan (net.FIGHT: the game's melee and firing flags and target)
        assert torch.equal(rg["in/fight"][0], rs["in/fight"][0][sl]), seen["move"]
        seen["fight"] += rs["in/fight"][0].sum(0)
        # the same slots' state (army_input STATE: morale, wavering, fatigue, ammunition - from the bridge's mp, ms, fat, a)
        cols = [ai.INDEX[k] for k in ai.STATE]
        ng, ns = rg["in/num"][0][:, cols], rs["in/num"][0][sl][:, cols]
        torch.testing.assert_close(ng, ns, rtol=0, atol=1e-6, msg=f"move {seen['move']}")
        seen["state"] = torch.maximum(seen["state"], torch.stack([(ns[:, 0] < 0.5).any(), (ns[:, 1] > 0).any(),
                                                                  (ns[:, 2] > 0).any(),
                                                                  ((ns[:, 3] > 0) & (ns[:, 3] < 1)).any()]).float())
        # the same task, unit by unit
        assert torch.equal(task_g["task_on"][0], task_s["task_on"][0][sl])
        err = float((task_g["task"][0].double() - task_s["task"][0][sl].double()).abs().max())
        seen["max_err"] = max(seen["max_err"], err)
        assert err < 1e-4, (seen["move"], err)
        seen["tasks"] += int(task_g["task_on"][0].sum())
        return opponents.ai_like(s)

    battle.run(st, check, params, until_s=seconds, every_s=1.0, compile=False)
    seen["own"] = set(names[:len(army["sides"][1]["units"])])
    return seen


def test_the_game_general_plans_and_tasks_as_the_simulators_runtime():
    # (a fresh general splits every unit ~evenly over the places: sharp makes its choices clear of float noise)
    seen = parity(general_actor(task_w=0.5, sharp=30.0, cards_w=0.5))
    plans = seen["plans"]
    # the plan at the start, the timer every 30 s of battle, the lord's fall, the broken group and the enemy come near
    # a group in between
    assert plans[0]["reason"] == "deploy" and len(plans) == seen["plans_s"] >= 9, (len(plans), seen["plans_s"])
    assert seen["reasons"] == {"deploy", "timer", "lord", "broken", "near"}, seen["reasons"]
    assert seen["tasks"] > 1000 and seen["max_err"] < 1e-4
    assert all(set(g["units"]) <= seen["own"] for p in plans for g in p["groups"])
    # some unit under fire and some in melee at the plans (the fight's numbers are read in both)
    assert float(seen["fight"][0]) > 0 and float(seen["fight"][1]) > 0, seen["fight"]
    # the slots' state moved off a fresh unit's at the plans: morale fell, wavering, tired, ammunition spent
    assert seen["state"].tolist() == [1, 1, 1, 1], seen["state"]


# --- the re-plans on the battle's time and its events, on the bridge's documents --------------------------------------
def unit(name, side, key, x, z=0.0, **fields):
    row = {"n": name, "side": side, "key": key, "x": x, "z": z, "b": 90 if side == 1 else 270, "men": 100,
           "hp": 1.0, "mp": 1.0, "ms": 2, "r": False, "s": False, "w": False, "m": False, "mv": False,
           "f": False, "a": 0, "fire": False, "t": "", "fat": "threshold_fresh", "k": 0, "ox": x, "oz": z,
           "lf": False, "rf": False, "bf": False, "v": True}
    row.update(fields)
    return row


def doc(t_s, move, batch="b-1", own=None, enemy=None):
    own, enemy = own or {}, enemy or {}
    units = [unit("own_lord", 1, LORD, -235.0, **dict({"men": 1}, **own.get("own_lord", {})))]
    units += [unit(f"own_spear_{i}", 1, SPEAR, -175.0, z=40.0 * (i - 2), **own.get(f"own_spear_{i}", {}))
              for i in range(1, 4)]
    units += [unit("enemy_lord", 2, LORD, 235.0, **dict({"men": 1}, **enemy.get("enemy_lord", {})))]
    units += [unit(f"enemy_spear_{i}", 2, SPEAR, 175.0, z=40.0 * (i - 2)) for i in range(1, 4)]
    return {"batch": batch, "move": move, "t": int(t_s * 1000), "done": False, "attacker": 2, "decide_ms": 1000,
            "factions": {"own": "wh_main_emp_empire", "enemy": "wh_main_emp_empire"}, "units": units}


def test_replans_come_on_the_battles_time_and_on_their_events():
    brain = loop.Brain(general_actor(), greedy=True, general=cg.config(general_actor()))
    got = []
    # the game at x5: the states come a second of battle apart however fast they come; the plan's clock is the
    # battle's (the state's t), never the wall clock
    for move, t in enumerate(range(0, 36), start=1):
        brain.decide(doc(t, move))
        got += [(t, p["reason"]) for p in brain.plans]
    assert got == [(0, "deploy"), (30, "timer")]
    # the enemy's lord falls at 40 s: a re-plan now (10 s after the last); once
    brain.decide(doc(40, 37, enemy={"enemy_lord": {"men": 0}}))
    assert [p["reason"] for p in brain.plans] == ["lord"] and brain.plans[0]["t"] == 40.0
    brain.decide(doc(41, 38, enemy={"enemy_lord": {"men": 0}}))
    assert brain.plans == []
    # our three spearmen rout at 42 s: a group is broken (more than half of it) - but not sooner than 3 s after the
    # last plan; at 44 s it is due
    rout = {f"own_spear_{i}": {"ms": 6, "r": True} for i in range(1, 4)}
    brain.decide(doc(42, 39, own=rout, enemy={"enemy_lord": {"men": 0}}))
    assert brain.plans == []
    brain.decide(doc(44, 40, own=rout, enemy={"enemy_lord": {"men": 0}}))
    assert [p["reason"] for p in brain.plans] == ["broken"]
    # the routing take no part in the groups
    assert all(not u.startswith("own_spear") for g in brain.plans[0]["groups"] for u in g["units"])
    # a new battle: its own plan from the start
    brain.decide(doc(0, 1, batch="b-2"))
    assert [(p["n"], p["reason"]) for p in brain.plans] == [(1, "deploy")]


def test_a_plan_record_names_the_groups_their_units_kinds_and_targets():
    brain = loop.Brain(general_actor(), greedy=True, general=cg.config(general_actor()))
    brain.decide(doc(0, 1))
    (rec,) = brain.plans
    own = {f"own_spear_{i}" for i in range(1, 4)} | {"own_lord"}
    assert rec["n"] == 1 and rec["t"] == 0.0 and rec["groups"]
    assert set().union(*(g["units"] for g in rec["groups"])) == own                 # every own unit in one group
    assert set().union(*(e["units"] for e in rec["enemy"])) <= {u.replace("own", "enemy") for u in own}
    for g in rec["groups"]:
        assert 0 <= g["kind"] < GCFG.kinds and g["g"].startswith("g")
        assert g["target"].startswith(("g", "e", "tile ")) and g["target"] != g["g"]
        assert g["kind_name"] == gconfig.MISSION_KINDS[g["kind"]].name
    line = cg.describe(rec)
    assert line.startswith("general plan 1 t 0.0 s (deploy):") and "kind" in line


def test_the_loop_logs_every_plan(tmp_path):
    lines = []
    brain = loop.Brain(general_actor(), greedy=True, general=cg.config(general_actor()))
    worker = threading.Thread(target=loop.run, args=(tmp_path, brain),
                              kwargs={"exit_on_done": True, "out": lines.append, "idle_exit_s": 30,
                                      "log": tmp_path / "companion.jsonl"})
    worker.start()
    exchange.write_atomic(tmp_path / exchange.STATE, json.dumps(dict(doc(0, 1), batch="b-2")))
    for _ in range(4000):
        if (tmp_path / exchange.ORDERS).exists():
            break
        threading.Event().wait(0.005)
    exchange.write_atomic(tmp_path / exchange.STATE, json.dumps(dict(doc(1, 2), batch="b-2", done=True)))
    worker.join(30)
    assert not worker.is_alive()
    assert any(ln.startswith("general plan 1 t 0.0 s (deploy)") for ln in lines), lines
    rows = [json.loads(ln) for ln in (tmp_path / "companion.jsonl").read_text().splitlines()]
    assert rows[0]["general"][0]["reason"] == "deploy" and rows[0]["general"][0]["groups"]


# --- without a general: the companion as it was --------------------------------------------------------------------------
def test_without_a_general_the_companion_is_as_it_was():
    plain = policy.Actor(TINY).eval()
    assert cg.config(plain) is None                       # a checkpoint without a general: no general
    assert loop.Brain(plain).general is None
    # a general whose task input is still at zero: the executor plays the unit network's orders to the bit (the task's
    # masks off: with them the untrained general's random plan may cut a unit's choices - which plan depends on the
    # random draws of the actor's shape: the token's 19 columns of 25.10.2026 made seed 3's lord a holder)
    import dataclasses
    torch.manual_seed(3)
    base = policy.Actor(TINY).eval()
    wrapped = executor.wrap(base, dataclasses.replace(GCFG, task_mask=False)).eval()
    a = loop.Brain(base, greedy=True)
    b = loop.Brain(wrapped, greedy=True, general=cg.config(wrapped))
    for move, t in enumerate(range(0, 5), start=1):
        oa, _, ua = a.decide(doc(t, move))
        ob_, _, ub = b.decide(doc(t, move))
        assert oa == ob_ and ua == ub
    assert b.general.rt is not None and b.general.plans == 1


def test_the_games_general_settings():
    actor = general_actor()
    g = cg.config(actor, cg.parse_sets(["replan_s=20", "replan-min-s=5"]))
    assert g.deploy == "script" and g.replan_s == 20.0 and g.replan_min_s == 5.0 and g.card == GCFG.card
    with pytest.raises(ValueError):
        cg.config(actor, {"deploy": "net"})               # the bridge does not move units
    with pytest.raises(ValueError):
        cg.config(actor, {"nonsense": "1"})


def test_a_unit_without_a_passport_plays_the_battle_without_the_general(monkeypatch):
    brain = loop.Brain(general_actor(), greedy=True, general=cg.config(general_actor()))
    # (the observation needs every passport too: a key the general's unit table lacks, as if it did)
    monkeypatch.setattr(gruntime, "unit_table", lambda: SimpleNamespace(index={"": 0, LORD: 1}))
    orders, _, _ = brain.decide(doc(0, 1))
    assert brain.plans and "off" in brain.plans[0] and cg.describe(brain.plans[0]).startswith("general off")
    assert len(orders) == 4

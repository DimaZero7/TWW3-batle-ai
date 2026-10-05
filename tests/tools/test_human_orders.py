"""A battle a human plays: the human build (tools/build.py human) and the orders inferred from the
recording (tools/nn/human_orders.py) counted by the gate's liveliness. Levels 1-2: no game."""
import json

import pytest

from tools import build
from tools import config as project
from tools.nn import gate, human_orders
from tools.nn import scenario as nn_scenario


def unit(n, side=1, **kw):
    row = {"n": n, "side": side, "x": 0.0, "z": 0.0, "ox": 0.0, "oz": 0.0, "men": 100, "r": False, "s": False,
           "m": False, "mv": False, "t": ""}
    row.update(kw)
    return row


def sample(t_s, *units):
    return {"event": "nn_sample", "t": int(t_s * 1000), "units": list(units)}


class TestInfer:
    def test_a_moved_point_is_a_move_a_target_an_attack_a_point_at_the_feet_a_hold(self):
        enemy = unit("e", side=2, x=50.0, z=0.0, ox=50.0, oz=0.0)
        rows = [sample(0, unit("a"), enemy),
                sample(1, unit("a", ox=30.0, oz=0.0, mv=True), enemy),                  # move
                sample(2, unit("a", x=5.0, ox=30.5, oz=0.0, mv=True), enemy),           # under 1 m: same order
                sample(3, unit("a", x=10.0, ox=45.0, oz=0.0, t="e", mv=True), enemy),   # attack e
                sample(4, unit("a", x=12.0, ox=13.0, oz=0.0), enemy)]                   # halt where it stands
        orders = human_orders.infer(rows)
        assert [(t, [(o["u"], o["k"], o.get("tg")) for o in os]) for t, os in orders] == [
            (1000, [("a", "move", None)]), (3000, [("a", "attack", "e")]), (4000, [("a", "hold", None)])]
        assert orders[0][1][0]["x"] == 30.0 and orders[0][1][0]["status"] == "given"

    def test_the_start_a_rally_and_the_other_side_are_not_orders(self):
        rows = [sample(0, unit("a", ox=10.0), unit("e", side=2, ox=0.0)),
                sample(1, unit("a", ox=80.0, r=True), unit("e", side=2, ox=90.0)),   # routing: the engine's point
                sample(2, unit("a", ox=-40.0), unit("e", side=2, ox=-90.0)),          # rallied: its first second
                sample(3, unit("a", ox=-40.5), unit("e", side=2, ox=0.0))]
        assert human_orders.infer(rows) == []


def write_events(path, rows):
    path.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows), encoding="utf-8")


class TestLiveliness:
    def test_inferred_orders_are_counted_as_the_gate_counts_the_networks(self, tmp_path):
        rows = [sample(t, unit("a", ox=float(20 * (t // 5)), mv=True), unit("e", side=2)) for t in range(0, 61)]
        events = tmp_path / "events.jsonl"
        write_events(events, rows)
        lively, n = human_orders.inferred_liveliness(events)
        assert n == 12                                       # a new point 20 m on every 5 s, from 5 s to 60 s
        # The same orders given by the network (nn_orders) count the same.
        net = rows[:]
        for t in range(5, 61, 5):
            net.append({"event": "nn_orders", "t": t * 1000,
                        "orders": [{"u": "a", "k": "move", "x": float(20 * (t // 5)), "z": 0.0, "status": "given"}]})
        net.sort(key=lambda r: (r["t"], r["event"] == "nn_orders"))
        write_events(tmp_path / "net.jsonl", net)
        real = gate.liveliness(tmp_path / "net.jsonl")
        assert lively["net"]["changes"] == real["net"]["changes"] == 12
        assert lively == real
        rates = gate.lively_rates(lively["net"])
        assert rates["order_changes_per_min"] == pytest.approx(12.0)

    def test_compare_puts_the_human_next_to_the_networks_same_battle_and_gate(self, tmp_path):
        run = tmp_path / "human" / "20261005-200000"
        run.mkdir(parents=True)
        army = {"seed": 7, "swap": True}
        (run / "manifest.json").write_text(json.dumps({"config": {"own_role": "defend", "army": army}}))
        rows = [sample(t, unit("a", ox=float(20 * (t // 10))), unit("e", side=2)) for t in range(0, 61)]
        write_events(run / "events.jsonl", rows)
        net_run = tmp_path / "runs" / "net1"
        net_run.mkdir(parents=True)
        write_events(net_run / "events.jsonl", rows)
        gates = tmp_path / "gates"
        for name, battles in (("20261001-000000", [{"seed": 7, "swap": True, "role": "defend", "run": "net1",
                                                    "lively": {"net": {}}}]),
                              ("20261002-000000", [{"seed": 9, "swap": False, "role": "attack", "run": "x",
                                                    "lively": {"net": {}}}])):
            (gates / name).mkdir(parents=True)
            (gates / name / "summary.json").write_text(json.dumps({
                "checkpoint": name, "battles": battles,
                "liveliness": {"net": {"defend": {"order_changes_per_min": 3.0}}}}))
        newest, same = human_orders.gate_reference("defend", army, gates=gates, runs=tmp_path / "runs")
        assert newest[0].name == "20261001-000000"           # the newest gate with a defending battle
        assert same[2] == net_run
        out = human_orders.compare(run, gate_dir=gates / "20261001-000000", gates=gates, runs=tmp_path / "runs")
        assert out["role"] == "defend"
        assert out["columns"]["human (inferred)"]["order_changes_per_min"] == pytest.approx(6.0)   # 6 in 60 s
        assert out["columns"]["net same battle (inf.)"] == out["columns"]["human (inferred)"]
        assert out["columns"]["net gate defend (real)"] == {"order_changes_per_min": 3.0}
        assert any("order changes" in line for line in human_orders.table(out["columns"]))


class TestBuild:
    def test_the_human_target_builds_the_gates_battle_16_at_x1(self, tmp_path, monkeypatch):
        monkeypatch.setattr(project, "BUILD", tmp_path)
        monkeypatch.setattr(nn_scenario, "SCENARIO", tmp_path / "nn_arena.xml")
        written = []
        monkeypatch.setattr(build, "build", lambda target, config, scenario=None: written.append((target, config, scenario)) or {})
        seed = 1_000_900_014
        assert build.main(["human", "--army-seed", str(seed), "--army-swap"]) == 0
        (target, config, scenario), = written
        plan = gate.plan(2, offset=14)[1]
        assert (plan["battle"], plan["seed"], plan["swap"], plan["role"]) == (16, seed, True, "defend")
        assert target == "human" and config["own_ai"] == "human" and config["own_role"] == plan["role"]
        assert config["factions"] == plan["factions"] and len(config["units"]["own"]) == plan["own_units"]
        assert (config["speed"], config["timeout_ms"], config["soldiers_every"]) == (1, 3_600_000, 5)
        assert scenario == str(tmp_path / "human" / f"random_{seed}_swap.xml")
        assert "<timeout_winning_alliance_index>0</timeout_winning_alliance_index>" in \
            (tmp_path / "human" / f"random_{seed}_swap.xml").read_text(encoding="utf-8")
        assert not (tmp_path / "nn_arena.xml").exists()

    def test_the_human_target_takes_no_other_commander(self, tmp_path, monkeypatch):
        monkeypatch.setattr(project, "BUILD", tmp_path)
        monkeypatch.setattr(build, "build", lambda *a, **k: {})
        with pytest.raises(SystemExit):
            build.main(["human", "--army-seed", "5", "--own-ai", "net"])

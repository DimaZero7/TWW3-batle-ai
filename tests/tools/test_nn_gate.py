"""The in-game gate: nn-arena builds of generated battles (--army-seed, --own-role) and
tools/nn/gate.py (the plan, the outcome of a run, the summary). Levels 1-2: no game."""
import json

import pytest

from tools import build
from tools import config as project
from tools.nn import gate
from tools.nn import scenario as nn_scenario
from tools.nn.armies import generate


def capture(monkeypatch, tmp_path):
    monkeypatch.setattr(project, "BUILD", tmp_path)
    monkeypatch.setattr(nn_scenario, "SCENARIO", tmp_path / "nn_arena.xml")
    written = []
    monkeypatch.setattr(build, "build", lambda target, config, scenario=None: written.append((config, scenario)) or {})
    return written


class TestBuild:
    @pytest.mark.parametrize("role,enemy_role,timeout_winner,attacker", [("attack", "defend", 1, 1),
                                                                         ("defend", "attack", 0, 2)])
    def test_a_generated_battle_under_the_network(self, role, enemy_role, timeout_winner, attacker, tmp_path,
                                                   monkeypatch):
        written = capture(monkeypatch, tmp_path)
        seed = 1_000_900_008
        assert build.main(["nn-arena", "--army-seed", str(seed), "--own-role", role, "--timeout", "3600"]) == 0
        (config, scenario), = written
        assert config["own_ai"] == "net" and config["own_role"] == role and config["enemy_role"] == enemy_role
        assert config["arena"] == f"random_{seed}" and config["timeout_ms"] == 3_600_000
        assert config["army"]["seed"] == seed and config["army"]["split"] == "eval"
        arena = generate.battle(seed)
        assert config["factions"] == {s: arena["sides"][s]["faction"] for s in ("own", "enemy")}
        assert [u["script_name"] for u in config["units"]["own"]] == [
            f"own_{u['slot']}" for u in arena["sides"]["own"]["units"]]
        assert len(config["units"]["enemy"]) == 20 and config["army"]["men"]["enemy"] > 1000
        # The battle file goes next to the build, not into scenarios/; the game's own limit is past ours.
        path = tmp_path / "nn-arena" / f"random_{seed}.xml"
        assert scenario == str(path) and not (tmp_path / "nn_arena.xml").exists()
        xml = path.read_text(encoding="utf-8")
        assert f"<timeout_winning_alliance_index>{timeout_winner}</timeout_winning_alliance_index>" in xml
        assert "<duration>3660</duration>" in xml and xml.count("<general>") == 2
        assert xml.count("<unit ") == len(config["units"]["own"]) + len(config["units"]["enemy"])
        # What the Lua side puts into the companion's state: attacker 1 when our side attacks.
        assert (1 if config["enemy_role"] == "defend" else 2) == attacker

    def test_the_swapped_battle_gives_our_side_the_other_army(self, tmp_path, monkeypatch):
        written = capture(monkeypatch, tmp_path)
        seed = 1_000_900_008
        assert build.main(["nn-arena", "--army-seed", str(seed), "--army-swap", "--own-role", "defend"]) == 0
        (config, scenario), = written
        arena = generate.battle(seed)
        assert config["army"]["swap"] is True and config["arena"] == f"random_{seed}_swap"
        assert config["factions"] == {"own": arena["sides"]["enemy"]["faction"], "enemy": arena["sides"]["own"]["faction"]}
        assert [u["key"] for u in config["units"]["own"]] == [u["key"] for u in arena["sides"]["enemy"]["units"]]
        assert config["army"]["cost"] == {"own": arena["sides"]["enemy"]["cost"], "enemy": arena["sides"]["own"]["cost"]}
        assert scenario.endswith(f"random_{seed}_swap.xml")
        with pytest.raises(SystemExit):
            build.main(["nn-arena", "--army-swap"])

    def test_the_build_reads_a_battle_file_outside_scenarios(self, tmp_path, monkeypatch):
        monkeypatch.setattr(project, "BUILD", tmp_path)
        xml = tmp_path / "random_1.xml"
        xml.write_text("<battle/>", encoding="utf-8")
        manifest = build.build("nn-arena", {"speed": 20, "timeout_ms": 60000, "tick_ms": 1000, "scenario": "x"},
                               dependencies=[], scenario=str(xml))
        assert manifest["syntax_checked"] is True and manifest["scenario"].endswith("nn_arena.xml")

    def test_defaults_and_wrong_combinations(self, tmp_path, monkeypatch):
        written = capture(monkeypatch, tmp_path)
        assert build.main(["nn-arena", "--own-ai", "net"]) == 0
        assert written[-1][0]["own_role"] == "defend" and written[-1][0]["enemy_role"] == "attack"
        assert written[-1][1] is None                      # the named arenas still use scenarios/
        assert build.main(["nn-arena"]) == 0
        assert written[-1][0]["own_ai"] == "attack" and "own_role" not in written[-1][0]
        for bad in (["--own-ai", "attack", "--own-role", "defend"], ["--army-seed", "5", "--arena", "whole_emp_v_skv"],
                    ["--timeout", "3601"]):
            with pytest.raises(SystemExit):
                build.main(["nn-arena", *bad])


class TestPlan:
    def test_swapped_pairs_of_eval_seeds_by_army_size_and_alternating_roles(self):
        rows = gate.plan(8)
        assert [r["pair"] for r in rows] == [1, 1, 2, 2, 3, 3, 4, 4]
        assert [r["swap"] for r in rows] == [False, True] * 4
        assert [r["role"] for r in rows] == ["attack", "defend"] * 4
        seeds = [r["seed"] for r in rows]
        assert seeds[0::2] == seeds[1::2] and len(set(seeds)) == 4 and seeds[0] == gate.GATE_BLOCK.start
        assert all(generate.split(s) == "eval" for s in seeds)
        for r, swapped in zip(rows[0::2], rows[1::2]):                       # the same battle, the armies swapped
            assert (r["own_units"], r["enemy_units"]) == (swapped["enemy_units"], swapped["own_units"])
            assert r["factions"] == {"own": swapped["factions"]["enemy"], "enemy": swapped["factions"]["own"]}
            assert r["side_budget"]["own"] == swapped["side_budget"]["enemy"]
        sizes = [r["own_units"] - 1 for r in rows[0::2]]                     # the generator's own army
        assert [lo <= n <= hi for n, (lo, hi) in zip(sizes, (gate.SIZE_BINS[b] for b in gate.PAIR_BINS))] == [True] * 4
        assert gate.plan(4) == rows[:4]                                     # 4 battles: 2 pairs, small and large
        assert gate.plan(1, offset=3) == [rows[3]]
        assert gate.min_wins(4) == 3 and gate.min_wins(200) == 150

    def test_symmetric_blocks_let_each_army_attack_and_defend_under_either_side(self):
        rows = gate.plan(8, symmetric=True)
        assert [r["pair"] for r in rows] == [1, 1, 2, 2, 3, 3, 4, 4]
        assert [(r["swap"], r["role"]) for r in rows] == list(gate.SYMMETRIC) * 2
        seeds = [r["seed"] for r in rows]
        assert len(set(seeds[:4])) == 1 and len(set(seeds[4:])) == 1 and seeds[0] != seeds[4]
        # the generator's own army (swap False for us) attacks in pair 1 and defends in pair 2
        attacker_army = {(r["pair"], "own" if (r["swap"] == (r["role"] == "defend")) else "enemy") for r in rows}
        assert attacker_army == {(1, "own"), (2, "enemy"), (3, "own"), (4, "enemy")}
        assert gate.plan(2, offset=2, symmetric=True) == rows[2:4]

    def test_the_battle_limit_is_the_simulator_s(self):
        assert gate.battle_limit_s() == 3600 and gate.deadline_s(3600) == 840


def result(**kw):
    row = {"event": "result", "status": "completed", "winner": 1, "duration_model_ms": 250_000, "duration_wall_s": 30,
           "side_1_men": 500, "side_2_men": 20, "side_1_standing_units": 5, "side_2_standing_units": 0,
           "nn_moves": 250, "nn_answered": 250, "nn_missed": 0, "nn_orders_given": 900, "nn_keeps": 10,
           "nn_bad_files": 0}
    row.update(kw)
    return row


def write_run(root, name, seed, role, res, difficulty=1, errors=(), swap=False):
    d = root / name
    d.mkdir(parents=True)
    config = {"own_role": role, "army": {"seed": seed, "men": {"own": 600, "enemy": 700}, "budget": 1000, "swap": swap},
              "factions": {"own": "wh_main_emp_empire", "enemy": "wh2_main_skv_skaven"},
              "units": {"own": [{}] * 3, "enemy": [{}] * 4}}
    (d / "manifest.json").write_text(json.dumps({"config": config}))
    (d / "launch.json").write_text("﻿" + json.dumps({"battle_difficulty": difficulty}), encoding="utf-8")
    (d / "status.json").write_text("﻿" + json.dumps({"status": "completed", "preferences_restored": True}),
                                   encoding="utf-8")
    lines = [json.dumps({"event": "error", "message": m}, separators=(",", ":")) for m in errors]
    if res:
        lines.append(json.dumps(res, separators=(",", ":")))
    (d / "events.jsonl").write_text("\n".join(lines) + "\n")
    return str(d)


class TestSummary:
    @pytest.mark.parametrize("status,winner,role,expected", [
        ("completed", 1, "attack", (1, "completed")), ("completed", 2, "defend", (2, "completed")),
        ("timeout", 0, "attack", (2, "timeout")), ("timeout", 0, "defend", (1, "timeout")),
        ("stalled", 0, "defend", (1, "stalled")), ("deadline", 0, "defend", (None, "deadline")),
        ("incomplete", 0, "attack", (None, "incomplete"))])
    def test_timeouts_go_to_the_defender(self, status, winner, role, expected):
        assert gate.outcome({"status": status, "winner": winner}, role) == expected
        assert gate.outcome(None, role) == (None, "no_result")

    def test_a_gate_folder(self, tmp_path, capsys):
        runs = tmp_path / "runs"
        battles = [
            {"battle": 1, "seed": 11, "role": "attack", "run": write_run(runs, "r1", 11, "attack", result())},
            {"battle": 2, "seed": 12, "role": "defend",
             "run": write_run(runs, "r2", 12, "defend", result(status="timeout", winner=0))},
            {"battle": 3, "seed": 13, "role": "attack",
             "run": write_run(runs, "r3", 13, "attack", result(status="completed", winner=2), errors=["boom"])},
            {"battle": 4, "seed": 14, "role": "defend", "run": write_run(runs, "r4", 14, "defend", result(), difficulty=3)},
        ]
        gate_dir = tmp_path / "gate"
        gate_dir.mkdir()
        (gate_dir / "battles.json").write_text(json.dumps({"checkpoint": "c.pt", "planned": 4, "min_wins": 3,
                                                           "battles": battles}))
        assert gate.main(["summary", str(gate_dir)]) == 1
        s = json.loads((gate_dir / "summary.json").read_text(encoding="utf-8"))
        assert (s["wins"], s["losses"], s["no_result"], s["passed"], s["fair"]) == (2, 1, 1, False, False)
        rows = s["battles"]
        assert [r["outcome"] for r in rows] == ["win", "win", "loss", "no_result"]
        assert rows[1]["how"] == "timeout" and rows[1]["winner"] == "net"
        assert rows[2]["lua_errors"] == ["boom"] and rows[3]["how"] == "not_normal_difficulty"
        assert rows[0]["units"] == {"own": 3, "enemy": 4} and rows[0]["men_left"] == {"own": 500, "enemy": 20}
        assert rows[0]["nn"]["moves"] == 250 and rows[0]["duration_s"] == 250.0
        out = capsys.readouterr().out
        assert "NOT PASSED" in out and "EMP-SKV" in out
        by = s["by_faction"]                                   # every run: our Empire against Skaven
        assert by["factions"]["EMP"]["attack"] == {"games": 2, "wins": 1, "win_rate": 0.5, "no_result": 0}
        assert by["factions"]["EMP"]["defend"] == {"games": 2, "wins": 1, "win_rate": 0.5, "no_result": 1}
        assert by["matchups"] == {"EMP-SKV": {"games": 4, "wins": 2, "win_rate": 0.5, "no_result": 1}}
        assert "by faction (wins/battles): EMP attack 1/2, defend 1/2 | EMP-SKV 2/4 0.50 (1 no result)" in out
        assert s["pairs"]["pairs"] == 0                                        # battles of an old plan: no pairs

    def test_pair_outcomes(self, tmp_path, capsys):
        runs = tmp_path / "runs"

        def entry(battle, seed, role, res, swap, run_swap=None):
            return {"battle": battle, "pair": (battle + 1) // 2, "swap": swap, "seed": seed, "role": role,
                    "run": write_run(runs, f"r{battle}", seed, role, res, swap=swap if run_swap is None else run_swap)}
        battles = [entry(1, 11, "attack", result(), False),                                  # pair 1: won both
                   entry(2, 11, "defend", result(status="timeout", winner=0), True),
                   entry(3, 13, "attack", result(winner=2), False),                          # pair 2: split
                   entry(4, 13, "defend", result(), True),
                   entry(5, 15, "attack", result(winner=2), False),                          # pair 3: lost both
                   entry(6, 15, "defend", result(winner=2), True),
                   entry(7, 17, "attack", result(), False),                                  # pair 4: incomplete
                   entry(8, 17, "defend", result(), True, run_swap=False)]   # the run is not the swapped battle
        (tmp_path / "battles.json").write_text(json.dumps({"planned": 8, "battles": battles}))
        s = gate.summarize(tmp_path)
        p = s["pairs"]
        assert [x["result"] for x in p["each"]] == ["won both", "split", "lost both", "incomplete"]
        assert s["battles"][7]["how"] == "run_does_not_match_plan"
        assert (p["pairs"], p["incomplete"]) == (3, 1)
        assert p["pair_score"] == pytest.approx(0.0) and p["won_both"] == pytest.approx(1 / 3)
        lines = "\n".join(gate.table(s))
        assert "1 won both; 2 split; 3 lost both; 4 incomplete; pair score +0.00 over 3 complete" in lines
        assert " 2   1s " in lines

    def test_gold_from_the_end_state_and_the_pair_gold(self, tmp_path, monkeypatch, capsys):
        """Each side: a lord (cost 100) and a unit (cost 300); gold lost as the simulator's reward counts it."""
        costs = {"lord": 100, "unit": 300}
        monkeypatch.setattr(gate, "_costs", lambda: costs)
        runs = tmp_path / "runs"

        def entry(battle, role, res, swap, end, final=True):
            d = runs / f"r{battle}"
            run = write_run(runs, f"r{battle}", 21, role, res, swap=swap)
            m = json.loads((d / "manifest.json").read_text())
            m["config"]["units"] = {s: [{"script_name": f"{s}_lord", "key": "lord"}, {"script_name": f"{s}_unit", "key": "unit"}]
                                    for s in ("own", "enemy")}
            m["config"]["army"]["cost"] = {"own": 400, "enemy": 400}
            (d / "manifest.json").write_text(json.dumps(m))
            units = [dict(n=n, side=1 if n.startswith("own") else 2, **u) for n, u in end.items()]
            sep = (",", ":")                                    # compact, as the game writes it
            ev = [json.dumps({"event": "nn_sample", "units": [dict(u, hp=1.0, r=False) for u in units]}, separators=sep),
                  json.dumps({"event": "nn_final" if final else "nn_sample", "units": units}, separators=sep)]
            old = (d / "events.jsonl").read_text().splitlines()
            (d / "events.jsonl").write_text("\n".join(ev + old) + "\n")
            return {"battle": battle, "pair": 1, "swap": swap, "seed": 21, "role": role, "run": run}
        full = {"men": 50, "hp": 1.0, "r": False, "s": False}
        b1 = entry(1, "attack", result(), False,                 # we win: their lord dead, their unit routs at half
                   {"own_lord": full, "own_unit": dict(full, hp=0.8), "enemy_lord": dict(full, men=0, hp=0.0),
                    "enemy_unit": dict(full, hp=0.5, r=True)})
        b2 = entry(2, "defend", result(winner=2), True,          # we lose: our unit shattered, no final event
                   {"own_lord": dict(full, hp=0.5), "own_unit": dict(full, s=True, r=True), "enemy_lord": full},
                   final=False)                                  # enemy_unit missing: lost whole
        (tmp_path / "battles.json").write_text(json.dumps({"planned": 2, "battles": [b1, b2]}))
        s = gate.summarize(tmp_path)
        g1, g2 = s["battles"][0]["gold"], s["battles"][1]["gold"]
        assert g1["lost"] == pytest.approx([60.0, 100 + 300 * (0.5 + 0.5 * 0.5)]) and g1["start"] == [400, 400]
        assert g1["destroyed"] == pytest.approx(325.0) and g1["budget"] == 1000
        assert g1["trade"] == pytest.approx((325 - 60) / 1000) and g1["margin"] == pytest.approx(1 - 60 / 400)
        assert g2["lost"] == pytest.approx([350.0, 300.0]) and g2["margin"] == pytest.approx(-(1 - 300 / 400))
        pg = s["pair_gold"]
        assert pg["pairs"] == 1 and pg["pair_gold"]["value"] == pytest.approx((325 + 300 - 60 - 350) / 1000)
        assert pg["weak"]["ratio_net"] == pytest.approx(300 / 350)             # army B: we lost with it
        assert pg["weak"]["ratio_opp"] == pytest.approx(60 / 325)              # the game AI with it in battle 1
        gate.main(["summary", str(tmp_path)])
        assert "pair gold (ours - game AI's, same armies, / budget): +0.215" in capsys.readouterr().out
        assert gate.ROUT_SHARE == 0.5                            # tools/nn/train/reward.py Weights.rout_share

    def test_a_missing_run_is_no_result_and_a_short_gate_does_not_pass(self, tmp_path):
        (tmp_path / "battles.json").write_text(json.dumps({"planned": 2, "battles": [
            {"battle": 1, "seed": 1, "role": "attack", "run": write_run(tmp_path, "r1", 1, "attack", result())},
            {"battle": 2, "seed": 2, "role": "defend", "run": None}]}))
        s = gate.summarize(tmp_path)
        assert s["min_wins"] == 2 and s["wins"] == 1 and s["no_result"] == 1 and not s["passed"]
        (tmp_path / "battles.json").write_text(json.dumps({"planned": 1, "battles": [
            {"battle": 1, "seed": 1, "role": "attack", "run": str(tmp_path / "r1")}]}))
        assert gate.summarize(tmp_path)["passed"] is True


class TestLiveliness:
    @staticmethod
    def events(tmp_path):
        """Two of our units and one of the game AI's, sampled every second for 4 s; our orders between."""
        def sample(t, own_t, enemy_t, m=False):
            units = [{"n": "own_a", "side": 1, "men": 90, "t": own_t, "m": m},
                     {"n": "own_b", "side": 1, "men": 90, "t": "", "m": False},
                     {"n": "enemy_a", "side": 2, "men": 90, "t": enemy_t},
                     {"n": "enemy_b", "side": 2, "men": 90, "t": ""}]
            return {"event": "nn_sample", "t": t, "units": units}

        def orders(t, *rows):
            return {"event": "nn_orders", "t": t, "orders": [dict(r, status="given") for r in rows]}
        lines = [sample(0, "", ""),
                 orders(500, {"u": "own_a", "k": "attack", "tg": "enemy_a"}, {"u": "own_b", "k": "move", "x": 0, "z": 0}),
                 sample(1000, "enemy_a", "own_a"),
                 orders(1500, {"u": "own_a", "k": "attack", "tg": "enemy_b"}, {"u": "own_b", "k": "move", "x": 0, "z": 20}),
                 sample(2000, "enemy_b", "own_b"),
                 orders(2500, {"u": "own_a", "k": "attack", "tg": "enemy_a"}, {"u": "own_b", "k": "move", "x": 0, "z": 26}),
                 sample(3000, "enemy_a", "own_a"),
                 sample(4000, "enemy_a", "own_a")]
        path = tmp_path / "events.jsonl"
        path.write_text("\n".join(json.dumps(x, separators=(",", ":")) for x in lines) + "\n", encoding="utf-8")
        return path

    def test_our_orders_and_both_sides_targets_from_a_recording(self, tmp_path):
        c = gate.liveliness(self.events(tmp_path))
        net, ai = c["net"], c["game_ai"]
        # own_a: attack a, b, a (2 switches, the last a flip); own_b: move, 20 m on (a change), 6 m (a re-point only)
        assert (net["changes"], net["switches"], net["flips"]) == (5, 2, 1)
        assert (net["repoints"], net["repoint_m"]) == (2, 26.0)
        assert net["unit_s"] == 8.0 and net["eng_switches"] == 2 and net["eng_flips"] == 1   # own_a: a, b, a
        assert ai["eng_switches"] == 2 and ai["eng_s"] == 8.0                                # enemy_a: own_a, own_b, own_a
        r = gate.lively_rates(net)
        assert r["order_changes_per_min"] == pytest.approx(5 / 8 * 60) and r["move_jitter_m"] == 13.0
        assert r["twitch_share"] is None                                                     # under 30 s out of melee

    def test_the_summary_has_rates_per_role_and_the_game_ais_band(self, tmp_path):
        c = gate.liveliness(self.events(tmp_path))
        rows = [{"role": "attack", "lively": c}, {"role": "defend", "lively": c}, {"role": "attack", "lively": None}]
        s = gate.lively_summary(rows)
        assert set(s["net"]) == {"attack", "defend"} and set(s["game_ai"]) == {"engine_switches_per_min",
                                                                                 "engine_flips_per_min"}
        assert s["band"]["game_ai"] == [15.0, 15.0, 15.0]
        text = "\n".join(gate.lively_lines(s))
        assert "the game's AI 15.00 [15.00, 15.00]" in text and "order changes 37.50 / 37.50" in text
        assert gate.lively_summary([{"role": "attack", "lively": None}]) is None

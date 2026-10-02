"""tools.nn.scenario, gamedata and measure: arenas with an army per side, their runs, the measurements."""
import json

import numpy as np
import pytest

from tools import build
from tools import config as project
from tools.nn import gamedata, measure
from tools.nn import scenario as nn_scenario

PASSPORTS = json.loads(measure.PASSPORTS.read_text(encoding="utf-8"))["units"]


class TestArenas:
    def test_the_default_arena_is_the_same_army_on_both_sides(self):
        arena = nn_scenario.load_arena()
        sides = nn_scenario.armies(arena)
        assert sides["own"] == sides["enemy"] and sides["own"]["faction"] == "wh_main_emp_empire"
        config = nn_scenario.run_config(arena)
        assert config["arena"] == "arena" and len(config["units"]["own"]) == len(config["units"]["enemy"]) == 7

    def test_every_named_arena_loads_with_its_own_armies_and_the_base_map(self):
        base = nn_scenario.load_arena()
        names = json.loads(nn_scenario.ARENAS.read_text(encoding="utf-8"))["arenas"]
        for name in names:
            arena = nn_scenario.load_arena(name)
            assert arena["name"] == name and arena["map"] == base["map"] and "units" not in arena
            assert measure.kind(name) is not None, name
            xml = nn_scenario.scenario_xml(arena, "enemy")
            for side, army in nn_scenario.armies(arena).items():
                assert f"<faction>{army['faction']}</faction>" in xml
                assert all(u["key"] in PASSPORTS for u in army["units"]), name
            assert xml.count("<general>") == sum(bool(u.get("general")) for a in nn_scenario.armies(arena).values()
                                                 for u in a["units"])

    def test_whole_battles_give_the_skaven_four_fifths_of_the_gold(self):
        """config/nn/pools.json budget_factor: Skaven 0.8 of the Empire's budget (2050 against 2500)."""
        for name in ("whole_emp_v_skv", "whole_skv_v_emp"):
            sides = nn_scenario.armies(nn_scenario.load_arena(name))
            cost = {a["faction"]: sum(PASSPORTS[u["key"]]["multiplayer_cost"] for u in a["units"])
                    for a in sides.values()}
            assert cost == {"wh_main_emp_empire": 2500, "wh2_main_skv_skaven": 2050}
            assert abs(cost["wh2_main_skv_skaven"] / cost["wh_main_emp_empire"] - 0.8) <= 0.05

    def test_an_unknown_arena_and_a_named_arena_that_moves_the_map_are_errors(self, tmp_path):
        with pytest.raises(KeyError):
            nn_scenario.load_arena("nowhere")
        bad = tmp_path / "arenas.json"
        army = {"faction": "f", "units": [{"slot": "a", "key": "k", "men": 1, "forward": 0, "lateral": 0, "width": 5}]}
        bad.write_text(json.dumps({"arenas": {"x": {"map": "m", "sides": {"own": army, "enemy": army}}}}))
        with pytest.raises(AssertionError):
            nn_scenario.load_arena("x", arenas_path=bad)

    def test_hold_and_a_named_arena_build(self, tmp_path, monkeypatch):
        monkeypatch.setattr(project, "BUILD", tmp_path)
        monkeypatch.setattr(nn_scenario, "SCENARIO", tmp_path / "nn_arena.xml")
        written = []
        monkeypatch.setattr(build, "build", lambda target, config, scenario=None: written.append(config) or {})
        assert build.main(["nn-arena", "--arena", "missile_archers_v_slave", "--own-ai", "hold"]) == 0
        config = written[0]
        assert config["own_ai"] == "hold" and config["enemy_role"] == "attack"
        assert config["arena"] == "missile_archers_v_slave"
        assert config["factions"] == {"own": "wh2_main_skv_skaven", "enemy": "wh_main_emp_empire"}
        assert [u["script_name"] for u in config["units"]["enemy"]] == ["enemy_archer"]
        # Our side holds, so it must win on timeout: the game's AI attacks.
        assert "<timeout_winning_alliance_index>0</timeout_winning_alliance_index>" in \
            (tmp_path / "nn_arena.xml").read_text(encoding="utf-8")


def unit(n, side, **kw):
    row = {"n": n, "side": side, "x": 0, "z": 0, "men": 100, "hp": 1.0, "mp": 1.0, "ms": 1, "a": 0,
           "r": False, "s": False, "w": False, "m": False, "mv": False, "f": False, "fire": False, "t": ""}
    row.update(kw)
    return row


def write_run(root, name, arena, own, enemy, frames, own_ai="attack", series=False, difficulty=1, factions=None):
    d = root / name
    d.mkdir(parents=True)
    units = {"own": [{"script_name": n, "slot": n[4:], "key": k} for n, k in own],
             "enemy": [{"script_name": n, "slot": n[6:], "key": k} for n, k in enemy]}
    config = {"arena": arena, "own_ai": own_ai, "enemy_role": "defend", "units": units}
    if factions:
        config["factions"] = factions
    (d / "manifest.json").write_text(json.dumps({"config": config}))
    (d / "launch.json").write_text(json.dumps({"battle_difficulty": difficulty}))
    if series:
        (d / "series.json").write_text("{}")
    compact = {"separators": (",", ":")}    # as the game writes it
    lines = [json.dumps({"event": "nn_sample", "t": t * 1000, "units": us}, **compact) for t, us in enumerate(frames)]
    lines.append(json.dumps({"event": "result", "status": "completed", "winner": 1}, **compact))
    (d / "events.jsonl").write_text("\n".join(lines))
    return d


SPEAR, SLAVE = "wh_main_emp_inf_spearmen_0", "wh2_main_skv_inf_skavenslave_spearmen_0"
ARCHER = "wh2_dlc13_emp_inf_archers_0"


class TestRuns:
    def test_a_run_with_its_own_armies_loads_by_its_manifest(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gamedata, "RUNS", tmp_path)
        frames = [[unit("own_spear", 1, men=120), unit("enemy_slave", 2, men=180, x=5, t="own_spear")]] * 3
        write_run(tmp_path, "20260101-000001", "pair_x", [("own_spear", SPEAR)], [("enemy_slave", SLAVE)], frames)
        write_run(tmp_path, "20260101-000002", "pair_x", [("own_spear", SPEAR)], [("enemy_slave", SLAVE)], frames,
                  series=True)
        write_run(tmp_path, "20260101-000003", "pair_x", [("own_spear", SPEAR)], [("enemy_slave", SLAVE)], frames,
                  difficulty=3)
        found = gamedata.runs()
        assert [d.name for d in found] == ["20260101-000001"]
        assert gamedata.runs(arena="arena") == []
        b = gamedata.load(found[0])
        assert b.arena == "pair_x" and b.names == ("own_spear", "enemy_slave") and b.keys == (SPEAR, SLAVE)
        assert b.f["men"].shape == (3, 2) and list(b.side) == [1, 2] and b.target[0, 1] == 0


def pair_frames():
    """Contact at t=2; the slaves lose a man a second, waver 18 s and rout 24 s after it."""
    frames = []
    for t in range(30):
        m = t >= 2
        men = 180 - max(0, t - 2)
        frames.append([unit("own_spear", 1, men=120 - (t >= 5), hp=1 - 0.001 * max(0, t - 2), m=m,
                            x=-30 + min(t, 3) * 12, mv=t < 3),
                       unit("enemy_slave", 2, men=men, hp=men / 180, m=m, x=0,
                            w=20 <= t < 26, r=t >= 26, mp=1 - 0.03 * t)])
    return frames


class TestMeasure:
    def test_a_melee_pair(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gamedata, "RUNS", tmp_path)
        write_run(tmp_path, "20260101-000001", "pair_x", [("own_spear", SPEAR)], [("enemy_slave", SLAVE)],
                  pair_frames())
        data = measure.targets()
        row = data["pair_x"]["runs"][0]
        assert row["contact_t_s"] == 2 and row["fight_s"] == 24
        slave = row["units"][1]
        assert slave["men_per_s"] == 1.0 and slave["hp_per_s"] == 50.0
        assert slave["waver_after_s"] == 18 and slave["rout_after_s"] == 24 and slave["rallied"] is False
        assert slave["steady_hp_per_s"] == 50.0 and slave["morale_every_10s"][0] == pytest.approx(0.94)
        assert slave["implied_foes_fighting"] == pytest.approx(
            50.0 / measure.melee_per_fighter(PASSPORTS[SPEAR], PASSPORTS[SLAVE]), abs=0.01)
        assert row["units"][0]["rout_after_s"] is None
        assert data["pair_x"]["summary"]["enemy_slave"]["routed_runs"] == 1
        assert [x["slot"] for x in row["approach"]] == ["own_spear"]      # the slaves stood
        assert row["approach"][0]["top_m_s"] == pytest.approx(12, abs=0.01)

    def test_the_database_melee_rule(self):
        spear, slave = PASSPORTS[SPEAR], PASSPORTS[SLAVE]
        assert measure.hit_chance(spear, slave) == 0.39            # 35 + 20 - 16
        assert measure.hit_chance(slave, spear) == 0.10            # 35 + 9 - 34
        assert measure.per_hit(290, 140, 0, 60) == 60               # a hit takes no more than a man has
        assert measure.per_hit(12, 4, 30, 69) == pytest.approx(4 + 12 * 0.775)

    def test_a_missile_run(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gamedata, "RUNS", tmp_path)
        frames = []
        for t in range(30):
            ammo = 1800 - 9 * max(0, t - 5)
            hp = 1 - 90 * max(0, t - 7) / 9000
            frames.append([unit("own_slave", 1, men=180, hp=hp, x=0),
                           unit("enemy_archer", 2, men=90, a=ammo, x=100, mv=t < 4)])
        write_run(tmp_path, "20260101-000001", "missile_x", [("own_slave", SLAVE)], [("enemy_archer", ARCHER)],
                  frames, own_ai="hold")
        row = measure.targets()["missile_x"]["runs"][0]
        assert row["shooting"] and row["first_shot_t_s"] == 6 and row["first_shot_after_stop_s"] == 2
        assert row["shots_per_man_per_s"] == pytest.approx(0.1) and row["implied_reload_s"] == pytest.approx(10)
        (bin_,) = row["by_distance"]
        assert bin_["distance_m"] == [90, 110] and bin_["hp_per_shot"] == pytest.approx(10, abs=0.01)
        assert bin_["implied_hit_rate"] == pytest.approx(10 / 19, abs=0.01)   # armour 0: 17 + 2

    def test_a_whole_battle(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gamedata, "RUNS", tmp_path)
        frames = [[unit("own_lord", 1, men=1), unit("own_spear", 1, men=120 - t, r=10 <= t < 20),
                   unit("enemy_slave", 2, men=180, m=t >= 5)] for t in range(40)]
        write_run(tmp_path, "20260101-000001", "whole_x", [("own_lord", "wh_main_emp_cha_general_0"),
                                                           ("own_spear", SPEAR)], [("enemy_slave", SLAVE)], frames,
                  factions={"own": "emp", "enemy": "skv"})
        data = measure.targets()["whole_x"]
        row = data["runs"][0]
        assert row["winner"] == "emp" and row["attacker"] == "emp" and row["first_contact_s"] == 5
        assert row["sides"]["emp"]["routed_units"] == 1 and row["sides"]["emp"]["rallied_units"] == 1
        assert row["sides"]["emp"]["men_every_30s"] == [121.0, 91.0]
        assert data["summary"]["wins"] == {"emp": 1}

    def test_the_mirror_arena_is_not_a_measurement(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gamedata, "RUNS", tmp_path)
        write_run(tmp_path, "20260101-000001", "arena", [("own_spear_1", SPEAR)], [("enemy_spear_1", SPEAR)],
                  [[unit("own_spear_1", 1), unit("enemy_spear_1", 2)]])
        assert measure.targets() == {}
        assert np.isnan(measure.stats([])["mean"] or np.nan)

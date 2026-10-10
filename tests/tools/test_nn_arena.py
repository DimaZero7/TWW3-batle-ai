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

    def test_whole_battles_give_the_skaven_their_budget_factor_of_the_gold(self):
        """config/nn/pools.json budget_factor: the Skaven's gold against the Empire's 2500 (factor 1.0 since
        02.10.2026: 2575; at 0.8 the arenas held 2050), read from the config."""
        from tools.nn.armies import pools
        p = pools.load()
        factor = p["wh2_main_skv_skaven"].budget_factor / p["wh_main_emp_empire"].budget_factor
        for name in ("whole_emp_v_skv", "whole_skv_v_emp"):
            sides = nn_scenario.armies(nn_scenario.load_arena(name))
            cost = {a["faction"]: sum(PASSPORTS[u["key"]]["multiplayer_cost"] for u in a["units"])
                    for a in sides.values()}
            assert cost["wh_main_emp_empire"] == 2500
            assert abs(cost["wh2_main_skv_skaven"] / cost["wh_main_emp_empire"] - factor) <= 0.05

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

    def test_the_enemy_under_a_script_of_the_simulator(self, tmp_path, monkeypatch):
        monkeypatch.setattr(project, "BUILD", tmp_path)
        monkeypatch.setattr(nn_scenario, "SCENARIO", tmp_path / "nn_arena.xml")
        written = []
        monkeypatch.setattr(build, "build", lambda target, config, scenario=None: written.append(config) or {})
        assert build.main(["nn-arena", "--own-ai", "net", "--enemy-ai", "ai_like"]) == 0
        assert written[0]["enemy_ai"] == "companion" and written[0]["enemy_script"] == "ai_like"
        assert written[0]["enemy_role"] == "attack" and all(u["width"] for u in written[0]["units"]["enemy"])
        assert build.main(["nn-arena", "--own-ai", "net"]) == 0
        assert "enemy_ai" not in written[1] and "enemy_script" not in written[1]
        with pytest.raises(SystemExit):
            build.main(["nn-arena", "--own-ai", "attack", "--enemy-ai", "ai_like"])

    def test_a_human_against_the_network(self, tmp_path, monkeypatch):
        """tools.build human --enemy-ai net: the human plays side 1 (defends), the network side 2 through the
        companion; --free-speed lets the human change the speed and pause; long limits as given."""
        monkeypatch.setattr(project, "BUILD", tmp_path)
        written = []
        monkeypatch.setattr(build, "build", lambda target, config, scenario=None: written.append(config) or {})
        assert build.main(["human", "--army-seed", "1000900014", "--army-swap", "--enemy-ai", "net", "--free-speed",
                           "--stall-minutes", "30", "--deadline", "21600"]) == 0
        c = written[0]
        assert (c["own_ai"], c["own_role"], c["enemy_role"]) == ("human", "defend", "attack")
        assert (c["enemy_ai"], c["enemy_script"], c["decide_ms"], c["poll_ms"]) == ("companion", "net", 1000, 100)
        assert c["free_speed"] is True and c["speed"] == 1 and c["timeout_ms"] == 3600000
        assert c["deadline_s"] == 21600 and c["stall_ms"] == 30 * 60000 and c["soldiers_every"] == 5
        assert c["army"]["seed"] == 1000900014 and c["army"]["swap"] is True and "skirmish" not in c
        assert build.main(["human", "--army-seed", "1000900014"]) == 0          # the human's battle as before
        assert not {"enemy_ai", "enemy_script", "free_speed", "decide_ms"} & set(written[1])
        for bad in (["nn-arena", "--own-ai", "net", "--enemy-ai", "net"],     # the network on both sides: no
                    ["nn-arena", "--own-ai", "net", "--free-speed"],
                    ["human", "--skirmish", "off"]):                          # no bridge to turn it off for
            with pytest.raises(SystemExit):
                build.main(bad)


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
        # Recorded before the collapse and morale values: all unknown.
        assert np.isnan(b.f["sv"]).all() and np.isnan(b.bop).all() and b.bop_side == 0
        assert gamedata.morale_coverage(b) == {"bop": 0.0, "sv": 0.0, "pcr": 0.0, "phr": 0.0, "mge": 0.0}
        assert np.isnan(gamedata.balance(b, 1)).all()

    def test_the_collapse_and_morale_values_are_read(self, tmp_path):
        # As nn_arena.lua writes them: bop / bop_side per sample, sv / pcr / phr / mge per unit,
        # each left out when the game gave nothing.
        frames = [[unit("own_spear", 1, sv=800.0 - 10 * t, pcr=0.05 * t, phr=0.02 * t, mge="Losses"),
                   unit("enemy_slave", 2, sv=400.0)] for t in range(3)]
        frames[1][1]["mge"] = "Outnumbered"
        d = write_run(tmp_path, "20260102-000001", "pair_x", [("own_spear", SPEAR)], [("enemy_slave", SLAVE)], frames)
        lines = (d / "events.jsonl").read_text().splitlines()
        for i, bop in enumerate((0.5, None, 0.75)):
            row = json.loads(lines[i])
            row["bop_side"] = 1
            if bop is not None:
                row["bop"] = bop
            lines[i] = json.dumps(row)
        (d / "events.jsonl").write_text("\n".join(lines))
        b = gamedata.load(d)
        assert b.bop_side == 1 and b.bop[0] == 0.5 and np.isnan(b.bop[1]) and b.bop[2] == 0.75
        assert list(gamedata.balance(b, 2)[[0, 2]]) == [0.5, 0.25]
        assert list(b.f["sv"][:, 0]) == [800.0, 790.0, 780.0] and list(b.f["sv"][:, 1]) == [400.0] * 3
        assert b.f["pcr"][2, 0] == pytest.approx(0.1) and np.isnan(b.f["phr"][:, 1]).all()
        assert list(b.mge[:, 0]) == ["Losses"] * 3 and list(b.mge[:, 1]) == [None, "Outnumbered", None]
        cov = gamedata.morale_coverage(b)
        assert cov["bop"] == pytest.approx(2 / 3) and cov["sv"] == 1.0 and cov["pcr"] == 0.5
        assert cov["mge"] == pytest.approx(4 / 6)
        assert "sv" not in gamedata.FLOAT_FIELDS     # not the simulator's state


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

    def test_the_first_strike_in_the_contact_second_counts_in_the_charge(self, tmp_path, monkeypatch):
        # The slaves lose 10 men between the last record without the melee flag (t=1) and the first with it
        # (t=2): the charge counts from t=1, the times from t=2.
        monkeypatch.setattr(gamedata, "RUNS", tmp_path)
        frames = pair_frames()
        for t in range(2, 30):
            slave = frames[t][1]
            slave["men"] -= 10
            slave["hp"] = slave["men"] / 180
        write_run(tmp_path, "20260101-000001", "pair_x", [("own_spear", SPEAR)], [("enemy_slave", SLAVE)], frames)
        row = measure.targets()["pair_x"]["runs"][0]
        slave = row["units"][1]
        per_man = PASSPORTS[SLAVE]["hp_total"] / 180
        assert row["contact_t_s"] == 2 and row["fight_s"] == 24
        assert slave["contact_s_hp_lost"] == pytest.approx(10 * per_man, abs=0.1)
        assert slave["charge_hp_lost"] == pytest.approx((10 + 15) * per_man, abs=0.1)   # t=1 to t=17

    def test_the_database_melee_rule(self):
        spear, slave = PASSPORTS[SPEAR], PASSPORTS[SLAVE]
        assert measure.hit_chance(spear, slave) == 0.39            # 35 + 20 - 16
        assert measure.hit_chance(slave, spear) == 0.10            # 35 + 9 - 34
        assert measure.per_hit(290, 140, 0, 60) == 60               # a hit takes no more than a man has
        # many blows to kill a man: the overkill of the last is lost (smoothed: hp / (hp / mean + 1/2) here)
        assert measure.per_hit(12, 4, 30, 69, single=True) == pytest.approx(4 + 12 * 0.775)
        assert measure.per_hit(12, 4, 30, 69) == pytest.approx(69 / (1 + (69 - (12.4 + 14.2) / 2) / 13.3 + 0.5))

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
        # armour 0: 17 + 2 a hit; the slave's 50 HP take 3 such hits, the last one's overkill lost (measure.per_hit)
        hp_man = measure.passports()[SLAVE]["hp_per_man"]
        assert bin_["implied_hit_rate"] == pytest.approx(10 / measure.per_hit(17, 2, 0, hp_man), abs=0.01)

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

"""The rules' numbers (tools/label/digest_rules.py; no torch): kinds by the passport, the person's right (the game's
world is left-handed: facing +x, +z is on the right), the deployment against the line, the first attack by kind."""
import json

from tools.label import digest as DG
from tools.label import digest_rules as R


def test_kinds():
    assert R.kind("wh2_dlc12_skv_inf_warplock_jezzails_0") == "sniper"
    assert R.kind("wh2_main_skv_inf_gutter_runner_slingers_0") == "shooters"
    assert R.kind("wh2_main_skv_art_warp_lightning_cannon") == "artillery"
    assert R.kind("wh_main_emp_cav_empire_knights") == "cavalry"
    assert R.kind("wh2_dlc16_skv_mon_wolf_rats_1") == "beasts"
    assert R.kind("wh2_main_skv_mon_rat_ogres") == "monster"
    assert R.kind("wh_main_emp_cha_general_0") == "lord"
    assert R.kind("wh_main_emp_inf_greatswords") == "melee"


def u(n, side, x, z, **k):
    return {"n": n, "side": side, "x": x, "z": z, "b": 90, "men": 100, "ow": 30, "t": "", "f": False, "m": False,
            "r": False, **k}


def test_right_and_deployment(tmp_path):
    lab, run = tmp_path / "lab", tmp_path / "runs" / "r"
    lab.mkdir()
    run.mkdir(parents=True)
    names = ["own_g1", "own_g2", "own_rifle", "enemy_a"]
    keys = ["wh_main_emp_inf_greatswords", "wh_main_emp_inf_greatswords", "wh3_dlc25_emp_inf_hochland_long_rifles",
            "wh2_main_skv_inf_clanrats_0"]
    (lab / "session.json").write_text(json.dumps({"batch": "B", "names": names, "keys": keys, "sides": [1, 1, 1, 2]}),
                                      encoding="utf-8")
    ev = [{"batch": "B", "event": "nn_sample", "t": s * 1000,
           "units": [u("own_g1", 1, -200, -20), u("own_g2", 1, -200, 20), u("own_rifle", 1, -230, 100),
                     u("enemy_a", 2, 200, 0)]} for s in range(0, 5)]
    (run / "events.jsonl").write_text("\n".join(json.dumps(e) for e in ev), encoding="utf-8")
    (lab / "commands.jsonl").write_text(json.dumps({"sec": 2.0, "name": "Attack Unit", "units": ["own_rifle"],
                                                    "target": "enemy_a", "bool1": True,
                                                    "ordered": {"own_rifle": {"fast": True}}}), encoding="utf-8")
    rec = DG.Record(lab, run)
    fr = R.Frame(rec, 1)
    right, fwd = fr.local(-230, 100)
    assert right > 0 and fwd < 0                         # +z is the right flank facing +x; behind the centre
    dep, _ = R.deployment(rec)
    s = dep["by_kind"]["sniper"]
    assert s["n"] == 1 and -45 < s["ahead_of_line_m"] < -25 and s["thirds"] == {"right": 1} and s["outside_line"] == 1
    t = R.targets(rec, fr)
    assert t["first_attack_s"] == {"melee": 2} and t["by_attacker"]["sniper"] == {"melee": 1}

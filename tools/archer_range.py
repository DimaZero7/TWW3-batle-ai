"""Scenario and config for src/entries/archer_range.lua: when do archers start
shooting, and does the depth of their block matter (user, 28.09.2026)?

Four lanes 200 m apart on The Moorlands Route (south, open ground); in each an
Empire archer unit at its own width — 60, 30, 15 and 8 m, so 7.9 to 46.3 m deep
(data/roster) — and one Empire spearmen unit in front of it as the target.
"""
from tools import config as project
from tools.sim.formation import probe_xml

ARCHERS, SPEARMEN, GENERAL = "wh2_dlc13_emp_inf_archers_0", "wh_main_emp_inf_spearmen_0", "wh_main_emp_cha_general_0"
LANES = [(-350, 60), (-150, 30), (50, 15), (250, 8)]
Z0 = -450  # the archers' front rank
SCENARIO = project.SCENARIOS / "archer_range.xml"


def write_scenario():
    own = [{"script_name": f"arc_{i}", "key": ARCHERS, "men": 90, "general": False, "x": x, "z": Z0 - 60}
           for i, (x, _) in enumerate(LANES, start=1)]
    own.append({"script_name": "own_general", "key": GENERAL, "men": 1, "general": True, "x": -450, "z": -150})
    enemy = [{"script_name": f"tgt_{i}", "key": SPEARMEN, "men": 120, "general": False, "x": x, "z": Z0 + 200}
             for i, (x, _) in enumerate(LANES, start=1)]
    enemy.append({"script_name": "enemy_general", "key": GENERAL, "men": 1, "general": True, "x": 450, "z": 100})
    SCENARIO.write_text(probe_xml({"own": own, "enemy": enemy}), encoding="utf-8")


def run_config(mode):
    """Entry config and the longest model length in seconds."""
    config = {"mode": mode, "z0": Z0, "start_d": 170, "end_d": 40, "step_m": 2, "step_ticks": 2, "after_ticks": 10,
              "lanes": [{"archer": f"arc_{i}", "target": f"tgt_{i}", "width": w, "x": x}
                        for i, (x, w) in enumerate(LANES, start=1)],
              "held": [{"side": 1, "name": "own_general"}, {"side": 2, "name": "enemy_general"}]}
    model_s = (170 - 40) / 2 * 2 + 60 if mode == "fire_at_will" else 180
    return config, model_s

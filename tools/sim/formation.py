"""Formation simulation: the same Lua code as in battle places an army.

    python -m tools.sim.formation <army config> [--out DIR]

<army config>: a name in config/armies/ (first_attack) or a path. Units come
from data/roster (measured card and formation). Our side goes through the same
chain as in battle, src/apps/plan: assessment -> strategy -> formation, facing
the enemy anchor. The enemy is drawn with the "wall and arc" roles and layout,
for the picture only (in battle the game AI places it).

Output (default research/analysis/formation/<army>/):
  plan.json  the Lua result for both sides, overlaps
  plan.png   rectangles of every unit, archer range arcs, checks
"""
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon, Wedge  # noqa: E402

from tools import config as project  # noqa: E402
from tools import roster  # noqa: E402
from tools.lua_runtime import new_runtime  # noqa: E402

LORD_SHAPE = [{"ordered_m": 5, "front_m": 1.0, "depth_m": 1.0}]
COLORS = {"wall": "#1f77b4", "arc": "#2ca02c", "lord": "#d62728", "other": "#7f7f7f"}


def load_army(name):
    path = Path(name)
    if not path.suffix:
        path = project.CONFIG_DIR / "armies" / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")), path.stem


# Card stats used by apps.assessment (data/roster card.stats keys).
CARD_STATS = {"armour": "stat_armour", "melee_attack": "stat_melee_attack", "melee_defence": "stat_melee_defence",
              "missile_damage": "stat_missile_damage_over_time", "ammo": "stat_ammo"}


def _number(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def roster_units(spec, entries):
    """Lua input units from the roster; one entry per unit instance. Unknown values are None."""
    by_key = {e["key"]: e for e in entries}
    units = []
    for item in spec["units"]:
        entry = by_key.get(item["key"])
        assert entry, f'{item["key"]} is not in data/roster (python -m tools.roster update ...)'
        profile, stats = entry["card"]["profile"], entry["card"]["stats"]
        shapes = [{"ordered_m": w["ordered_m"], "front_m": w["front_m"], "depth_m": w["depth_m"]}
                  for w in entry["formation"].get("widths", []) if "front_m" in w]
        card = {name: _number((stats.get(key) or {}).get("value")) for name, key in CARD_STATS.items()}
        for i in range(item["count"]):
            unit = dict(card)
            unit.update({"id": f'{item["key"]}#{i + 1}', "key": item["key"], "class": profile.get("unit_class"),
                         "men": profile.get("initial_number_of_men"), "commanding": profile.get("commanding") is True,
                         "fire": entry["ours"].get("fire"), "range_m": profile.get("missile_range") or 0,
                         "health": _number(entry["card"]["details"].get("HealthMax")),
                         # A fresh list per unit: lupa converts a repeated Python object only once.
                         "shapes": [dict(x) for x in (shapes or LORD_SHAPE)]})
            units.append(unit)
    return units


class Planner:
    """The Lua modules, called exactly as in battle."""

    def __init__(self):
        self.lua = new_runtime()
        self._start = self.lua.eval("""
            function(input)
                local plan = require('apps.plan.services').start(input)
                if plan.formation then
                    plan.formation.overlaps = require('apps.formation.services').overlaps(plan.formation.placements)
                end
                return require('apps.core.json').encode(plan)
            end
        """)
        self._picture = self.lua.eval("""
            function(input)
                local strategy = require('apps.strategy.services')
                local assessment = require('apps.assessment.services')
                local f = require('apps.formation.services')
                local roles = strategy.assign_roles(strategy.find('wall_and_arc'), input.units, assessment.category)
                for _, u in ipairs(input.units) do u.role = roles[u.id] end
                return require('apps.core.json').encode(f.plan('line_and_blocks', input, {}))
            end
        """)
        self._facing = self.lua.eval("function(a, b) return require('apps.formation.services').facing(a, b) end")

    def facing(self, a, b):
        return self._facing(self.lua.table_from({"x": a[0], "z": a[1]}), self.lua.table_from({"x": b[0], "z": b[1]}))

    def start(self, role, anchor, bearing, units, enemy_units, enemy_lord=None, formation_params=None):
        """apps.plan.start: {status, features, decision, formation}."""
        data = {"role": role, "bearing": bearing, "own": {"anchor": {"x": anchor[0], "z": anchor[1]}, "units": units},
                "enemy": {"units": enemy_units}}
        if enemy_lord:
            data["enemy"]["lord"] = {"x": enemy_lord[0], "z": enemy_lord[1]}
        if formation_params:
            data["formation_params"] = formation_params
        return json.loads(self._start(self.lua.table_from(data, recursive=True)))

    def picture(self, anchor, bearing, units):
        data = {"anchor": {"x": anchor[0], "z": anchor[1]}, "bearing": bearing, "units": units}
        return json.loads(self._picture(self.lua.table_from(data, recursive=True)))


def corners(p):
    """Rectangle of a placement: front rank centre, width along 'right', depth backwards."""
    b = math.radians(p["bearing"])
    f = (math.sin(b), math.cos(b))
    r = (math.cos(b), -math.sin(b))
    half = p["front_m"] / 2
    pts = []
    for along, back in ((-half, 0), (half, 0), (half, p["depth_m"]), (-half, p["depth_m"])):
        pts.append((p["x"] + r[0] * along - f[0] * back, p["z"] + r[1] * along - f[1] * back))
    return pts


def draw(path, sides, units_by_id, title):
    fig, ax = plt.subplots(figsize=(10, 11))
    for side, plan in sides.items():
        for p in plan["placements"]:
            ax.add_patch(Polygon(corners(p), closed=True, fc=COLORS[p["role"]], ec="k", lw=0.5,
                                 alpha=0.85 if side == "own" else 0.35))
            if side == "own" and p["role"] == "arc":
                u = units_by_id[p["id"]]
                b = math.radians(p["bearing"])
                cx = p["x"] - math.sin(b) * p["depth_m"] / 2
                cz = p["z"] - math.cos(b) * p["depth_m"] / 2
                # Room to turn in place: the circle the block sweeps.
                ax.add_patch(plt.Circle((cx, cz), math.hypot(p["front_m"], p["depth_m"]) / 2, fill=False,
                                        ls="--", ec=COLORS["arc"], lw=0.8))
                # Wedge angles are counter-clockwise from +X in the X/Z plot.
                centre = math.degrees(math.atan2(math.cos(b), math.sin(b)))
                ax.add_patch(Wedge((cx, cz), u["range_m"], centre - 30, centre + 30, fill=False,
                                   ec=COLORS["arc"], lw=0.6, alpha=0.6))
    for side, plan in sides.items():
        for p in plan["placements"]:
            if p["role"] == "lord":
                ax.scatter([p["x"]], [p["z"]], marker="*", s=260, c=COLORS["lord"], ec="k", zorder=5,
                           alpha=1 if side == "own" else 0.45)
                ax.annotate("наш лорд" if side == "own" else "лорд врага", (p["x"], p["z"]), fontsize=8,
                            xytext=(6, 6), textcoords="offset points")
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set(xlabel="X, m", ylabel="Z, m", title=title)
    ax.grid(alpha=0.3)
    handles = [plt.Line2D([], [], color=c, lw=6, label=l) for l, c in
               (("стена (пехота)", COLORS["wall"]), ("стрелки навесом", COLORS["arc"]), ("лорд", COLORS["lord"]))]
    ax.legend(handles=handles, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def simulate(army, planner=None, params=None):
    """sides["own"]: our formation plus strategy, features, decision; sides["enemy"]: the picture."""
    planner = planner or Planner()
    entries = roster.load_roster()
    own_units, enemy_units = roster_units(army["own"], entries), roster_units(army["enemy"], entries)
    # The enemy first: our lord needs to know where the enemy lord stands.
    sides = {"enemy": planner.picture(army["enemy"]["anchor"],
                                      planner.facing(army["enemy"]["anchor"], army["own"]["anchor"]),
                                      [dict(u, shapes=[dict(s) for s in u["shapes"]]) for u in enemy_units])}
    lords = [p for p in sides["enemy"]["placements"] if p["role"] == "lord"]
    plan = planner.start(army["own"].get("role"), army["own"]["anchor"],
                         planner.facing(army["own"]["anchor"], army["enemy"]["anchor"]), own_units,
                         enemy_units, (lords[0]["x"], lords[0]["z"]) if lords else None, params)
    own = plan.get("formation") or {"placements": [], "options": [], "overlaps": [], "unplaced": []}
    own.update(status=plan["status"], strategy=plan["decision"]["strategy"], features=plan["features"],
               decision=plan["decision"])
    sides["own"] = own
    return sides, {u["id"]: u for u in own_units}


def probe(army, speed, settle_ms, planner=None):
    """Scenario XML and entry config for src/entries/formation_probe.lua.

    Our units get the same Lua input as in the simulation (id = script name);
    the game plans them again in battle. The enemy is placed at its simulated
    plan by teleport and held, so the picture and the battle match.
    """
    entries = roster.load_roster()
    sides, _ = simulate(army, planner)
    config, xml_sides = {"speed": speed, "settle_ms": settle_ms}, {}
    for side, prefix in (("own", "own"), ("enemy", "enemy")):
        units = roster_units(army[side], entries)
        names = {}
        for i, u in enumerate(units, start=1):
            names[u["id"]] = f"{prefix}_{i}"
        xml_units = []
        ax, az = army[side]["anchor"]
        for i, u in enumerate(units):
            # Spawn points only: every unit is teleported to its plan after deployment.
            sign = -1 if side == "own" else 1
            xml_units.append({"script_name": names[u["id"]], "key": u["key"], "men": u["men"],
                              "general": u["commanding"] and u["men"] == 1,
                              "x": ax - 37.5 + i * 15, "z": az + sign * 30})
        xml_sides[side] = xml_units
        if side == "own":
            for u in units:
                u["id"] = names[u["id"]]
            # Lua config literals have no null: unknown fields are left out.
            units = [{k: v for k, v in u.items() if v is not None} for u in units]
            config["own"] = {"anchor": {"x": ax, "z": az}, "units": units}
            config["role"] = army["own"].get("role")
        else:
            config["enemy"] = {"anchor": {"x": ax, "z": az},
                               "units": [{k: v for k, v in u.items() if v is not None} for u in units],
                               "placements": [
                {"script_name": names[p["id"]], "role": p["role"], "x": p["x"], "z": p["z"], "bearing": p["bearing"],
                 "width": p.get("width") or 5} for p in sides["enemy"]["placements"]]}
    return probe_xml(xml_sides), config, sides


PROBE_UNIT = """      <unit num_soldiers="{men}" script_name="{script_name}">
        <unit_type type="{key}"/>
        <position x="{x}" y="{z}"/><orientation radians="0"/><width metres="10"/>
        <unit_experience level="0"/>
{general}      </unit>
"""


def probe_xml(xml_sides, faction="wh_main_emp_empire"):
    head = roster.SCENARIO_HEAD
    # Reuse the roster scenario frame; swap its fixed enemy army for ours.
    start = head.index('  <alliance id="1">')
    end = head.index("  </alliance>", start) + len("  </alliance>\n")
    blocks = {}
    for side, units in xml_sides.items():
        blocks[side] = "".join(PROBE_UNIT.format(
            men=int(u["men"]), script_name=u["script_name"], key=u["key"], x=u["x"], z=u["z"],
            general=f'        <general><name>{side}</name><star_rating level="1"/></general>\n' if u["general"] else "")
            for u in units)
    enemy_alliance = head[head.index('  <alliance id="0">'):head.index('  <alliance id="1">')]
    enemy_alliance = enemy_alliance.replace('<alliance id="0">', '<alliance id="1">') \
        .replace('<orientation radians="4.71"/>', '<orientation radians="1.57"/>') \
        .replace('<rout_position x="600" y="0"/>', '<rout_position x="-600" y="0"/>') \
        .replace("{units}", blocks["enemy"])
    text = head[:start] + enemy_alliance + head[end:]
    text = text.replace("Generated by tools/roster.py from config/roster/capture.json. Edit the\n"
                        "     list there, not this file. Side 1: units whose card and formation are\n"
                        "     read (src/entries/roster_capture.lua). Side 2: a general held by script\n"
                        "     far away, only because a battle needs two armies.",
                        "Generated by tools/sim/formation.py from config/armies/. Side 1: our\n"
                        "     army, placed by src/apps/formation in battle (src/entries/formation_probe.lua).\n"
                        "     Side 2: the enemy, teleported to its simulated plan and held.")
    return text.format(faction=faction, units=blocks["own"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("army")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    army, name = load_army(args.army)
    sides, units_by_id = simulate(army)
    out = args.out or project.ROOT / "research" / "analysis" / "formation" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(sides, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    own = sides["own"]
    if own["strategy"] == "none":
        print(json.dumps({"status": own["status"], "decision": own["decision"]}, ensure_ascii=False, indent=1))
        return 0
    c = own["choice"]
    title = (f'{own["strategy"]}: {own["status"]}. Стена {c["wall_width"]} м ({c["wall_front_m"]:.0f}×{c["wall_depth_m"]:.0f}), '
             f'лучники {c["archer_width"]} м в {c["rows"]} ряд(а), запас дальности {c["min_reach_m"]:.0f} м\n'
             f'пунктир — место для разворота лучников; бледные — враг (для картинки)')
    draw(out / "plan.png", sides, units_by_id, title)
    print(json.dumps({"status": own["status"], "choice": c, "overlaps": own["overlaps"],
                      "top_options": own["options"][:5]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

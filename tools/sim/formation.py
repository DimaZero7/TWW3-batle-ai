"""Formation simulation: the same Lua code as in battle places an army.

    python -m tools.sim.formation <army config> [--out DIR]

<army config>: a name in config/armies/ (first_attack) or a path. Units come
from data/roster (measured card and formation). Our side goes through the same
chain as in battle, src/apps/plan: assessment -> strategy -> formation, facing
the enemy anchor. The enemy is drawn with the "wall and arc" roles and layout,
for the picture only (in battle the game AI places it). Then src/apps/vision
sees the enemy (soldiers laid out evenly in each unit's rectangle, strength from
apps.assessment) and finds the groups of both sides; src/apps/battlefield
builds the field between the two main groups (axis, front lines, 40 m margin).

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
        self._groups = self.lua.eval("""
            function(units, total)
                return require('apps.core.json').encode(require('apps.vision.services').groups(units, nil, total))
            end
        """)
        self._battle_picture = self.lua.eval("""
            function(input)
                return require('apps.core.json').encode(require('apps.vision.services').picture(input))
            end
        """)
        self._battlefield = self.lua.eval("""
            function(input)
                return require('apps.core.json').encode(require('apps.battlefield.services').frame(input))
            end
        """)
        self._strength = self.lua.eval("function(u) return require('apps.assessment.services').strength(u) end")

    def strength(self, unit):
        return self._strength(self.lua.table_from(unit, recursive=True))

    def groups(self, units, total=None):
        """apps.vision.groups on {id, points, strength} units."""
        return json.loads(self._groups(self.lua.table_from(units, recursive=True), total))

    def battlefield(self, own_points, own_centre, enemy_points, enemy_centre):
        """apps.battlefield.frame between the two main groups."""
        data = {"own": {"points": own_points, "centre": own_centre},
                "enemy": {"points": enemy_points, "centre": enemy_centre}}
        return json.loads(self._battlefield(self.lua.table_from(data, recursive=True)))

    def battle_picture(self, own, enemy, enemy_total=None):
        """apps.vision.picture: {own, enemy} groups."""
        data = {"own": own, "enemy": enemy}
        if enemy_total:
            data["enemy_total"] = enemy_total
        return json.loads(self._battle_picture(self.lua.table_from(data, recursive=True)))

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


def soldier_points(p, men):
    """Soldiers laid out evenly in a placement's rectangle (the simulation has no real soldiers)."""
    if men <= 1:
        return [p["x"], p["z"]]
    front, depth = max(p["front_m"], 0.1), max(p["depth_m"], 0.1)
    cols = max(1, round(math.sqrt(men * front / depth)))
    rows = math.ceil(men / cols)
    b = math.radians(p["bearing"])
    f, r = (math.sin(b), math.cos(b)), (math.cos(b), -math.sin(b))
    pts = []
    for k in range(men):
        i, j = k % cols, k // cols
        along = -front / 2 + front * (i + 0.5) / cols
        back = depth * (j + 0.5) / rows
        pts += [p["x"] + r[0] * along - f[0] * back, p["z"] + r[1] * along - f[1] * back]
    return pts


def seen_units(planner, placements, units_by_id):
    """{id, points, strength} for apps.vision from simulated placements."""
    return [{"id": p["id"], "points": soldier_points(p, units_by_id[p["id"]]["men"]),
             "strength": planner.strength(units_by_id[p["id"]])} for p in placements]


def see_battle(planner, own_placements, own_by_id, enemy_placements, enemy_by_id):
    """apps.vision for both simulated sides (all of the enemy is seen here), then
    apps.battlefield between the two main groups. Returns (picture, field)."""
    own = seen_units(planner, own_placements, own_by_id)
    enemy = seen_units(planner, enemy_placements, enemy_by_id)
    picture = planner.battle_picture(own, enemy, sum(u["strength"] for u in enemy))
    field = None
    if picture["own"].get("main") and picture["enemy"].get("main"):
        def points(units, group):
            ids = set(group["ids"])
            return [v for u in units if u["id"] in ids for v in u["points"]]
        field = planner.battlefield(points(own, picture["own"]["main"]), picture["own"]["main"]["centre"],
                                    points(enemy, picture["enemy"]["main"]), picture["enemy"]["main"]["centre"])
    return picture, field


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
    field = sides.get("battlefield")
    if field:
        c = field["corners"]
        ax.add_patch(Polygon([(q["x"], q["z"]) for q in c], closed=True, fill=False, ec="#9467bd", lw=1.2, ls="-."))
        b = math.radians(field["bearing"])
        f, r = (math.sin(b), math.cos(b)), (math.cos(b), -math.sin(b))
        o, h = field["origin"], field["half_width_m"]
        ax.plot([o["x"] + f[0] * field["own"]["back_m"], o["x"] + f[0] * field["enemy"]["back_m"]],
                [o["z"] + f[1] * field["own"]["back_m"], o["z"] + f[1] * field["enemy"]["back_m"]],
                c="#9467bd", lw=0.8, ls="--")
        for along in (field["own"]["front_m"], field["enemy"]["front_m"]):
            ax.plot([o["x"] + f[0] * along - r[0] * h, o["x"] + f[0] * along + r[0] * h],
                    [o["z"] + f[1] * along - r[1] * h, o["z"] + f[1] * along + r[1] * h], c="#9467bd", lw=0.6)
        ax.annotate(f'поле боя: между передними линиями {field["gap_m"]:.0f} м, '
                    f'ширина {2 * h:.0f} м (запас {field["margin_m"]:.0f} м)',
                    (min(q["x"] for q in c), min(q["z"] for q in c)), fontsize=8, xytext=(4, 4),
                    textcoords="offset points", color="#9467bd")
    for side in ("own", "enemy"):
        plan = sides[side]
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
    for side in ("own", "enemy"):
        for p in sides[side]["placements"]:
            if p["role"] == "lord":
                ax.scatter([p["x"]], [p["z"]], marker="*", s=260, c=COLORS["lord"], ec="k", zorder=5,
                           alpha=1 if side == "own" else 0.45)
                ax.annotate("наш лорд" if side == "own" else "лорд врага", (p["x"], p["z"]), fontsize=8,
                            xytext=(6, 6), textcoords="offset points")
    for side, label in (("own", "мы"), ("enemy", "враг")):
        vision = sides[side].get("vision")
        if not vision or not vision.get("main"):
            continue
        for n, h in enumerate(vision["groups"]):
            b = h["bounds"]
            ax.add_patch(plt.Rectangle((b["min_x"] - 3, b["min_z"] - 3), b["max_x"] - b["min_x"] + 6,
                                       b["max_z"] - b["min_z"] + 6, fill=False, ls=":", lw=1.2,
                                       ec="k" if n == 0 else "#7f7f7f"))
        main = vision["main"]
        ax.scatter([main["centre"]["x"]], [main["centre"]["z"]], marker="X", s=120, c="k", zorder=6)
        ax.annotate(f'{label}: основная армия ({len(main["ids"])} отр.)', (main["centre"]["x"], main["centre"]["z"]),
                    fontsize=8, xytext=(8, -12), textcoords="offset points")
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
    enemy_by_id = {u["id"]: u for u in enemy_units}
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
    # How we see the battle: groups of both sides and their main armies (apps.vision).
    own_by_id = {u["id"]: u for u in own_units}
    picture, field = see_battle(planner, own["placements"], own_by_id, sides["enemy"]["placements"], enemy_by_id)
    sides["own"]["vision"], sides["enemy"]["vision"] = picture["own"], picture["enemy"]
    sides["battlefield"] = field
    return sides, own_by_id


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

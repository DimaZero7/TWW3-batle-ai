"""When do archers start shooting, by the depth of their block (src/entries/archer_range.lua).

    python -m tools.analysis.archer_range <run dir> [<run dir> ...]

For every lane: the unit's width and depth (its soldiers along the lane), the
distance from its front rank and from its rear rank to the target's nearest
rank when the engine first said unit_in_range and when the first arrow was
spent (ammo went down), and whether it ever fired. Prints a Markdown table.
"""
import argparse
import json
import sys
from pathlib import Path


def lanes_of(run):
    rows = [json.loads(line) for line in (run / "events.jsonl").read_text(encoding="utf-8").splitlines() if line]
    mode = next((r.get("mode") for r in rows if r["event"] == "start"), None)
    errors = [r["message"][:160] for r in rows if r["event"] == "error"]
    samples = [r for r in rows if r["event"] == "range_sample"]
    out = {}
    for s in samples:
        for lane in s["lanes"]:
            o = out.setdefault(lane["lane"], {"width": lane["width"], "first_ammo": lane["ammo"]})
            if lane.get("archer_front") is None or lane.get("target_near") is None:
                continue
            front = lane["target_near"] - lane["archer_front"]
            rear = lane["target_near"] - lane["archer_rear"]
            o["depth"] = round(lane["archer_front"] - lane["archer_rear"], 1)
            o["range"] = lane.get("range")
            if lane.get("in_range") and "in_range" not in o:
                o["in_range"] = (round(front, 1), round(rear, 1), s["tick"])
            if lane.get("ammo") is not None and o["first_ammo"] is not None and lane["ammo"] < o["first_ammo"] \
                    and "fired" not in o:
                o["fired"] = (round(front, 1), round(rear, 1), s["tick"], lane.get("moving"))
            if lane.get("firing") and "firing" not in o:
                o["firing"] = (round(front, 1), round(rear, 1), s["tick"])
            o["last"] = (round(front, 1), round(rear, 1), s["tick"])
    return mode, errors, out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    print("| Режим | Ширина, м | Глубина, м | Дальность | «В радиусе»: от переднего / заднего ряда | Первая стрела: от переднего / заднего ряда | Такт |")
    print("|---|---|---|---|---|---|---|")
    for run in args.runs:
        mode, errors, lanes = lanes_of(run)
        if errors:
            print(f"| {mode} | ошибка: {errors[0]} | | | | | |")
        for i in sorted(lanes):
            o = lanes[i]
            ir = o.get("in_range")
            fi = o.get("fired")
            print(f'| {mode} | {o["width"]} | {o.get("depth")} | {o.get("range")} | '
                  f'{f"{ir[0]} / {ir[1]}" if ir else "нет"} | '
                  f'{f"{fi[0]} / {fi[1]}" if fi else "не стрелял (последнее: " + str(o.get("last")) + ")"} | '
                  f'{fi[2] if fi else ""} |')
    return 0


if __name__ == "__main__":
    sys.exit(main())

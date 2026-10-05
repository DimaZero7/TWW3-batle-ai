"""A human's liveliness in a real battle, by the gate's definitions (docs/en/launch/run.md "A battle
played by a human"): how many orders a human gives, next to our network's numbers from the gate.

    python -m tools.nn.human_orders                       # the newest build/human/runs/<time>
    python -m tools.nn.human_orders build/human/runs/<time> [--gate build/nn-gate/<time>]

A human gives no nn_orders, so the orders are inferred from the per-second recording (nn_sample) of
the unit's ordered point (ox, oz: unit:ordered_position()) and its own current target (t):
  * an order: a standing unit's ordered point moved more than ORDER_EPS_M since the last second (the
    manual target's rule, src/entries/manual_record.lua). Not on the first second a unit stands (the
    start, a rally: the engine sets the point, not the player);
  * its kind: an attack on the unit's current target when it has one (after the network's moves the
    target is empty in 2772 of 2772 cases, after its attacks set in 393 of 444: 2 gate battles), a
    hold when the unit stands still within HOLD_M of the new point, else a move to the point.
The inferred orders go into tools/nn/gate.py liveliness unchanged (as nn_orders, status given), so
every number has the gate's definition: order changes, attack-target switches, flips A-B-A, move
jitter, out of melee, per unit-minute of our side (side 1). What the inference cannot see: an order
to the point already ordered (a repeated click), an attack whose ordered point did not move (a new
target from where the unit stands), an order given and replaced within one second. Checked on 4
network battles of the gate (inferred / real order changes): 911/890, 248/234, 12/11, 228/322 - the
last one (an older bridge that re-aimed shooters in place) undercounts switches 84/168. The same
inference runs on our network's recording of the same battle, so human and network are compared by
one method; the network's real numbers (its nn_orders) are printed next to it. The units' own target
switches (every second) need no inference.

Plain python (no torch): runs in the project's .venv.
"""
import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

from tools import config as project
from tools.nn import gate

ORDER_EPS_M = 1.0      # = src/entries/manual_record.lua ORDER_MOVE_M
HOLD_M = 3.0           # a standing unit ordered within this of where it stands: a halt (hold)
RUNS = project.BUILD / "human" / "runs"


def _up(u):
    return (u.get("men") or 0) > 0 and not u.get("r") and not u.get("s")


def _xz(u, kx="x", kz="z"):
    x, z = u.get(kx), u.get(kz)
    return None if x is None or z is None else (float(x), float(z))


def infer(samples, side=1):
    """[(t_ms, [order])] from nn_sample events (dicts, in time order): the orders of the side's units
    seen as changes of the ordered point; an order: {"u", "k": "move"|"attack", "x", "z", "tg"?,
    "status": "given"}."""
    last, out = {}, []               # name -> the ordered point of the last second it stood, else None
    for ev in samples:
        units = ev.get("units") or []
        where = {u.get("n"): u for u in units}
        given = []
        for u in units:
            if u.get("side") != side:
                continue
            name, point = u.get("n"), _xz(u, "ox", "oz")
            if not _up(u) or point is None:
                last[name] = None
                continue
            before = last.get(name)
            last[name] = point
            if before is None or math.dist(point, before) <= ORDER_EPS_M:
                continue
            order = {"u": name, "k": "move", "x": point[0], "z": point[1], "status": "given"}
            target = where.get(u.get("t") or "")
            here = _xz(u)
            if target and target.get("side") != side:
                order.update(k="attack", tg=target["n"], x=None, z=None)
            elif not u.get("mv") and here and math.dist(here, point) <= HOLD_M:
                order.update(k="hold", x=None, z=None)
            given.append(order)
        if given:
            out.append((ev.get("t", 0), given))
    return out


def _samples(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if '"event":"nn_sample"' in line:
                yield json.loads(line)


def inferred_liveliness(path):
    """gate.liveliness of a recording with its orders inferred (infer) in place of any nn_orders."""
    events = Path(path)
    orders = infer(_samples(events))
    fd, tmp = tempfile.mkstemp(suffix=".jsonl", prefix="human_orders_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out, open(events, encoding="utf-8") as f:
            pending = iter(orders)
            nxt = next(pending, None)
            for line in f:
                if '"event":"nn_sample"' not in line:
                    continue
                t = json.loads(line).get("t", 0)
                out.write(line if line.endswith("\n") else line + "\n")
                # An order seen in this second's sample: after it, as the bridge logs it after the state.
                while nxt is not None and nxt[0] <= t:
                    out.write(json.dumps({"event": "nn_orders", "t": nxt[0], "orders": nxt[1]},
                                         separators=(",", ":")) + "\n")
                    nxt = next(pending, None)
        return gate.liveliness(tmp), sum(len(o) for _, o in orders)
    finally:
        os.unlink(tmp)


def newest_run(root=RUNS):
    runs = sorted(d for d in Path(root).glob("*") if (d / "events.jsonl").exists())
    return runs[-1] if runs else None


def _role(run):
    cfg = gate._json(Path(run) / "manifest.json").get("config", {})
    return cfg.get("own_role") or ("defend" if cfg.get("enemy_role") == "attack" else "attack"), cfg


def gate_reference(role, army, gates=gate.OUT, runs=project.BUILD / "nn-arena" / "runs"):
    """(the newest gate's summary folder with a battle in this role, that summary, the network's run of the
    same battle: same seed, swap and role in the newest gate that played it, or None)."""
    folders = sorted((d for d in Path(gates).glob("*") if (d / "summary.json").exists()), reverse=True)
    newest = same = None
    for d in folders:
        s = gate._json(d / "summary.json")
        battles = s.get("battles") or []
        if newest is None and any(b.get("role") == role and b.get("lively") for b in battles):
            newest = (d, s)
        if same is None and army:
            for b in battles:
                if (b.get("seed") == army.get("seed") and bool(b.get("swap")) == bool(army.get("swap"))
                        and b.get("role") == role and b.get("run") and (Path(runs) / b["run"]).exists()):
                    same = (d, s, Path(runs) / b["run"])
        if newest and same:
            break
    return newest, same


COLUMNS = (("order_changes_per_min", "order changes", "{:.2f}"), ("target_switches_per_min", "attack-target switches", "{:.2f}"),
           ("flips_per_min", "flips A-B-A", "{:.3f}"), ("move_jitter_m", "move jitter, m", "{:.1f}"),
           ("move_repoints_per_min", "move repoints", "{:.2f}"), ("free_changes_per_min", "out of melee", "{:.2f}"),
           ("twitch_share", "twitching units", "{:.2f}"), ("engine_switches_per_min", "own target switches", "{:.2f}"),
           ("engine_flips_per_min", "own target flips", "{:.3f}"))


def table(columns):
    """Text lines: one row per measure, one column per source ({title: rates})."""
    titles = list(columns)
    width = max(len(c[1]) for c in COLUMNS) + 2
    lines = [" " * width + "".join(f"{t:>24}" for t in titles)]
    for key, name, fmt in COLUMNS:
        cells = []
        for t in titles:
            v = (columns[t] or {}).get(key)
            cells.append("-" if v is None else fmt.format(v))
        lines.append(f"{name:<{width}}" + "".join(f"{c:>24}" for c in cells))
    return lines


def compare(run, gate_dir=None, gates=gate.OUT, runs=project.BUILD / "nn-arena" / "runs"):
    """{"run", "role", "columns": {title: rates}, "notes": [...]}: the human (inferred orders) next to the
    network (the same battle, inferred and real) and the gate's numbers of that role."""
    run = Path(run)
    role, cfg = _role(run)
    human, n_human = inferred_liveliness(run / "events.jsonl")
    columns = {"human (inferred)": gate.lively_rates((human or {}).get("net"))}
    notes = [f"human: {run.name}, our side {role}s, {n_human} orders inferred"]
    newest, same = gate_reference(role, cfg.get("army"), gates, runs)
    if gate_dir:
        s = gate._json(Path(gate_dir) / "summary.json")
        newest = (Path(gate_dir), s) if s else newest
    if same:
        d, s, net_run = same
        inferred, n_net = inferred_liveliness(net_run / "events.jsonl")
        real = gate.liveliness(net_run / "events.jsonl")
        columns["net same battle (inf.)"] = gate.lively_rates((inferred or {}).get("net"))
        columns["net same battle (real)"] = gate.lively_rates((real or {}).get("net"))
        notes.append(f"network, same battle: gate {d.name} ({s.get('checkpoint')}), run {net_run.name}, "
                     f"{n_net} orders inferred")
    if newest:
        d, s = newest
        lv = s.get("liveliness") or {}
        rates = dict((lv.get("net") or {}).get(role) or {})
        columns[f"net gate {role} (real)"] = rates
        notes.append(f"network, gate: {d.name} ({s.get('checkpoint')}), its {role} battles")
    game_ai = gate.lively_rates((human or {}).get("game_ai"))
    columns["game AI (this battle)"] = {k: v for k, v in game_ai.items() if k.startswith("engine_")}
    return {"run": str(run), "role": role, "columns": columns, "notes": notes}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", nargs="?", type=Path, help="a run folder (default: the newest build/human/runs/<time>)")
    parser.add_argument("--gate", type=Path, help="a gate folder for the network's numbers (default: the newest "
                        "with a battle in the human's role)")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    run = args.run or newest_run()
    if not run or not (Path(run) / "events.jsonl").exists():
        parser.error("no recording: give a run folder with events.jsonl")
    out = compare(run, args.gate)
    for line in out["notes"]:
        print(line)
    print("per unit-minute of our side; jitter in m (gate definitions, tools/nn/gate.py liveliness)")
    for line in table(out["columns"]):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())

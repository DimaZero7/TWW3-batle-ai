"""A battle's numbers for the chain-vs-human base (a run of build/ or data/ nn-arena or human runs): the winner, the
time, the men at the start and the end of each side, the gold of the units broken (routed or dead at the end: the unit's
multiplayer cost x its lost share), the first contact (a unit of each side in melee), own units that never moved.
    python -m game.battle.compare <run>..."""
import json
import sys

from game.battle import costs, run_dir


def one(run):
    U = costs()
    run = run_dir(run)
    m = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    keys = {u["script_name"]: u["key"] for s in ("own", "enemy") for u in m["config"]["units"][s]}
    samples, res = [], None
    for line in (run / "events.jsonl").open(encoding="utf-8"):
        d = json.loads(line)
        if d.get("event") == "nn_sample":
            samples.append(d)
        elif d.get("event") == "result":
            res = d
    first, last = samples[0], samples[-1]
    start = {u["n"]: u for u in first["units"]}
    end = {u["n"]: u for u in last["units"]}
    contact = next((round(s["t"] / 1000) for s in samples if any(u.get("m") for u in s["units"])), None)
    moved, fired = set(), set()
    for s in samples:
        for u in s["units"]:
            if u.get("mv"):
                moved.add(u["n"])
            if u.get("fire") or u.get("f") is True and False:
                fired.add(u["n"])
    out = {"run": run.name, "winner": res and res.get("winner"), "status": res and res.get("status"),
           "time_s": round((res or {}).get("duration_model_ms", last["t"]) / 1000), "first_contact_s": contact}
    for side in (1, 2):
        names = [n for n, u in start.items() if int(u.get("side") or 0) == side]
        men0 = sum(int(start[n].get("men") or 0) for n in names)
        men1 = sum(int(end[n].get("men") or 0) for n in names if not end[n].get("r"))
        gold0 = sum(U.get(keys.get(n, ""), {}).get("multiplayer_cost") or 0 for n in names)
        lost = 0.0
        for n in names:
            cost = U.get(keys.get(n, ""), {}).get("multiplayer_cost") or 0
            m0, m1 = int(start[n].get("men") or 0) or 1, int(end[n].get("men") or 0)
            lost += cost if end[n].get("r") or m1 == 0 else cost * (1 - m1 / m0)
        out[f"side{side}"] = {"men_start": men0, "men_standing_end": men1, "gold": gold0, "gold_lost": round(lost)}
        if side == 1:
            out["own_never_moved"] = [n for n in names if n not in moved]
    return out


if __name__ == "__main__":
    for r in sys.argv[1:]:
        print(json.dumps(one(r), ensure_ascii=False))

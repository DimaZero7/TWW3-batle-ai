"""A person's battles read as numbers for rules (no torch): the deployment relative to the army's own line, the order of
targets by the enemy's kind, the pace, the roles of the groups the bridge reads, the regrouping, the enemy AI's
behaviour. build/diag_label/human_rules.md is written from these (python -m tools.label.digest_rules <record> ...
-> one json of every record's numbers).

Kinds (by the passport, config/nn/units.json): lord (class com), melee (melee infantry), sniper (missile infantry with a
range of 250 m or more: jezzails, long rifles), shooters (other missile infantry), artillery, cavalry (cav_*),
monster (monsters and monstrous infantry), beasts (war beasts: wolf rats). fast: a run of 5 m/s or more.

The army's frame at the battle's first second: forward = from my units' centre to the enemy's, right = the person's right
(Frame.local: facing +x, +z); the line = my melee infantry: its forward (the mean) and its width (across).
"""
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from tools.label import bridge_infer as BI
from tools.label import digest as DG
from tools.label import pool as PL

KINDS = ("lord", "melee", "sniper", "shooters", "artillery", "cavalry", "monster", "beasts")
KIND_RU = {"lord": "лорд", "melee": "пехота", "sniper": "снайперы", "shooters": "стрелки", "artillery": "артиллерия",
           "cavalry": "конница", "monster": "монстры", "beasts": "звери (крысоволки)"}


def kind(key):
    p = PL.passports().get(key or "", {})
    cl, caste = p.get("class") or "", p.get("caste") or ""
    rng = ((p.get("missile") or {}).get("range_m") or 0)
    if cl == "com":
        return "lord"
    if cl.startswith("art"):
        return "artillery"
    if cl.startswith("cav"):
        return "cavalry"
    if caste == "war_beast":
        return "beasts"
    if caste in ("monster", "monstrous_infantry"):
        return "monster"
    if cl == "inf_mis":
        return "sniper" if rng >= 250 else "shooters"
    return "melee"


def fast(key):
    return float(((PL.passports().get(key or "", {}).get("speed") or {}).get("run")) or 0) >= 5.0


class Frame:
    def __init__(self, rec, sec=1):
        st = rec.state(sec)
        own = [r for r in st.values() if int(r["side"]) == 1 and (r.get("men") or 0) > 0]
        foe = [r for r in st.values() if int(r["side"]) == 2 and (r.get("men") or 0) > 0]
        self.cx = sum(float(r["x"]) for r in own) / len(own)
        self.cz = sum(float(r["z"]) for r in own) / len(own)
        ex = sum(float(r["x"]) for r in foe) / len(foe)
        ez = sum(float(r["z"]) for r in foe) / len(foe)
        d = math.hypot(ex - self.cx, ez - self.cz)
        self.fx, self.fz = (ex - self.cx) / d, (ez - self.cz) / d
        self.gap = d

    def local(self, x, z):
        """(right, forward) m of a world point. The game's world is left-handed seen from above (x, height y, z): facing
        +x, +z is on the right - the person's own words agree (M1: "shooters on the right flank, the forest on the
        left": the shooters stood at +z, the forest at -z)."""
        dx, dz = x - self.cx, z - self.cz
        return -dx * self.fz + dz * self.fx, dx * self.fx + dz * self.fz


def deployment(rec):
    fr = Frame(rec, 1)
    st = rec.state(1)
    out = {"gap_m": round(fr.gap), "units": []}
    pos = {}
    for n, r in st.items():
        if int(r["side"]) != 1:
            continue
        k = kind(rec.key_of.get(n))
        s, f = fr.local(float(r["x"]), float(r["z"]))
        pos[n] = (k, s, f)
    line = [(s, f) for k, s, f in pos.values() if k == "melee"]
    lf = sum(f for _, f in line) / len(line)
    ls0, ls1 = min(s for s, _ in line), max(s for s, _ in line)
    out["line"] = {"width_m": round(ls1 - ls0), "units": len(line), "depth_m": round(max(f for _, f in line) -
                                                                                     min(f for _, f in line))}
    by = defaultdict(list)
    for n, (k, s, f) in pos.items():
        third = "left" if s < ls0 + (ls1 - ls0) / 3 else ("right" if s > ls1 - (ls1 - ls0) / 3 else "centre")
        near = min((math.hypot(s - s2, f - f2) for n2, (k2, s2, f2) in pos.items() if k2 == "melee" and n2 != n),
                   default=None)
        by[k].append({"name": n, "ahead_of_line_m": round(f - lf), "right_m": round(s), "third": third,
                      "outside_line": s < ls0 - 10 or s > ls1 + 10, "nearest_melee_m": round(near) if near else None})
    out["by_kind"] = {k: {"n": len(v), "ahead_of_line_m": round(sum(x["ahead_of_line_m"] for x in v) / len(v)),
                          "thirds": dict(Counter(x["third"] for x in v)),
                          "outside_line": sum(x["outside_line"] for x in v), "units": v} for k, v in by.items()}
    cg = BI.ctrlg_of(rec)
    out["ctrlg"] = {str(g): dict(Counter(kind(rec.key_of.get(n)) for n in names)) for g, names in cg.items()}
    return out, fr


def targets(rec, fr):
    cmds = [c for c in BI.battle_commands(rec) if c["name"] == "Attack Unit" and c.get("target")]
    first = {}
    by_attacker = defaultdict(Counter)
    attacks = []
    for c in cmds:
        tk = kind(rec.key_of.get(c["target"]))
        first.setdefault(tk, float(c["sec"]))
        for u in c["units"]:
            ak = kind(rec.key_of.get(u))
            by_attacker[ak][tk] += 1
            attacks.append((float(c["sec"]), u, c["target"]))
    # focus: within a window of 15 s, how many of my units attacked the same target
    focus = Counter()
    for a in range(0, rec.end, 15):
        per = defaultdict(set)
        for s, u, t in attacks:
            if a <= s < a + 15:
                per[t].add(u)
        for t, us in per.items():
            if len(us) >= 3:
                focus[kind(rec.key_of.get(t))] += 1
    # the enemy's losses in order: the first rout and the death of each enemy unit, by kind
    gone = {}
    for t, st in sorted(rec.states.items()):
        for n, r in st.items():
            if int(r["side"]) == 2 and n not in gone and ((r.get("men") or 0) <= 0 or r.get("r")):
                gone[n] = t
    gone_by = defaultdict(list)
    for n, t in gone.items():
        gone_by[kind(rec.key_of.get(n))].append(t)
    return {"commands": len(cmds), "unit_attacks": len(attacks),
            "first_attack_s": {k: round(v) for k, v in sorted(first.items(), key=lambda kv: kv[1])},
            "by_attacker": {k: dict(v.most_common()) for k, v in by_attacker.items()},
            "focus_windows": dict(focus),
            "enemy_out_s": {k: sorted(v) for k, v in sorted(gone_by.items(), key=lambda kv: min(kv[1]))}}


def tempo(rec):
    out = defaultdict(Counter)
    for c in BI.battle_commands(rec):
        if c["name"] not in BI.ORDER_COMMANDS:
            continue
        for u, o in (c.get("ordered") or {}).items():
            out[f"{c['name']} · {kind(rec.key_of.get(u))}"]["run" if o.get("fast") else "walk"] += 1
    return {k: dict(v) for k, v in out.items()}


def roles(rec, wins):
    """Per inferred group and window: its role by what it was and did."""
    cnt = Counter()
    for w in wins:
        for k, g in w.groups.items():
            ks = Counter(kind(rec.key_of.get(n)) for n in g)
            tk = {kind(rec.key_of.get(t)) for t in w.targets.get(k, [])}
            a = w.area.get(k) or {}
            moved = False
            s0, s1 = rec.state(w.sec), rec.state(w.until)
            d = [math.hypot(float(s1[n]["x"]) - float(s0[n]["x"]), float(s1[n]["z"]) - float(s0[n]["z"]))
                 for n in g if n in s0 and n in s1]
            moved = bool(d) and sum(d) / len(d) >= BI.MOVE_M
            ranged = sum(ks[x] for x in ("sniper", "shooters", "artillery"))
            if "lord" in tk:
                r = "охота на лорда"
            elif "artillery" in tk and not ranged:
                r = "охота на артиллерию"
            elif tk & {"beasts", "cavalry"} or any(fast(rec.key_of.get(t)) for t in w.targets.get(k, [])) and not ranged:
                r = "против быстрых"
            elif ranged == len(g):
                r = "огонь"
            elif not moved:
                r = "держит линию"
            else:
                r = "наступает" if tk else "манёвр"
            cnt[r] += 1
            _ = a
    return dict(cnt.most_common())


def regroup(rec, wins):
    dec = Counter(x for w in wins for x in w.why)
    return {"windows": len(wins), "decisions": sum(w.decision == "human" for w in wins), "why": dict(dec),
            "groups_mean": round(sum(len(w.groups) for w in wins) / max(len(wins), 1), 1),
            "group_size": dict(sorted(Counter(len(g) for w in wins for g in w.groups.values()).items()))}


def enemy_ai(rec, fr):
    """What the game's AI did: its centre's approach, its artillery's moves, the first contact, its routs."""
    def centre(sec, kinds=None):
        st = rec.state(sec)
        rows = [r for n, r in st.items() if int(r["side"]) == 2 and (r.get("men") or 0) > 0 and not r.get("r")
                and (kinds is None or kind(rec.key_of.get(n)) in kinds)]
        if not rows:
            return None
        return fr.local(sum(float(r["x"]) for r in rows) / len(rows), sum(float(r["z"]) for r in rows) / len(rows))
    out = {}
    c0 = centre(1)
    for t in (30, 60, 90, 120):
        c = centre(t)
        if c and c0:
            out[f"advance_{t}s_m"] = round(c0[1] - c[1])
    a0, a1 = centre(1, {"artillery"}), centre(120, {"artillery"})
    if a0 and a1:
        out["artillery_moved_120s_m"] = round(math.hypot(a1[0] - a0[0], a1[1] - a0[1]))
    sn0, sn1 = centre(1, {"sniper"}), centre(120, {"sniper"})
    if sn0 and sn1:
        out["snipers_moved_120s_m"] = round(math.hypot(sn1[0] - sn0[0], sn1[1] - sn0[1]))
    out["first_melee_s"] = min((t for t, st in rec.states.items() if any(r.get("m") for r in st.values())), default=None)
    men = lambda st, s: sum(int(r.get("men") or 0) for r in st.values() if int(r["side"]) == s)   # noqa: E731
    out["men_start"] = [men(rec.state(1), 1), men(rec.state(1), 2)]
    out["men_end"] = [men(rec.state(rec.end), 1), men(rec.state(rec.end), 2)]
    # the enemy lord
    lord = next((n for n in rec.names if rec.side_of.get(n) == 2 and kind(rec.key_of.get(n)) == "lord"), None)
    if lord:
        lr = [(t, st[lord]) for t, st in sorted(rec.states.items()) if lord in st]
        out["enemy_lord_rout_s"] = next((t for t, r in lr if r.get("r")), None)
        out["enemy_lord_dead_s"] = next((t for t, r in lr if (r.get("men") or 0) <= 0), None)
    # the cascade: enemy units out by time
    st0 = rec.state(1)
    n2 = sum(1 for r in st0.values() if int(r["side"]) == 2)
    curve = []
    for t in range(0, rec.end + 1, 30):
        st = rec.state(t)
        curve.append([t, sum(1 for r in st.values() if int(r["side"]) == 2 and (r.get("men") or 0) > 0 and not r.get("r")),
                      sum(1 for r in st.values() if int(r["side"]) == 1 and (r.get("men") or 0) > 0 and not r.get("r"))])
    out["standing_curve"] = curve
    out["enemy_units"] = n2
    return out


def retreats(rec, fr):
    """My shooters / snipers stepping back (>= 20 m away from the enemy in a window) while an enemy is within 120 m."""
    ev = []
    for a in range(1, rec.end - 15, 15):
        s0, s1 = rec.state(a), rec.state(a + 15)
        for n, r in s0.items():
            if int(r["side"]) != 1 or kind(rec.key_of.get(n)) not in ("sniper", "shooters", "artillery") or n not in s1:
                continue
            _, f0 = fr.local(float(r["x"]), float(r["z"]))
            _, f1 = fr.local(float(s1[n]["x"]), float(s1[n]["z"]))
            near = min((math.hypot(float(e["x"]) - float(r["x"]), float(e["z"]) - float(r["z"])) for e in s0.values()
                        if int(e["side"]) == 2 and (e.get("men") or 0) > 0 and not e.get("r")), default=1e9)
            if f0 - f1 >= 20 and near <= 120:
                ev.append({"sec": a, "unit": n, "back_m": round(f0 - f1), "enemy_m": round(near)})
    return ev


def lord_use(rec, fr):
    lord = next((n for n in rec.names if rec.side_of.get(n) == 1 and kind(rec.key_of.get(n)) == "lord"), None)
    if not lord:
        return None
    path = []
    for t in range(0, rec.end + 1, 30):
        r = rec.state(t).get(lord)
        if r:
            s, f = fr.local(float(r["x"]), float(r["z"]))
            path.append([t, round(s), round(f), int(r.get("men") or 0)])
    atk = [(round(float(c["sec"])), c["target"]) for c in BI.battle_commands(rec)
           if c["name"] == "Attack Unit" and lord in c["units"]]
    abil = [(round(float(c["sec"])), c.get("string1")) for c in BI.battle_commands(rec) if c["name"] == "Special Ability"]
    return {"path_right_forward": path, "attacks": atk, "abilities": abil}


def analyse(folder):
    rec = DG.Record(folder)
    dep, fr = deployment(rec)
    wins = BI.infer(rec)
    return {"record": DG.rel(rec.folder), "end_s": rec.end, "result": rec.run_result, "deployment": dep,
            "targets": targets(rec, fr), "tempo": tempo(rec), "roles": roles(rec, wins), "regroup": regroup(rec, wins),
            "enemy_ai": enemy_ai(rec, fr), "retreats": retreats(rec, fr), "lord": lord_use(rec, fr),
            "chat": [(c.get("sec"), c.get("text")) for c in DG._jsonl(rec.folder / "chat.jsonl")]}


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    argv = argv or sys.argv[1:]
    out = {Path(f).name: analyse(f) for f in argv if not f.startswith("--")}
    dst = Path("build/diag_label/human_rules.json")
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(dst)


if __name__ == "__main__":
    main()

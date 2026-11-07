"""The bridge: the levels' decisions read off a person's real battle, even where he did not label them (a live labelling
record, tools/label/live.py, and its run; no torch for the reading, torch for the walk's check).

The person groups once (Ctrl+G at the deployment) and then plays the units as he likes; the units' real company changes
under the groups he made. This module reads, per operative window (15 s), which units actually worked together:

    a link between two of my units in a window, any of
      order     the same command of the game (commands.jsonl: 'Attack Unit' / 'Move Orientation Width' given to both at
                once - the game's own multi-selection)
      target    the same enemy attacked (a command's target, or the unit's current target while it fights or fires)
      march     both moved (>= MOVE_M) and stayed together: within NEAR_M at the window's start and end, their moves
                differing by <= SAME_M
      line      neither moved and they stand within LINE_M at the window's end (a held line)
    the groups  the links' connected components; a component over MAX_GROUP (10, the tactician's) is cut in two along its
                longest side, over and over; more than G (6, the operative's) groups: the two nearest merged until G;
                the numbers kept from window to window by the largest overlap
    the targets per group the enemies its members attacked or fought in the window (counted; the seed the most often)
    the area    the box round its members' places over the window and their ordered points, along the group's facing
    a decision  the operative decides in a window when MOVED_UNITS (2) or more units are in other groups than the window
                before (one unit changing company is the links' noise), or a group's target seed changed and stays
                so in the next window; else 'keep'

    infer(rec)                        [Window]: groups {k: [names]}, targets {k: [names]}, seed {k}, area {k}, decision
    compare(rec, wins)                the person's Ctrl+G groups against the inferred: per Ctrl+G group the share of
                                      windows its alive members were in one inferred group, and its usual company
    unit_orders(rec)                  the unit level's orders from commands.jsonl (and what bool1 / target / fast mean)
    python -m tools.label.bridge_infer build/label/<date>_live [--out DIR]   -> bridge.json, bridge.md, bridge_<k>.png
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from tools.label import digest as DG

ORDER_COMMANDS = ("Attack Unit", "Move Orientation Width")
MOVE_M = 20.0          # m in a window: the unit moved
NEAR_M = 60.0          # m: two units together
SAME_M = 25.0          # m: two moves the same
LINE_M = 45.0          # m: neighbours of a held line (formations 20-40 m wide side by side)
MAX_GROUP = 10         # the tactician's group (config/nn/tactic.json max_group)
G = 6                  # the operative's groups (config/nn/operative.json groups)
MOVED_UNITS = 2        # units in other groups than in the window before: a decision of the groups


def battle_commands(rec):
    """[command] of the battle (the deployment's commands come first in the file with the deployment's own clock: they
    are dropped - everything before the first drop of the seconds)."""
    rows = DG._jsonl(rec.folder / "commands.jsonl")
    out, cut = [], 0
    for i in range(1, len(rows)):
        if float(rows[i]["sec"]) + 1e-6 < float(rows[i - 1]["sec"]):
            cut = i
            break
    for c in rows[cut:] if cut else rows:
        out.append(c)
    return out


@dataclass
class Window:
    sec: int
    until: int
    groups: dict = field(default_factory=dict)      # {k: [names]}
    targets: dict = field(default_factory=dict)     # {k: [enemy names]}
    seed: dict = field(default_factory=dict)        # {k: enemy name}
    area: dict = field(default_factory=dict)        # {k: {cx, cz, heading, front, depth}}
    links: dict = field(default_factory=dict)       # {kind: count}
    moved: list = field(default_factory=list)       # units in another group than the window before
    decision: str = "keep"
    why: list = field(default_factory=list)


def _alive(rec, st, n):
    r = st.get(n)
    return r is not None and (r.get("men") or 0) > 0 and not r.get("r") and not r.get("s")


def _components(nodes, edges):
    parent = {n: n for n in nodes}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in edges:
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    comp = defaultdict(list)
    for n in nodes:
        comp[find(n)].append(n)
    return [sorted(v) for v in comp.values()]


def _cut(group, pos):
    """Cut a group over MAX_GROUP in two along its longest side (repeatedly)."""
    if len(group) <= MAX_GROUP:
        return [group]
    xs = [pos[n][0] for n in group]
    zs = [pos[n][1] for n in group]
    axis = 0 if max(xs) - min(xs) >= max(zs) - min(zs) else 1
    srt = sorted(group, key=lambda n: pos[n][axis])
    h = len(srt) // 2
    return _cut(srt[:h], pos) + _cut(srt[h:], pos)


def _centre(group, pos):
    return (sum(pos[n][0] for n in group) / len(group), sum(pos[n][1] for n in group) / len(group))


def _merge_to(groups, pos, limit=G):
    groups = [list(g) for g in groups]
    while len(groups) > limit:
        best = None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if len(groups[i]) + len(groups[j]) > MAX_GROUP:
                    continue
                ci, cj = _centre(groups[i], pos), _centre(groups[j], pos)
                d = math.hypot(ci[0] - cj[0], ci[1] - cj[1])
                if best is None or d < best[0]:
                    best = (d, i, j)
        if best is None:
            break
        _, i, j = best
        groups[i] = sorted(groups[i] + groups[j])
        del groups[j]
    return groups


def _number(groups, prev):
    """{k: group} numbered 1..: each takes the previous number it shares most units with, then the least free."""
    out, used = {}, set()
    order = sorted(groups, key=len, reverse=True)
    for g in order:
        best = max(((len(set(g) & set(v)), k) for k, v in (prev or {}).items() if k not in used), default=(0, None))
        k = best[1] if best[0] > 0 else None
        if k is None:
            k = next(i for i in range(1, 100) if i not in used and i not in out)
        used.add(k)
        out[k] = g
    return dict(sorted(out.items()))


def infer(rec, window_s=15):
    cmds = [c for c in battle_commands(rec) if c.get("name") in ORDER_COMMANDS]
    own = [n for n in rec.names if rec.side_of.get(n) == 1]
    wins = []
    prev = None
    end = rec.end
    for a in range(1, end, window_s):
        b = min(a + window_s, end)
        s0, s1 = rec.state(a), rec.state(b)
        alive = [n for n in own if _alive(rec, s1, n) and n in s0]
        if not alive:
            break
        p0 = {n: (float(s0[n]["x"]), float(s0[n]["z"])) for n in alive}
        p1 = {n: (float(s1[n]["x"]), float(s1[n]["z"])) for n in alive}
        disp = {n: (p1[n][0] - p0[n][0], p1[n][1] - p0[n][1]) for n in alive}
        edges, kinds = [], Counter()
        wc = [c for c in cmds if a <= float(c["sec"]) < b]
        hit = defaultdict(Counter)                                # my unit -> enemies attacked / fought
        for c in wc:
            us = [u for u in c.get("units") or [] if u in p0]
            for i in range(len(us)):
                for j in range(i + 1, len(us)):
                    edges.append((us[i], us[j]))
                    kinds["order"] += 1
            if c.get("target") and c["name"] == "Attack Unit":
                for u in us:
                    hit[u][c["target"]] += 3
        for s in range(a, b):
            st = rec.states.get(s) or {}
            for n in alive:
                r = st.get(n) or {}
                t = r.get("t") or ""
                if t and rec.side_of.get(t) == 2 and (r.get("m") or r.get("fire")):
                    hit[n][t] += 1
        by_target = defaultdict(list)
        for n, cnt in hit.items():
            by_target[cnt.most_common(1)[0][0]].append(n)
        for t, us in by_target.items():
            for i in range(len(us)):
                for j in range(i + 1, len(us)):
                    edges.append((us[i], us[j]))
                    kinds["target"] += 1
        for i, n in enumerate(alive):
            for m in alive[i + 1:]:
                d0 = math.dist(p0[n], p0[m])
                d1 = math.dist(p1[n], p1[m])
                mn, mm = math.hypot(*disp[n]), math.hypot(*disp[m])
                if mn >= MOVE_M and mm >= MOVE_M and d0 <= NEAR_M and d1 <= NEAR_M and \
                        math.dist(disp[n], disp[m]) <= SAME_M:
                    edges.append((n, m))
                    kinds["march"] += 1
                elif mn < MOVE_M and mm < MOVE_M and d1 <= LINE_M:
                    edges.append((n, m))
                    kinds["line"] += 1
        comps = _components(alive, edges)
        comps = [g for c in comps for g in _cut(c, p1)]
        comps = _merge_to(comps, p1)
        groups = _number(comps, prev.groups if prev else None)
        w = Window(a, b, groups=groups, links=dict(kinds))
        for k, g in groups.items():
            cnt = Counter()
            for n in g:
                cnt.update(hit.get(n, {}))
            if cnt:
                top = cnt.most_common(1)[0][1]
                w.targets[k] = [t for t, v in cnt.most_common() if v >= 0.25 * top]
                w.seed[k] = cnt.most_common(1)[0][0]
            w.area[k] = area(rec, g, a, b, wc)
        if prev is None:
            w.decision, w.why = "human", ["the first window"]
        else:
            # (a unit that died or ran is no decision; one unit changing company is noise of the links: a decision
            # needs MOVED_UNITS or more units in other groups, or a group's target seed changed and still the same in
            # the next window - checked when that window comes, below)
            num = {n: k for k, g in groups.items() for n in g}
            was = {n: k for k, g in prev.groups.items() for n in g}
            moved = [n for n in num if n in was and num[n] != was[n]]
            w.moved = moved
            if len(moved) >= MOVED_UNITS:
                w.decision, w.why = "human", ["groups"]
        if len(wins) >= 2:
            # the target seed of a group changed in the previous window and stayed: that window decided it
            p2, p1 = wins[-2], wins[-1]
            for k, t in p1.seed.items():
                if t != p2.seed.get(k) and w.seed.get(k) == t and p1.decision == "keep":
                    p1.decision, p1.why = "human", ["targets"]
                    break
        wins.append(w)
        prev = w
    return wins


def area(rec, g, a, b, wc):
    """{cx, cz, heading, front, depth}: the box round the group's places over the window and its ordered points, along
    its mean facing."""
    pts, sx, sz = [], 0.0, 0.0
    for s in range(a, b + 1):
        st = rec.states.get(s) or {}
        for n in g:
            r = st.get(n)
            if r:
                pts.append((float(r["x"]), float(r["z"])))
                sx += math.sin(math.radians(float(r.get("b") or 0)))
                sz += math.cos(math.radians(float(r.get("b") or 0)))
    for c in wc:
        for n, o in (c.get("ordered") or {}).items():
            if n in g and o.get("ox") is not None and c["name"] == "Move Orientation Width":
                pts.append((float(o["ox"]), float(o["oz"])))
    if not pts:
        return None
    h = math.degrees(math.atan2(sx, sz)) % 360
    t = math.radians(h)
    loc = [((x * math.cos(t) - z * math.sin(t)), (x * math.sin(t) + z * math.cos(t))) for x, z in pts]
    s0, s1 = min(p[0] for p in loc), max(p[0] for p in loc)
    f0, f1 = min(p[1] for p in loc), max(p[1] for p in loc)
    ms, mf = (s0 + s1) / 2, (f0 + f1) / 2
    cx, cz = ms * math.cos(t) + mf * math.sin(t), -ms * math.sin(t) + mf * math.cos(t)
    return {"cx": round(cx, 1), "cz": round(cz, 1), "heading": round(h, 1), "front": round(s1 - s0 + 40, 1),
            "depth": round(f1 - f0 + 30, 1)}


def ctrlg_of(rec):
    """The person's own groups {k: [names]} (the deployment's, kept through the battle: the first window's)."""
    w = (rec.deploy or rec.windows or [{}])[0]
    return {int(k): v for k, v in ((w.get("groups") or {}).get("own") or {}).items()}


def compare(rec, wins):
    """Per Ctrl+G group: windows where its alive members (2+) were in one inferred group, and its usual company."""
    cg = ctrlg_of(rec)
    out = {}
    for k, names in sorted(cg.items()):
        together, n, company, split = 0, 0, Counter(), Counter()
        for w in wins:
            where = {m: g for g, ms in w.groups.items() for m in ms}
            alive = [m for m in names if m in where]
            if len(alive) < 2 and len(names) >= 2:
                continue
            n += 1
            gs = {where[m] for m in alive}
            together += len(gs) == 1
            split[len(gs)] += 1
            for g in gs:
                for m in w.groups[g]:
                    if m not in names:
                        company[m] += 1
        out[k] = {"members": names, "windows": n, "together": round(together / max(n, 1), 2),
                  "split_into": dict(sorted(split.items())),
                  "company": [(m, c) for m, c in company.most_common(4)]}
    return out


def unit_orders(rec):
    """What the commands give the unit level: per kind the count, bool1, the target, the run flag, and whether the
    run flag and the target show in the unit's state a little later (is_moving_fast; current_target)."""
    rows = battle_commands(rec)
    out = {}
    for name in sorted({c["name"] for c in rows}):
        cs = [c for c in rows if c["name"] == name]
        fast = Counter(v.get("fast") for c in cs for v in (c.get("ordered") or {}).values())
        b1 = Counter(c.get("bool1") for c in cs)
        tgt = sum(1 for c in cs if c.get("target"))
        seen_fast, fast_n, seen_t, t_n = 0, 0, 0, 0
        for c in cs:
            s = int(math.floor(float(c["sec"])))
            for u, o in (c.get("ordered") or {}).items():
                later = [((rec.states.get(s + d) or {}).get(u) or {}) for d in (1, 2, 3)]
                if o.get("fast") and name in ORDER_COMMANDS:
                    fast_n += 1
                    seen_fast += any(r.get("f") for r in later)
                if c.get("target") and name == "Attack Unit":
                    t_n += 1
                    seen_t += any(r.get("t") == c["target"] for r in later)
        out[name] = {"commands": len(cs), "units": sum(len(c.get("units") or []) for c in cs), "bool1": dict(b1),
                     "with_target": tgt, "fast": {str(k): v for k, v in fast.items()},
                     "fast_then_moving_fast": f"{seen_fast}/{fast_n}" if fast_n else None,
                     "target_then_current": f"{seen_t}/{t_n}" if t_n else None}
    return out


def summary_md(rec, wins, cmp, uo):
    L = ["# Мост: группы и решения по факту", ""]
    dec = Counter(w.decision for w in wins)
    why = Counter(x for w in wins for x in w.why)
    L.append(f"Окон {len(wins)}: решений {dec.get('human', 0)} (группы {why.get('groups', 0)}, цели "
             f"{why.get('targets', 0)}, первое 1), «оставить» {dec.get('keep', 0)}. Групп по факту в окне: "
             f"{round(sum(len(w.groups) for w in wins) / max(len(wins), 1), 1)}.")
    L += ["", "| Группа Ctrl+G | Состав | Окон | Вместе | Делилась на | Чаще всего с |", "|---|---|---|---|---|---|"]
    for k, c in cmp.items():
        L.append(f"| {k} | {DG.counted(rec, c['members'])} | {c['windows']} | {round(100 * c['together'])} % | "
                 f"{', '.join(f'{a}: {b}' for a, b in c['split_into'].items())} | "
                 f"{', '.join(DG.unit_name(rec, m) for m, _ in c['company'][:3])} |")
    L += ["", "| Команда | Раз | Отрядов | bool1 | С целью | Бег (fast) | Бег виден потом | Цель видна потом |",
          "|---|---|---|---|---|---|---|---|"]
    for k, v in uo.items():
        L.append(f"| {k} | {v['commands']} | {v['units']} | {v['bool1']} | {v['with_target']} | {v['fast']} | "
                 f"{v['fast_then_moving_fast'] or '—'} | {v['target_then_current'] or '—'} |")
    return "\n".join(L) + "\n"


def frames(rec, wins, out, picks):
    """bridge_<k>.png: the window's Ctrl+G groups (left) against the inferred (right)."""
    from PIL import Image, ImageDraw
    from tools.label import replay as RP
    cg = ctrlg_of(rec)
    paths = []
    for i in picks:
        w = wins[i]
        view = RP.View(rec, secs=[w.sec, w.until], margin=120)          # (zoomed on the window's battle)
        base = next((x for x in rec.windows if int(x["sec"]) == w.sec), None) or {}
        enemy = (base.get("groups") or {}).get("enemy") or {}
        left = RP.frame(rec, w.sec, {"sec": w.sec, "until": w.until, "source": "Ctrl+G",
                                     "groups": {"own": {str(k): v for k, v in cg.items()}, "enemy": enemy}}, view)
        right = RP.frame(rec, w.sec, {"sec": w.sec, "until": w.until, "source": "по факту",
                                      "groups": {"own": {str(k): v for k, v in w.groups.items()},
                                                 "enemy": {str(k): v for k, v in w.targets.items()}}}, view)
        img = Image.new("RGB", (left.width * 2 + 10, left.height), (255, 255, 255))
        img.paste(left, (0, 0))
        img.paste(right, (left.width + 10, 0))
        dr = ImageDraw.Draw(img)
        f = RP._font(22)
        dr.text((20, 40), "группы Ctrl+G (враг — его группы)", fill=(0, 0, 0), font=f)
        dr.text((left.width + 30, 40), "по факту: " + ("решение — " + ", ".join({"groups": "группы", "targets": "цели", "the first window": "первое окно"}.get(x, x) for x in w.why) if w.decision == "human" else "оставить") + " (враг — цели групп)", fill=(0, 0, 0), font=f)
        p = Path(out) / f"bridge_{i + 1:02d}_{w.sec}s.png"
        img.save(p)
        paths.append(p)
    return paths


def to_json(rec, wins, cmp, uo):
    return {"record": DG.rel(rec.folder), "windows": [{"sec": w.sec, "until": w.until, "decision": w.decision,
                                                       "why": w.why, "groups": w.groups, "targets": w.targets,
                                                       "seed": w.seed, "area": w.area, "links": w.links} for w in wins],
            "ctrlg": {str(k): v for k, v in ctrlg_of(rec).items()}, "compare": {str(k): v for k, v in cmp.items()},
            "unit_orders": uo,
            "rules": {"MOVE_M": MOVE_M, "NEAR_M": NEAR_M, "SAME_M": SAME_M, "LINE_M": LINE_M, "MAX_GROUP": MAX_GROUP,
                      "G": G}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder")
    ap.add_argument("--run", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--frames", default="", help="window numbers (1-based) for the side-by-side pictures, a,b,c")
    a = ap.parse_args(argv)
    rec = DG.Record(a.folder, a.run)
    out = Path(a.out) if a.out else rec.folder / "digest"
    out.mkdir(parents=True, exist_ok=True)
    wins = infer(rec)
    cmp = compare(rec, wins)
    uo = unit_orders(rec)
    (out / "bridge.json").write_text(json.dumps(to_json(rec, wins, cmp, uo), ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    (out / "bridge.md").write_text(summary_md(rec, wins, cmp, uo), encoding="utf-8")
    picks = [int(x) - 1 for x in a.frames.split(",") if x.strip()]
    paths = frames(rec, wins, out, picks) if picks else []
    print(json.dumps({"windows": len(wins), "decisions": sum(w.decision == "human" for w in wins),
                      "frames": [str(p) for p in paths]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

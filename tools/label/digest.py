"""A live labelling record (tools/label/live.py: build/label/<date>_live/) read back as three things:

    digest.md       a readable feed by the operative's windows (15 s): the time, my groups (who, where they stand, which
                    way, how wide), the enemy's groups, what changed, my orders in the window (compressed: "группа 1:
                    Тяжелые мечники ×3 идут к (−120, 40), бегом"), the chat's words, contacts and losses; the armies and
                    the map at the start, the result at the end
    frames          a picture a window (replay.py: the field from above, the units as formations with their men, my
                    groups by colour, the enemy's groups framed, the orders' arrows, the caption) and a gif of the battle
    samples.pt      the export for training, in the labelling record's format (record.load / record.rebuild read it):
                    the operative's samples of the record (the window's raw input, the person's groups as the walk's
                    choices, or 'keep'), and the unit level's order samples from the orders stream (unit_order: what of
                    the unit network's output the person's order gives one to one - the pace, the facing, the target, the
                    width in metres - and what it lacks: export.json)

The battle's state every second comes from the run's events (build/human/runs/<run>/events.jsonl: nn_sample - every
unit of both sides; nn_soldiers - the men's places every few seconds; result) when the run is found by the record's
batch; else from the record alone (my units only: orders.jsonl; the enemy at the windows' starts: samples.pt).

    python -m tools.label.digest build/label/<date>_live [--run build/human/runs/<run>] [--no-gif] [--export]
        (--export needs torch: the container; the digest and the frames run on the host too)
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from tools import config as project
from tools.label import view as VW

RUNS = project.ROOT / "build" / "human" / "runs"
COLORS = ("#e6194b", "#f58231", "#d4b800", "#3cb44b", "#20b4d8", "#d132e6")


# ------------------------------------------------------------------ reading

def _jsonl(f):
    if not f.exists():
        return []
    return [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]


def find_run(batch, runs=RUNS):
    """The run folder whose events carry the batch (None: not found)."""
    if not batch or not Path(runs).exists():
        return None
    for d in sorted(Path(runs).iterdir(), reverse=True):
        f = d / "events.jsonl"
        if not f.exists():
            continue
        with open(f, encoding="utf-8") as fh:
            for _, line in zip(range(40), fh):
                if f'"batch": "{batch}"' in line or f'"batch":"{batch}"' in line:
                    return d
    return None


class Record:
    """A live record and (when found) its run: session, matchup, windows (decisions.jsonl), orders {sec: {name: row}},
    chat, states {sec: {name: row}} (both sides; the run), soldiers {sec: {name: [x, z, ...] m}}, the run's result."""

    def __init__(self, folder, run=None):
        self.folder = Path(folder)
        self.session = json.loads((self.folder / "session.json").read_text(encoding="utf-8"))
        mf = self.folder / "matchup.json"
        self.matchup = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else {}
        rf = self.folder / "result.json"
        self.result = json.loads(rf.read_text(encoding="utf-8")) if rf.exists() else {}
        rows = _jsonl(self.folder / "decisions.jsonl")
        # the deployment (tools/label/live_deploy.py: the strategist's sample) apart from the operative's windows
        self.deploy = [w for w in rows if w.get("level") == "strategist"]
        self.windows = [w for w in rows if w.get("level") != "strategist"]
        self.orders = {int(r["sec"]): r["units"] for r in _jsonl(self.folder / "orders.jsonl")}
        chat = _jsonl(self.folder / "chat.jsonl")
        self.chat = [c for c in chat if not c.get("deploy")]          # (the deployment's: in its own section)
        self.names = list(self.session.get("names") or [u["n"] for u in self.matchup.get("units", [])])
        self.keys = list(self.session.get("keys") or [u.get("key") for u in self.matchup.get("units", [])])
        self.sides = list(self.session.get("sides") or [u.get("side") for u in self.matchup.get("units", [])])
        self.key_of = dict(zip(self.names, self.keys))
        self.side_of = dict(zip(self.names, self.sides))
        self.run = Path(run) if run else find_run(self.session.get("batch"))
        self.states, self.soldiers, self.run_result, self.manifest = {}, {}, {}, {}
        if self.run is not None:
            for e in _jsonl(self.run / "events.jsonl"):
                ev = e.get("event")
                if ev == "nn_sample" and e.get("batch") == self.session.get("batch", e.get("batch")):
                    self.states[int(round(float(e.get("t", 0)) / 1000.0))] = {u["n"]: u for u in e["units"]}
                elif ev == "nn_soldiers":
                    self.soldiers[int(round(float(e.get("t", 0)) / 1000.0))] = {
                        u["n"]: [v / 10.0 for v in u.get("xz_dm") or []] for u in e["units"]}
                elif ev == "result":
                    self.run_result = e
            mf = self.run / "manifest.json"
            if mf.exists():
                self.manifest = json.loads(mf.read_text(encoding="utf-8"))
        self.end = max(list(self.states) + list(self.orders) + [int(self.result.get("seconds") or 0)] or [0])

    def state(self, sec):
        """{name: row} at the second (the nearest at or before it; my units from the orders without the run)."""
        if self.states:
            ks = [k for k in self.states if k <= sec] or [min(self.states)]
            return self.states[max(ks)]
        ks = [k for k in self.orders if k <= sec] or ([min(self.orders)] if self.orders else [])
        out = {n: dict(r, side=1, n=n) for n, r in (self.orders[max(ks)] if ks else {}).items()}
        return out

    def men_at(self, sec):
        if not self.soldiers:
            return None
        k = min(self.soldiers, key=lambda s: abs(s - sec))
        return self.soldiers[k] if abs(k - sec) <= 3 else None


# ------------------------------------------------------------------ words

def unit_name(rec, n):
    return VW.names_ru().get(rec.key_of.get(n) or "", None) or n


def counted(rec, names):
    c = Counter(unit_name(rec, n) for n in names)
    return ", ".join(f"{k} ×{v}" if v > 1 else k for k, v in c.items())


def rel(p):
    try:
        return Path(p).resolve().relative_to(project.ROOT.resolve()).as_posix()
    except ValueError:
        return Path(p).as_posix()


def compass(b):
    """The side of the world a bearing looks to, as in the game's view from above (replay.py: x right, z down - the
    game's x, z with the height y up are right-handed: facing +x, +z is on the right): 'north' is up the picture, -z; a bearing b looks along (sin b, cos b)."""
    words = ("север", "северо-восток", "восток", "юго-восток", "юг", "юго-запад", "запад", "северо-запад")
    return words[int((((180.0 - b) % 360) + 22.5) // 45) % 8]


def xz(x, z):
    return f"({round(x / 10) * 10:.0f}, {round(z / 10) * 10:.0f})"


def where(rec, names, st):
    """(centre x, z, facing, width m, alive names) of units by their rows in st."""
    rows = [st[n] for n in names if n in st and (st[n].get("men") or 0) > 0]
    if not rows:
        return None
    cx = sum(float(r["x"]) for r in rows) / len(rows)
    cz = sum(float(r["z"]) for r in rows) / len(rows)
    sx = sum(math.sin(math.radians(float(r.get("b") or 0))) for r in rows)
    sz = sum(math.cos(math.radians(float(r.get("b") or 0))) for r in rows)
    b = math.degrees(math.atan2(sx, sz)) % 360
    a = math.radians(b)
    across = [(float(r["x"]) - cx) * math.cos(a) - (float(r["z"]) - cz) * math.sin(a) for r in rows]
    half = [float(r.get("ow") or 20) / 2 for r in rows]
    width = max(s + h for s, h in zip(across, half)) - min(s - h for s, h in zip(across, half))
    return cx, cz, b, width, [r["n"] for r in rows]


def groups_text(rec, groups, st, own=True):
    out = []
    for k, names in sorted(groups.items(), key=lambda kv: int(kv[0])):
        w = where(rec, names, st)
        head = f"{'группа' if own else 'враг'} {k}: {counted(rec, names)}"
        if w is None:
            out.append(head + " — никого не осталось")
            continue
        cx, cz, b, width, alive = w
        lost = len(names) - len(alive)
        out.append(f"{head} — центр {xz(cx, cz)}, фронт на {compass(b)} ({b:.0f}°), ширина {width:.0f} м" +
                   (f", выбыло {lost}" if lost else ""))
    return out


def order_of(rec, row, st):
    """(kind, words) of a unit's order now: attack a target, go to a point, hold."""
    t = row.get("t") or ""
    run = " бегом" if row.get("f") else ""
    if t:
        return ("attack", t, bool(row.get("f"))), f"атакуют {unit_name(rec, t)}{run}"
    ox, oz = row.get("ox"), row.get("oz")
    if ox is not None and math.hypot(float(ox) - float(row["x"]), float(oz) - float(row["z"])) > 8:
        return ("go", round(float(ox) / 20) * 20, round(float(oz) / 20) * 20, bool(row.get("f"))), \
            f"идут к {xz(float(ox), float(oz))}{run}"
    return ("hold",), "стоят"


def orders_text(rec, w, own):
    """Per group the orders given within the window (a unit's order that changed), compressed."""
    a, b = int(w["sec"]), int(w.get("until") or w["sec"] + 15)
    group_of = {n: k for k, names in own.items() for n in names}
    before = rec.orders.get(a - 1) or {}
    changed = defaultdict(list)
    seen = {}
    for s in range(a, b):
        rows = rec.orders.get(s)
        if not rows:
            continue
        for n, r in rows.items():
            if (r.get("men") or 0) <= 0:
                continue
            key = (r.get("ox"), r.get("oz"), r.get("t"), r.get("f"))
            prev = seen.get(n) or (before.get(n) and (before[n].get("ox"), before[n].get("oz"), before[n].get("t"),
                                                       before[n].get("f")))
            if prev is not None and key != prev:
                changed[n].append(s)
            seen[n] = key
    if not changed:
        return []
    last = rec.orders.get(max(s for s in range(a, b) if s in rec.orders), {}) if any(s in rec.orders for s in range(a, b)) else {}
    by = defaultdict(lambda: defaultdict(list))
    for n in changed:
        r = last.get(n)
        if not r:
            continue
        kind, words = order_of(rec, r, None)
        key = ("go", kind[-1]) if kind[0] == "go" else (kind, words)
        by[group_of.get(n, "—")][key].append((n, r, words))
    out = []
    for g in sorted(by, key=lambda k: (k == "—", str(k))):
        parts = []
        for key, items in by[g].items():
            names = [n for n, _, _ in items]
            if key[0] == "go":
                ox = [float(r["ox"]) for _, r, _ in items]
                oz = [float(r["oz"]) for _, r, _ in items]
                if max(ox) - min(ox) <= 80 and max(oz) - min(oz) <= 80:
                    parts.append(f"{counted(rec, names)} идут к {xz(sum(ox) / len(ox), sum(oz) / len(oz))}" +
                                 (" бегом" if key[1] else ""))
                else:
                    parts += [f"{unit_name(rec, n)} {w}" for n, _, w in items]
            else:
                parts.append(f"{counted(rec, names)} {key[1]}")
        out.append(f"{'без группы' if g == '—' else 'группа ' + str(g)}: " + "; ".join(parts))
    return out


def groups_line(rec, groups, st, own=True):
    """The groups as one line: number, centre, facing (when nothing in them changed)."""
    arrows = "↑↗→↘↓↙←↖"
    parts = []
    for k, names in sorted(groups.items(), key=lambda kv: int(kv[0])):
        w = where(rec, names, st)
        if w is None:
            parts.append(f"{k} —")
            continue
        lost = len(names) - len(w[4])
        parts.append(f"{k} {xz(w[0], w[1])}{arrows[int(((w[2] % 360) + 22.5) // 45) % 8]}" + (f" −{lost}" if lost else ""))
    return ("свои: " if own else "враг: ") + " · ".join(parts)


def fight_text(rec, w):
    a, b = int(w["sec"]), int(w.get("until") or w["sec"] + 15)
    if not rec.states:
        return []
    s0, s1 = rec.state(a), rec.state(b)
    out = []
    men = lambda st, side: sum(int(r.get("men") or 0) for r in st.values() if int(r.get("side") or 0) == side)  # noqa
    for side, word in ((1, "свои"), (2, "враг")):
        d = men(s1, side) - men(s0, side)
        if d:
            out.append(f"потери: {word} {d:+d} бойцов ({men(s0, side)} → {men(s1, side)})")
    new_melee = [n for n, r in s1.items() if r.get("m") and not (s0.get(n) or {}).get("m") and int(r.get("side")) == 1]
    if new_melee:
        out.append("в рукопашную: " + "; ".join(f"{unit_name(rec, n)}" + (f" ↔ {unit_name(rec, s1[n]['t'])}" if s1[n].get("t") else "")
                                                for n in new_melee))
    routs = [n for n, r in s1.items() if r.get("r") and not (s0.get(n) or {}).get("r")]
    if routs:
        out.append("бегут: " + counted(rec, routs))
    dead = [n for n, r in s1.items() if (r.get("men") or 0) <= 0 < ((s0.get(n) or {}).get("men") or 0)]
    if dead:
        out.append("уничтожены: " + counted(rec, dead))
    return out


SOURCE_RU = {"human": "решение человека: группы поменялись", "keep": "оставить как есть", "none": "групп нет",
             "deploy": "расстановка (образец стратега)"}


FORM_RU = {"line": "линия", "square": "квадрат"}


def deploy_md(rec):
    """The deployment's section: the person's groups as read for the strategist (centre, facing, form, the spreads),
    the enemy's groups, the chat of the deployment, what the strategist's walk refused, what is not one to one."""
    if not rec.deploy:
        return []
    d = rec.deploy[0]
    st = rec.state(0)
    L = ["## Расстановка · " + ("образец стратега" if not d.get("errors") else "образец стратега не принят проверкой")]
    for k, g in sorted((d.get("read") or {}).items(), key=lambda kv: int(kv[0])):
        L.append(f"- группа {k}{' (остаток)' if g.get('rest') else ''}: {counted(rec, g['members'])} — центр "
                 f"{xz(g['cx'], g['cz'])}, фронт на {compass(g['heading'])} ({g['heading']:.0f}°, сектор {g['turn']}), "
                 f"форма {FORM_RU.get(g['form'], g['form'])} (разброс по фронту {g['front_spread']:.0f} м, "
                 f"в глубину {g['depth_spread']:.0f} м)")
    enemy = {int(k): v for k, v in ((d.get("groups") or {}).get("enemy") or {}).items()}
    if enemy:
        L += ["- " + t for t in groups_text(rec, enemy, st, False)] if st else             [f"- враг {k}: {counted(rec, v)}" for k, v in sorted(enemy.items())]
    for c in d.get("chat") or []:
        L.append(f"- 💬 расстановка: «{c.get('text')}»")
    if d.get("errors"):
        L.append(f"- не принято проверкой: {'; '.join(d['errors'])[:300]}")
    if d.get("missing"):
        L.append(f"- не один в один: {len(d['missing'])} пункта (места внутри группы, клетка зоны, форма по разбросу, "
                 f"приказ начала боя) — `decisions.jsonl`, поле `missing`")
    L.append("")
    return L


def digest_md(rec):
    S = rec.session
    L = [f"# Бой {S.get('batch', '')}: лента разметки", ""]
    fac = rec.matchup.get("factions") or {}
    L.append(f"Карта: `{S.get('map') or rec.matchup.get('map')}`. Запись: `{rel(rec.folder)}`" +
             (f", прогон: `{rel(rec.run)}`" if rec.run else ", прогон не найден (враг — только на начале окон)") + ".")
    L.append("")
    for side, word in ((1, "Мои"), (2, "Враг")):
        names = [n for n in rec.names if rec.side_of.get(n) == side]
        cost = (rec.manifest.get("config", {}).get("army", {}).get("cost", {}) or {}).get("own" if side == 1 else "enemy")
        L.append(f"**{word}** ({fac.get('own' if side == 1 else 'enemy', '')}{', ' + str(cost) + ' золота' if cost else ''},"
                 f" {len(names)} отр.): {counted(rec, names)}.")
    L.append("")
    L += deploy_md(rec)
    prev = None
    last_errs = ""
    for i, w in enumerate(rec.windows):
        a, b = int(w["sec"]), int(w.get("until") or w["sec"] + 15)
        st = rec.state(a)
        g = w.get("groups") or {}
        own = {int(k): v for k, v in (g.get("own") or {}).items()}
        enemy = {int(k): v for k, v in (g.get("enemy") or {}).items()}
        L.append(f"## {a}–{b} с · {SOURCE_RU.get(w.get('source'), w.get('source'))}")
        errs = "; ".join(dict.fromkeys(w.get("errors") or []))[:300]
        if errs and errs != last_errs:
            L.append(f"- не принято проверкой: {errs}")
        elif errs:
            L.append("- не принято проверкой (то же)")
        last_errs = errs
        # (the groups in full when they are new or changed; else one line of centres and facings)
        same_own = prev is not None and prev[0] == own
        same_enemy = prev is not None and prev[1] == enemy
        if own:
            L += ["- " + groups_line(rec, own, st, True)] if same_own else ["- " + t for t in groups_text(rec, own, st, True)]
        if g.get("rest") and not same_own:
            L.append(f"- (отряды без группы собраны в группы {', '.join(map(str, g['rest']))})")
        if enemy:
            L += ["- " + groups_line(rec, enemy, st, False)] if same_enemy else ["- " + t for t in groups_text(rec, enemy, st, False)]
        if not own and not enemy:
            L.append("- групп нет")
        if prev is not None:
            ch = []
            po, pe = prev
            for k in sorted(set(po) | set(own)):
                if sorted(po.get(k, [])) != sorted(own.get(k, [])):
                    ch.append(f"группа {k}: " + ("новая" if k not in po else "распущена" if k not in own else
                                                 f"состав {counted(rec, po[k])} → {counted(rec, own[k])}"))
            for k in sorted(set(pe) | set(enemy)):
                if sorted(pe.get(k, [])) != sorted(enemy.get(k, [])):
                    ch.append(f"враг {k}: " + ("новая цель" if k not in pe else "снята" if k not in enemy else "состав поменялся"))
            if ch:
                L.append("- изменилось: " + "; ".join(ch))
        L += ["- приказы: " + t for t in orders_text(rec, w, own)]
        for c in (w.get("chat") or []) + [c for c in rec.chat if a <= float(c.get("sec") or -1) < b and not w.get("chat")]:
            L.append(f"- 💬 {c.get('sec')} с: «{c.get('text')}»")
        L += ["- " + t for t in fight_text(rec, w)]
        L.append("")
        prev = (own, enemy)
    r, rr = rec.result, rec.run_result
    L.append("## Итог")
    if rr:
        win = {1: "победа", 2: "поражение"}.get(int(rr.get("winner") or 0), "ничья")
        L.append(f"- {win}; время {round(float(rr.get('duration_model_ms') or 0) / 1000)} с; бойцов: свои {rr.get('side_1_men')},"
                 f" враг {rr.get('side_2_men')}; стоят отрядов: свои {rr.get('side_1_standing_units')}, враг "
                 f"{rr.get('side_2_standing_units')}")
    elif rec.states:
        st = rec.state(rec.end)
        for side, word in ((1, "свои"), (2, "враг")):
            rows = [x for x in st.values() if int(x.get("side") or 0) == side]
            L.append(f"- {word}: бойцов {sum(int(x.get('men') or 0) for x in rows)}, стоят "
                     f"{sum(1 for x in rows if (x.get('men') or 0) > 0 and not x.get('r'))} из {len(rows)}")
    L.append(f"- разметка: окон {r.get('windows', len(rec.windows))}, решений {r.get('human', 0)}, «оставить» "
             f"{r.get('keep', 0)}, без групп {r.get('none', 0)}, не принято {r.get('refused', 0)}; причина конца — "
             f"{r.get('why', '?')}")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder")
    ap.add_argument("--run", default=None)
    ap.add_argument("--out", default=None, help="default: <folder>/digest")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--export", action="store_true", help="samples.pt for training (torch)")
    a = ap.parse_args(argv)
    rec = Record(a.folder, a.run)
    out = Path(a.out) if a.out else rec.folder / "digest"
    out.mkdir(parents=True, exist_ok=True)
    (out / "digest.md").write_text(digest_md(rec), encoding="utf-8")
    from tools.label import replay as RP
    n = RP.frames(rec, out, gif=not a.no_gif)
    info = {"digest": rel(out / "digest.md"), "frames": n, "run": rel(rec.run) if rec.run else None}
    if a.export:
        from tools.label import digest_export as DX
        info["export"] = DX.export(rec, out)
    print(json.dumps(info, ensure_ascii=False))
    return info


if __name__ == "__main__":
    main()

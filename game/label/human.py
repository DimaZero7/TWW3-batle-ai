"""The person's decision -> the level's choices in the walk's own format, and the checks (torch).

The person never builds a parcel by hand: the page sends an edit (who is in which group, the target's seed, where the
zone should go, the form, the turn, the verb, the pace; at the tactician each unit's frame), this module turns it into
the choices of the level's walk (operative/net.choose, tactic/decide.choose - the act the network's sampling gives and
the training's labels take), and the walk itself runs with heads of zeros (operative/net.NullHeads; ZeroZoneHeads
here) and those choices. So the person's decision goes through exactly the network's road: the same bans (each choice
the walk would have masked is found by watching its _pick: the mask is NEG there), the same code after it (the target
grown from the seed, the step towards the goal, the nudge of the frames), and every parcel through tactic/parcel.check.
An error goes back to the page and nothing is written.

Operative (edit {"groups": [{"g", "members": [slots], "task": null | {"seed", "goal": [x, z], "form", "turn", "verb",
"pace"}}]} or {"keep": true}):
    a group with the same members and no task: keep; the same members and a task: move; other members: re-form (its
    units free, each pointing at its group in the edit); a new group: opened. A task left out for a group that needs
    one: the defaults (the old seed if still seen, else the seen enemy nearest the group; the goal where the group
    stands; square, sector 0, hold, any)
Tactician (edit {"rows": [{"g", "units": [{"slot", "keep", "w", "d", "x", "z"}]}]} or {"keep": true}):
    per unit keep its frame or change it to the size (w x d m, the steps 48 / 64 / 80) at the cell of the near window
    nearest the frame's centre (x, z)
"""
from contextlib import contextmanager

import torch

from net.operative import cfg as ocfg
from net.operative import groups as GR
from net.operative import net as NN
from net.operative import step as SP
from net.tactic import cfg as tcfg
from net.tactic import decide as DC
from net.tactic import parcel as PC
from net.tactic import raw as TR
from net.tactic import uzone as U
from net.unit import zmap as Z
from net.unit.net import NEG

FORMS = NN.FORMS
OP_HEADS = ("form", "target", "turn", "cell", "verb", "pace")
WHY = {
    "comp": "Отряд {u} не может войти в группу {g}: в ней уже 10 отрядов или её отряды стоят дальше 600 м от него. "
            "Как поправить: возьми группу поменьше или собери в неё только соседей.",
    "form": "Эта форма зоны сейчас закрыта.",
    "target": "Цель группы {g} — не видимый враг. Как поправить: щёлкни по врагу, которого видно на поле.",
    "turn": "Поворот зоны группы {g} закрыт: в таком повороте её цель не помещается в окно группы. "
            "Как поправить: поверни стрелку ближе к цели или выбери цель поменьше.",
    "cell": "Зону группы {g} сюда поставить нельзя: она стоит ближе 80 м к зоне группы с другой целью (у обеих есть "
            "цель, общего врага нет). Как поправить: отодвинь зону от соседней группы или дай им общую цель.",
    "verb": "Этот приказ группе {g} пока закрыт: сейчас открыт только «сдержать».",
    "pace": "Этот темп группе {g} пока закрыт: сейчас открыт только «любой».",
    "t_keep": "Отряд {u} (группа {g}) должен получить новую рамку: старой нет, она на выбывшей плитке или вне зоны "
              "группы. Как поправить: поставь ему рамку внутри зоны (или «Подсказка формулы»).",
    "t_cell": "Рамку отряда {u} (группа {g}) сюда поставить нельзя: она вылезает за зону группы, наезжает на соседа "
              "или не оставляет места отрядам после него. Как поправить: сдвинь рамку на свободное место или сделай "
              "её меньше.",
}

PARCEL = (          # tactic/parcel.check's messages (English, the level's code) -> what the person reads
    ("my slots:", "В группе слишком много или слишком мало отрядов: можно от 1 до 10."),
    ("my slots not alive", "В группе погибший или чужой отряд. Как поправить: убери его из группы."),
    ("target slots not the enemy", "Цель — не враг. Как поправить: щёлкни по вражескому отряду."),
    ("target not connected", "Цель разорвана: враги цели стоят врозь. Как поправить: щёлкни по одному врагу — "
                             "соседей код добавит сам."),
    ("target not compact", "Цель слишком широкая для такой группы. Как поправить: выбери другого врага или добавь "
                           "отрядов в группу (зона станет больше)."),
    ("a unit in two groups", "Один отряд оказался в двух группах. Как поправить: оставь его в одной."),
    ("zone outside the field", "Зона вылезает за край поля. Как поправить: потяни её внутрь."),
    ("the zone farther than half", "Зона слишком далеко от группы: за один раз её можно сдвинуть только на полокна. "
                                   "Как поправить: поставь её ближе, дальше — следующим решением."),
)


def human_parcel(m):
    return next((t for k, t in PARCEL if m.startswith(k)), m)


class Refused(ValueError):
    """The person's decision breaks a rule: the list of messages (nothing is written)."""

    def __init__(self, errors):
        super().__init__("; ".join(errors))
        self.errors = list(errors)


@contextmanager
def watch_picks(module):
    """Every module._pick(lg, given, greedy) call of the walk: (logits, given) in call order."""
    calls = []
    orig = module._pick

    def spy(lg, given, greedy):
        calls.append((lg.detach(), None if given is None else given.detach()))
        return orig(lg, given, greedy)
    module._pick = spy
    try:
        yield calls
    finally:
        module._pick = orig


def banned(lg, given):
    """[...] bool: the given choice was masked (NEG) in the logits."""
    return lg.gather(-1, given.long().clamp(min=0).unsqueeze(-1)).squeeze(-1) <= NEG / 2


# ---------------------------------------------------------------- operative


def _f(v):
    return round(float(v), 2)


def op_rows(ctx):
    """(slot list of the rows, own [K] bool, enemy [K] bool) of battle 0."""
    return ctx.slot[0].tolist(), ctx.own[0].tolist(), ctx.enemy[0].tolist()


def op_state(groups, ctx):
    """The groups in force as the page edits them (all kept)."""
    slots, own, _ = op_rows(ctx)
    out = []
    for g in range(groups.G):
        if not bool(groups.on[0, g]):
            continue
        mem = [s for s, o in zip(slots, own) if o and int(groups.member[0, s]) == g]
        out.append({"g": g, "members": mem, "task": None})
    return {"groups": out}


def op_result(out, ctx, axes):
    """The walk's result for the page: the groups after (members, target, zone), the goal, and as an edit."""
    new = out["new"]
    slots, own, _ = op_rows(ctx)
    act = out["act"]
    K = len(slots)
    groups, edit = [], []
    for g in range(new.G):
        if not bool(new.on[0, g]):
            continue
        mem = [s for s, o in zip(slots, own) if o and int(new.member[0, s]) == g]
        n0 = new.n0[0, g:g + 1]
        line = new.line[0, g:g + 1]
        front, depth = SP.size(n0, line)
        tgt = new.target[0, g].nonzero()[:, 0].tolist()
        task = bool(out["task"][0, g])
        groups.append({"g": g, "members": mem, "target": tgt, "seed": int(new.seed[0, g]), "cx": _f(new.cx[0, g]),
                       "cz": _f(new.cz[0, g]), "heading": _f(new.heading[0, g]), "line": bool(line[0]),
                       "front": _f(front[0]), "depth": _f(depth[0]), "n0": int(n0[0]), "task": task,
                       "goal": [_f(out["goal_x"][0, g]), _f(out["goal_z"][0, g])] if task else None,
                       "verb": DC.VERBS[int(new.verb[0, g])], "pace": DC.PACES[int(new.pace[0, g])]})
        t = None
        if task:
            ti = int(act["target"][0, g])
            t = {"seed": slots[ti] if ti < K else -1, "goal": [_f(out["goal_x"][0, g]), _f(out["goal_z"][0, g])],
                 "form": FORMS[int(act["form"][0, g])], "turn": int(act["turn"][0, g]),
                 "verb": DC.VERBS[int(act["verb"][0, g])], "pace": DC.PACES[int(act["pace"][0, g])]}
        edit.append({"g": g, "members": mem, "task": t})
    return {"groups": groups, "edit": {"groups": edit}}


def _centre(raw, slots_):
    I = TR.I
    x = [float(raw[0, s, I["x"]]) for s in slots_]
    z = [float(raw[0, s, I["z"]]) for s in slots_]
    return sum(x) / len(x), sum(z) / len(z)


def op_act(edit, ctx, axes):
    """(act {keep [1, G], comp [1, K], form, target, turn, cell, verb, pace [1, G]}, plan {free rows, task groups},
    filled edit): the walk's choices of the person's edit (module doc). Refused on a malformed edit."""
    c = ocfg()
    G = int(c["groups"])
    raw, old = ctx.raw, ctx.groups
    dev = raw.device
    slots, own, enemy = op_rows(ctx)
    K = len(slots)
    row_of = {s: j for j, s in enumerate(slots) if s >= 0}
    mine = [s for s, o in zip(slots, own) if o]
    seen = [s for s, e in zip(slots, enemy) if e]
    on = old.on[0].tolist()
    old_mem = {g: {s for s in mine if int(old.member[0, s]) == g} for g in range(G) if on[g]}
    act = {"keep": torch.full((1, G), GR.REFORM, dtype=torch.long, device=dev),
           "comp": torch.full((1, K), -1, dtype=torch.long, device=dev),
           **{h: torch.zeros((1, G), dtype=torch.long, device=dev) for h in OP_HEADS}}
    act["verb"][:] = DC.VERBS.index("hold")
    if edit.get("keep"):
        if not any(on):
            raise Refused(["Групп ещё нет — оставлять нечего. Как поправить: нажми «Поменять» и составь группы или прими подсказку формулы."])
        act["keep"] = torch.where(old.on, torch.full_like(act["keep"], GR.KEEP), act["keep"])
        return act, {"free": [], "tasks": []}, op_state(old, ctx)
    errs = []
    eg = {}
    where = {}
    for e in edit.get("groups", []):
        g = int(e.get("g", -1))
        mem = [int(s) for s in e.get("members", [])]
        if not 0 <= g < G:
            errs.append(f"группа {g}: номер 0-{G - 1}")
            continue
        if g in eg:
            errs.append(f"группа {g + 1} дважды")
            continue
        if not mem:
            continue
        for s in mem:
            if s not in mine:
                errs.append(f"Отряд {s} — не мой живой отряд: убери его из группы.")
            elif s in where:
                errs.append(f"Отряд {s} оказался в двух группах ({where[s] + 1} и {g + 1}): оставь его в одной.")
            where[s] = g
        if len(mem) > int(tcfg()["max_group"]):
            errs.append(f"В группе {g + 1} {len(mem)} отрядов, а можно не больше {tcfg()['max_group']}: раздели её на две.")
        eg[g] = {"members": set(mem), "task": e.get("task")}
    lost = [s for s in mine if s not in where]
    if lost:
        errs.append("Не у всех отрядов есть группа: " + ", ".join(f"отряд {s}" for s in lost) + ". Как поправить: выдели их рамкой и нажми цвет группы (каждый мой отряд — в одной группе).")
    if errs:
        raise Refused(errs)
    keep = [GR.REFORM] * G
    for g in range(G):
        if on[g] and g in eg and eg[g]["members"] == old_mem[g]:
            keep[g] = GR.KEEP if not eg[g]["task"] else GR.MOVE
    act["keep"] = torch.tensor([keep], dtype=torch.long, device=dev)
    rg = ctx.row_group[0].tolist()
    free = [j for j in range(K) if own[j] and (rg[j] < 0 or not on[rg[j]] or keep[rg[j]] == GR.REFORM)]
    for j in free:
        act["comp"][0, j] = where[slots[j]]
    tasks = [g for g in range(G) if g in eg and ((on[g] and keep[g] == GR.MOVE) or (keep[g] == GR.REFORM))]
    filled = []
    for g in range(G):
        if g not in eg:
            continue
        mem = sorted(eg[g]["members"])
        t = eg[g]["task"]
        if g in tasks:
            t = dict(t or {})
            gx, gz = _centre(raw, mem)
            if "seed" not in t or t["seed"] is None:
                s0 = int(old.seed[0, g]) if on[g] else -1
                if s0 not in seen:
                    s0 = min(seen, key=lambda s: (float(raw[0, s, TR.I["x"]]) - gx) ** 2 +
                             (float(raw[0, s, TR.I["z"]]) - gz) ** 2) if seen else -1
                t["seed"] = s0
            t.setdefault("goal", [gx, gz])
            if t["goal"] is None:
                t["goal"] = [gx, gz]
            t.setdefault("form", "square")
            t.setdefault("turn", 0)
            t.setdefault("verb", "hold")
            t.setdefault("pace", "any")
            seed = int(t["seed"])
            if seed >= 0 and seed not in seen:
                errs.append(f"Цель группы {g + 1} — не видимый враг. Как поправить: щёлкни по врагу, которого видно.")
                continue
            if t["form"] not in FORMS or t["verb"] not in DC.VERBS or t["pace"] not in DC.PACES:
                errs.append(f"группа {g + 1}: форма / глагол / темп не из словаря")
                continue
            act["form"][0, g] = FORMS.index(t["form"])
            act["target"][0, g] = row_of[seed] if seed >= 0 else K
            act["turn"][0, g] = int(t["turn"]) % int(c["sectors"])
            x = torch.tensor([float(t["goal"][0])], device=dev)
            z = torch.tensor([float(t["goal"][1])], device=dev)
            act["cell"][0, g] = SP.cell_of(axes, x, z)[0]
            act["verb"][0, g] = DC.VERBS.index(t["verb"])
            act["pace"][0, g] = DC.PACES.index(t["pace"])
        filled.append({"g": g, "members": mem, "task": t if g in tasks else None})
    if errs:
        raise Refused(errs)
    return act, {"free": free, "tasks": tasks}, {"groups": filled}


def op_walk(inp, ctx, act, plan=None):
    """(out, errors): the walk with the choices act and heads of zeros; errors: the choices it would have banned (plan:
    op_act's, to name them) and the members it did not take."""
    B, K = ctx.slot.shape
    G = int(ocfg()["groups"])
    dev = ctx.raw.device
    with watch_picks(NN) as calls:
        out = NN.choose(NN.null_enc(B, K, G, dev), inp, ctx, NN.NullHeads(B, K, G, dev), act=act)
    errs = []
    slots = ctx.slot[0].tolist()
    if plan is not None:
        names = [("keep", None)] + [("comp", j) for j in plan["free"]] + \
            [(h, g) for g in plan["tasks"] for h in OP_HEADS]
        if len(names) != len(calls):
            names = [("?", None)] * len(calls)
        for (h, who), (lg, given) in zip(names, calls):
            if given is None or not bool(banned(lg, given).any()):
                continue
            if h == "comp":
                errs.append(WHY["comp"].format(u=slots[who], g=int(act["comp"][0, who]) + 1))
            elif h in WHY:
                errs.append(WHY[h].format(g=who + 1))
            else:
                errs.append("выбор закрыт правилами уровня")
        want = act["comp"][0]
        got = out["act"]["comp"][0]
        for j in plan["free"]:
            if int(got[j]) != int(want[j]) and not errs:
                errs.append(f"отряд {slots[j]} не взят в группу {int(want[j]) + 1}")
    return out, errs


def op_parcels(out, raw, axes):
    """(Parcel of the groups with a new task, their rows) - what Tacticians.apply sends - and parcel.check's errors."""
    new = out["new"]
    G = new.G
    alive = raw[..., TR.I["alive"]] > 0
    t = out["task"][0]
    rows = t.nonzero()[:, 0]
    if rows.numel() == 0:
        return None, rows, []
    mem = new.members()[0] & alive[0][None]
    h = new.heading[0, rows] * TR.DEG
    p = PC.Parcel(torch.zeros_like(rows), mem[rows], new.target[0, rows], new.cx[0, rows], new.cz[0, rows],
                  torch.sin(h), torch.cos(h), new.line[0, rows], rows)
    f = axes.field
    field = {"cx": f["x"], "cz": f["z"], "b": f["b"], "front": f["front"], "depth": f["depth"]}
    errs = []
    for k, e in enumerate(PC.check(raw, p, field)):
        for m in e:
            errs.append(f"Группа {int(rows[k]) + 1}: {human_parcel(m)}")
    return p, rows, errs


# ---------------------------------------------------------------- tactician


class ZeroZoneHeads:
    """The tactician's heads of zeros (tactic/decide.ZoneHeads' calls): the walk run with given choices."""

    def __init__(self):
        self.S = len(tcfg()["size_steps_m"])
        self.C = int(tcfg()["cells"])

    def keep_logits(self, cards, prev):
        return torch.zeros(cards.shape[:2] + (2,), device=cards.device)

    def width(self, card):
        return torch.zeros((card.shape[0], self.S), device=card.device)

    depth = width

    def where(self, card, w, d, near):
        return torch.zeros((card.shape[0], self.C * self.C), device=card.device)


def zero_tenc(inp):
    R, K = inp["slot"].shape
    W = int(tcfg()["council"]["width"])
    return {"cards": torch.zeros((R, K, W), device=inp["slot"].device), "map": None}


def tac_walk(inp, ctx, act, rows_g=None):
    """(out, errors): tactic/decide.choose with the choices act and heads of zeros; errors: the units whose 'keep' was
    banned (they must change) or whose cell was closed; ParcelError (no room at all) as an error."""
    errs = []
    try:
        with watch_picks(DC) as calls:
            out = DC.choose(zero_tenc(inp), ctx, ZeroZoneHeads(), act=act)
    except PC.ParcelError as e:
        return None, [f"нет места в зоне: {e}"]
    my = ctx.my
    R, G = my.shape
    if calls:
        lg, given = calls[0]
        bad = banned(lg, given).view(R, G)
        for r, j in bad.nonzero().tolist():
            errs.append(WHY["t_keep"].format(u=int(my[r, j]), g=(rows_g[r] if rows_g else r) + 1))
        changing = ctx.able & ~out["kept"]
        js = [j for j in range(G) if bool(changing[:, j].any())]
        rest = calls[1:]
        if len(rest) == 3 * len(js):
            for k, j in enumerate(js):
                lg, given = rest[3 * k + 2]
                for r in banned(lg, given).nonzero()[:, 0].tolist():
                    if bool(changing[r, j]):
                        errs.append(WHY["t_cell"].format(u=int(my[r, j]), g=(rows_g[r] if rows_g else r) + 1))
        elif any(bool(banned(lg, gv).any()) for lg, gv in rest if gv is not None):
            errs.append("место рамки закрыто")
    return out, errs


def steps_idx(m):
    st = U.steps()
    return int((st - float(m)).abs().argmin())


def tac_act(edit, rows, groups_of_rows, ctx, tiles, prev):
    """act {keep, width, depth, cell} [r, Gm] of the person's edit; rows: the active rows' group numbers in order."""
    my = ctx.my
    R, G = my.shape
    dev = my.device
    C = int(tcfg()["cells"])
    act = {"keep": torch.full((R, G), DC.KEEP, dtype=torch.long, device=dev),
           **{k: torch.zeros((R, G), dtype=torch.long, device=dev) for k in ("width", "depth", "cell")}}
    by_g = {int(e["g"]): {int(u["slot"]): u for u in e.get("units", [])} for e in edit.get("rows", [])}
    cell = tiles.near_size() / C
    errs = []
    for r in range(R):
        g = groups_of_rows[r]
        units = by_g.get(g, {})
        for j in range(G):
            s = int(my[r, j])
            if s < 0 or not bool(ctx.able[r, j]):
                continue
            u = units.get(s)
            if edit.get("keep") or u is None or u.get("keep"):
                act["keep"][r, j] = DC.KEEP
                if prev is not None:
                    act["width"][r, j] = steps_idx(2 * float(prev["hw"][r, j]))
                    act["depth"][r, j] = steps_idx(2 * float(prev["hd"][r, j]))
                continue
            act["keep"][r, j] = DC.CHANGE
            act["width"][r, j] = steps_idx(u["w"])
            act["depth"][r, j] = steps_idx(u["d"])
            x = torch.tensor(float(u["x"]), device=dev)
            z = torch.tensor(float(u["z"]), device=dev)
            ls, lf = Z.local(x, z, tiles.cx[r], tiles.cz[r], tiles.heading[r])
            row, col = Z.cell_of(ls, lf, cell[r], C)
            act["cell"][r, j] = int(row.clamp(0, C - 1)) * C + int(col.clamp(0, C - 1))
    if errs:
        raise Refused(errs)
    return act


def tac_frames(frames, tiles, my, rows_g):
    """The frames as the page draws and edits them: {g: {slot: {x, z, b, w, d}}} (frames on)."""
    w = U.world(frames, tiles)
    out = {}
    R, G = my.shape
    for r in range(R):
        g = rows_g[r]
        out[g] = {}
        for j in range(G):
            s = int(my[r, j])
            if s < 0 or not bool(frames["on"][r, j]):
                continue
            out[g][s] = {"x": _f(w["x"][r, j]), "z": _f(w["z"][r, j]), "b": _f(w["b"][r, j]),
                         "w": _f(w["front"][r, j]), "d": _f(w["depth"][r, j])}
    return out


class HumanTactic:
    """The 'network' of operative/lower.Tacticians when the person is the tactician: its tick() builds the tactician's
    input (TacticInput, the network's road) and calls this with it; the choices are the person's (act, set before the
    tick), the walk the network's with heads of zeros. The last call is kept (the sample to write)."""

    def __init__(self):
        self.act = None
        self.rows_g = None
        self.last = None

    def __call__(self, inp, order, p, tiles, prev=None, greedy=False):
        ctx = DC.ctx_of(inp, p, tiles, prev)
        if self.act is None:
            raise RuntimeError("the tactician's tick without the person's choices")
        out, errs = tac_walk(inp, ctx, self.act, self.rows_g)
        if errs:
            raise Refused(errs)
        self.last = {"inp": inp, "order": order, "parcel": p, "tiles": tiles, "prev": prev, "act": self.act,
                     "out": out}
        self.act = None
        return out, None, ctx

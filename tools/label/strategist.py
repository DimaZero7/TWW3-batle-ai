"""The strategist's level in Разметка: the person deploys my army once, before the battle, through the strategist's own
road (tools/nn/strategist - only imported, never changed here; build/diag_strategist/plan.md "Выход").

The strategist sees my army, the enemy's composition (not its places), the map and both deployment zones; its output:
each of my units' group (comp), per group form, turn (16 sectors from "my zone -> the enemy's zone") and cell (where the
zone's centre goes, inside my deployment zone), then the operative's starting command (verb, pace). The code
(strategist/deploy.places) puts every unit on its place in its group's tiles; the groups go to the operative as its last
groups and to the tacticians as their parcels at second 0 (strategist/battle.play's order).

    Level(sess)        the zones from the matchup's deployment strips (pool.field: my strip faces the enemy's), the
                       strategist's input (strategist/inputs.build on the battle's raw state), the formula's hint
    pending()          what the page needs: my units, the enemy's composition, the zones, the edit, the hint
    act(edit)          the person's edit {"groups": [{"g", "members", "form", "turn", "goal": [x, z]}], "verb", "pace"}
                       -> the walk's choices (comp [1, K] by rows, form, turn, cell [1, G], verb, pace [1])
    walk(act)          (out, errors): strategist/net.choose with heads of zeros and those choices; its bans watched (the
                       cells outside my zone or on a placed group: NEG), no cell for a group, the deployment's errors
                       (strategist/deploy.errors on the state built with the places)
    apply(act, out)    the battle rebuilt with my units on their places, the start groups, the command; the sample
"""
import copy

import torch

from tools.label import human as HU
from tools.label import pool as PL
from tools.nn.operative import groups as GR
from tools.nn.operative import step as SP
from tools.nn.strategist import cfg as scfg
from tools.nn.strategist import deploy as DP
from tools.nn.strategist import inputs as SIN
from tools.nn.strategist import net as SNN
from tools.nn.strategist import teacher as STC
from tools.nn.strategist import zones as ZS
from tools.nn.tactic import decide as DC
from tools.nn.tactic import raw as TR

FORMS = ("square", "line")
WHY = {"comp": "отряд {u}: группа {g} полна (10 отрядов)",
       "form": "форма закрыта", "turn": "поворот закрыт",
       "cell": "место зоны группы {g}: зона не целиком в моей зоне расстановки или наезжает на уже поставленную группу",
       "verb": "глагол закрыт (config/nn/strategist.json verbs_open)",
       "pace": "темп закрыт (config/nn/strategist.json paces_open)"}


def zone_of(r, toward, device):
    """A deployment strip (pool.field rectangle) as the strategist's zone {x, z, b, front, depth} [1]: its forward
    towards the point toward (the other strip's centre), its sides the strip's."""
    x, z = (r["min_x"] + r["max_x"]) / 2, (r["min_z"] + r["max_z"]) / 2
    b = float(torch.rad2deg(torch.atan2(torch.tensor(toward[0] - x), torch.tensor(toward[1] - z)))) % 360.0
    across = abs(((b + 45.0) % 180.0) - 45.0) < 45.0          # facing along z: the front runs along x
    w, d = r["max_x"] - r["min_x"], r["max_z"] - r["min_z"]
    front, depth = (w, d) if across else (d, w)
    t = lambda v: torch.tensor([float(v)], device=device)       # noqa: E731
    return {"x": t(x), "z": t(z), "b": t(b), "front": t(front), "depth": t(depth)}


def act_of(edit, inp, axes, device):
    """The walk's choices of the person's edit (Level.act; the live deployment: tools/label/live_deploy.py). Refused on a
    malformed edit."""
    G = int(scfg()["groups"])
    dev = device
    slots = inp["slot"][0].tolist()
    own = inp["own"][0].tolist()
    mine = [s for s, o in zip(slots, own) if o]
    row_of = {s: j for j, s in enumerate(slots)}
    K = len(slots)
    act = {"comp": torch.full((1, K), -1, dtype=torch.long, device=dev),
           **{k: torch.zeros((1, G), dtype=torch.long, device=dev) for k in ("form", "turn", "cell")}}
    errs, where = [], {}
    for e in edit.get("groups", []):
        g = int(e.get("g", -1))
        if not 0 <= g < G:
            errs.append(f"группа {g}: номер 0-{G - 1}")
            continue
        for s in e.get("members", []):
            if s not in mine:
                errs.append(f"отряд {s}: не мой")
            elif s in where:
                errs.append(f"отряд {s} в двух группах")
            where[s] = g
        if not e.get("members"):
            continue
        act["form"][0, g] = FORMS.index(e.get("form", "square")) if e.get("form", "square") in FORMS else 0
        act["turn"][0, g] = int(e.get("turn", 0)) % int(scfg()["sectors"])
        gl = e.get("goal")
        if gl is None:
            errs.append(f"группа {g + 1}: не задано, куда зона")
            continue
        act["cell"][0, g] = SP.cell_of(axes, torch.tensor([float(gl[0])], device=dev),
                                       torch.tensor([float(gl[1])], device=dev))[0]
    lost = [s for s in mine if s not in where]
    if lost:
        errs.append(f"отряды без группы: {lost}")
    for s, g in where.items():
        act["comp"][0, row_of[s]] = g
    v, p = edit.get("verb", "hold"), edit.get("pace", "any")
    if v not in DC.VERBS or p not in DC.PACES:
        errs.append("глагол / темп не из словаря")
    if errs:
        raise HU.Refused(errs)
    act["verb"] = torch.tensor([DC.VERBS.index(v)], device=dev)
    act["pace"] = torch.tensor([DC.PACES.index(p)], device=dev)
    if edit.get("places"):
        # the raw (not a choice): every unit's place as the person left it - x, z, bearing, his width, the formation's
        # front / depth when the record knows them (nan: walk_of estimates them) - the check goes by these
        pl = torch.full((1, K, 6), float("nan"), device=dev)
        for s, v in edit["places"].items():
            s = int(s)
            if s in row_of:
                vals = (list(v) + [None] * 6)[:6]
                pl[0, row_of[s]] = torch.tensor([float("nan") if a is None else float(a) for a in vals], device=dev)
        act["places"] = pl
    return act


def exact_dims(places, inp):
    """(front, depth) [1, K] of the person's formations: the record's when it has them, else from the formation at the
    simulator's default frontage (inp front / depth) turned to his width - the same men, so the depth by the area
    (front x depth kept; a unit of one entity - a lord - keeps its own). An estimate: a rank (~2 m) off at most."""
    f0, d0 = inp["front"], inp["depth"]
    w = places[..., 3]
    single = f0 < 3.0
    front = torch.where(single | torch.isnan(w), f0, w)
    depth = torch.where(single | torch.isnan(w), d0, d0 * f0 / front.clamp(min=1.0))
    front = torch.where(torch.isnan(places[..., 4]), front, places[..., 4])
    depth = torch.where(torch.isnan(places[..., 5]), depth, places[..., 5])
    return front, depth


def exact_check(act, inp, ctx):
    """(errors, notes) of the person's own places (act places): every unit placed, its formation wholly inside my
    deployment zone (any turn - the army turned 45 deg is fine); two formations closer than unit_gap_m - a note (the
    game took his deployment as it is; our formations' model may differ from the game's)."""
    from tools.nn.sim import geometry
    pl = act["places"]
    own = inp["own"]
    slots = inp["slot"][0].tolist()
    have = ~torch.isnan(pl[..., 0])
    errs, notes = [], []
    lost = [slots[j] for j in range(own.shape[1]) if bool(own[0, j]) and not bool(have[0, j])]
    if lost:
        errs.append(f"отряды без места: {lost}")
    on = own & have
    x, z, b = (torch.nan_to_num(pl[..., k]) for k in range(3))
    front, depth = exact_dims(pl, inp)
    cs = DP.corners(x, z, b, front, depth)                                     # [1, K, 4, 2]
    s, f = ZS.local(ctx.mine, cs[..., 0].reshape(1, -1), cs[..., 1].reshape(1, -1))
    s, f = s.view(cs.shape[:-1]), f.view(cs.shape[:-1])
    out = ((s.abs() - ctx.mine["front"][:, None, None] / 2).amax(2).clamp(min=0) +
           (f.abs() - ctx.mine["depth"][:, None, None] / 2).amax(2).clamp(min=0))
    for j in range(own.shape[1]):
        if bool(on[0, j]) and float(out[0, j]) > 1e-3:
            errs.append(f"отряд {slots[j]}: строй вне зоны расстановки на {float(out[0, j]):.1f} м")
    gap = float(scfg()["unit_gap_m"])
    g = geometry.rect_gap(x, z, b, front, depth)[0]
    for i in range(own.shape[1]):
        for j in range(i + 1, own.shape[1]):
            if bool(on[0, i]) and bool(on[0, j]) and float(g[i, j]) < gap:
                notes.append(f"строи {slots[i]} и {slots[j]}: {float(g[i, j]):.1f} м (меньше {gap:g})")
    return errs, notes



def walk_of(act, inp, ctx):
    """(out, errors) of the walk with the choices act and heads of zeros (Level.walk without the deployment's own check).
    act with the person's places (act_of of an edit with 'places': the live deployment): the cell is the nearest one to
    his group's exact centre (a choice, not where he stood) - its bans and 'no cell' are not errors; the check goes by
    his exact places instead (exact_check: out['notes'] - his formations closer than unit_gap_m)."""
    B, K = inp["own"].shape
    G = int(scfg()["groups"])
    dev = inp['own'].device
    S, C = int(scfg()["sectors"]), int(scfg()["cells"])
    with HU.watch_picks(SNN) as calls:
        out = SNN.choose(SNN.null_enc(B, K, G, dev), inp, ctx, SNN.NullHeads(B, dev), act=act)
    names = {G: "comp", 2: "form", S: "turn", C * C: "cell", len(DC.VERBS): "verb", len(DC.PACES): "pace"}
    errs = []
    exact = act.get("places") is not None
    slots = inp["slot"][0].tolist()
    comp_rows = [j for j in range(K) if bool(inp["own"][0, j])]
    cells = [g for g in range(G) if bool(out["plan"]["on"][0, g])]
    ci = ce = 0
    for lg, given in calls:
        h = names.get(lg.shape[-1], "?")
        bad = given is not None and bool(HU.banned(lg, given).any())
        if h == "comp":
            j = comp_rows[ci] if ci < len(comp_rows) else -1
            ci += 1
            if bad:
                errs.append(WHY["comp"].format(u=slots[j] if j >= 0 else "?", g=int(act["comp"][0, j]) + 1))
        elif h == "cell":
            g = cells[ce] if ce < len(cells) else -1
            ce += 1
            if bad and exact:
                out.setdefault("snapped", []).append(g + 1)
            elif bad:
                errs.append(WHY["cell"].format(g=g + 1))
        elif bad:
            errs.append(WHY.get(h, "выбор закрыт правилами уровня"))
    if exact:
        e, out["notes"] = exact_check(act, inp, ctx)
        errs += e
    elif int(out["plan"]["no_cell"][0]) > 0:
        errs.append("одной из групп нет места в зоне расстановки")
    return out, errs



class Level:
    def __init__(self, sess):
        self.s = sess
        dev = sess.device
        fl = PL.field(sess.m.get("map", "flat"))
        c = lambda r: ((r["min_x"] + r["max_x"]) / 2, (r["min_z"] + r["max_z"]) / 2)        # noqa: E731
        self.mine = zone_of(fl["own"], c(fl["enemy"]), dev)
        self.enemy = zone_of(fl["enemy"], c(fl["own"]), dev)
        self.strips = {"own": fl["own"], "enemy": fl["enemy"]}
        b = sess.b
        self.raw0 = TR.Seer(sess.P).raw(b.st)
        self.axes = ZS.axes(self.mine, self.enemy, self.raw0, b.st.bounds)
        self.K = min(int(scfg()["rows"]), b.st.N)
        self.inp = SIN.build(self.raw0, b.card, b.side, self.axes, self.mine, self.enemy, b.command,
                             bounds=b.st.bounds, K=self.K)
        self.ctx = SNN.Ctx(self.axes, self.mine, self.enemy)
        self.hint_out = STC.play_formula(self.inp, self.ctx)

    # -------------------------------------------------------------- the page

    def result(self, out):
        """The groups and the places the walk gives, for the page and as an edit."""
        plan, act = out["plan"], out["act"]
        slots = self.inp["slot"][0].tolist()
        own = self.inp["own"][0].tolist()
        groups, edit = [], []
        for g in range(plan["on"].shape[1]):
            if not bool(plan["on"][0, g]):
                continue
            mem = [slots[j] for j in range(len(slots)) if own[j] and bool(plan["members"][0, g, j])]
            n = plan["n"][0, g:g + 1]
            front, depth = SP.size(n, plan["line"][0, g:g + 1])
            groups.append({"g": g, "members": mem, "cx": round(float(plan["cx"][0, g]), 1),
                           "cz": round(float(plan["cz"][0, g]), 1), "heading": round(float(plan["heading"][0, g]), 1),
                           "front": float(front[0]), "depth": float(depth[0]), "line": bool(plan["line"][0, g]),
                           "target": []})
            edit.append({"g": g, "members": mem, "form": FORMS[int(act["form"][0, g])], "turn": int(act["turn"][0, g]),
                         "goal": [round(float(plan["cx"][0, g]), 1), round(float(plan["cz"][0, g]), 1)]})
        pl = self.places(out)
        places = [{"slot": i, "x": round(float(pl["x"][0, i]), 1), "z": round(float(pl["z"][0, i]), 1),
                   "b": round(float(pl["b"][0, i]), 1), "width": round(float(pl["width"][0, i]), 1)}
                  for i in range(pl["on"].shape[1]) if bool(pl["on"][0, i])]
        return {"groups": groups, "places": places,
                "edit": {"groups": edit, "verb": DC.VERBS[int(act["verb"][0])], "pace": DC.PACES[int(act["pace"][0])]}}

    def pending(self):
        b = self.s.b
        r = self.result(self.hint_out)
        own = [i for i in range(b.st.N) if bool(b.own[0, i])]
        foes = [b.st.keys[0][i] for i in range(b.st.N) if float(b.st.u["side"][0, i]) == 2]
        mz = {k: float(v[0]) for k, v in self.mine.items()}
        ez = {k: float(v[0]) for k, v in self.enemy.items()}
        return {"level": "strategist", "sec": 0, "seen": own, "own": own, "enemy_keys": foes, "zones": [mz, ez],
                "edit": r["edit"], "hint": r["edit"], "edit_from": "hint", "can_keep": False,
                "sectors": int(scfg()["sectors"]), "verbs_open": scfg()["verbs_open"],
                "paces_open": scfg()["paces_open"], "base": float(self.ctx.base[0]),
                "command": {"verb": b.verb, "pace": b.pace}, "_hint_act": self.hint_out["act"]}

    # -------------------------------------------------------------- the decision

    def act(self, edit):
        return act_of(edit, self.inp, self.axes, self.s.device)

    def places(self, out):
        from tools.nn.strategist import battle as SB
        b = self.s.b
        mem = SB.slots_of(out["plan"], self.inp, b.st.N)
        return DP.places(mem, b.shooter, out["plan"]["cx"], out["plan"]["cz"], out["plan"]["heading"],
                         out["plan"]["line"], b.st.u["width"], DP.slot_of(out["plan"]))

    def matchup_after(self, out):
        """The matchup with my units on the strategist's places (and widths)."""
        pl = self.places(out)
        m = copy.deepcopy(self.s.m)
        units = m["sides"]["1"]["units"] if "1" in m["sides"] else m["sides"][1]["units"]
        for slot, who in enumerate(self.s.b.slots):
            if who is None or who[0] != 1 or not bool(pl["on"][0, slot]):
                continue
            u = units[who[1]]
            u.update(x=float(pl["x"][0, slot]), z=float(pl["z"][0, slot]), b=float(pl["b"][0, slot]),
                     width=float(pl["width"][0, slot]))
        return m, pl

    def walk(self, act):
        out, errs = walk_of(act, self.inp, self.ctx)
        if not errs:
            errs += self.deploy_errors(out)
        return out, errs

    def deploy_errors(self, out):
        """strategist/deploy.errors on the battle built with my units on the walk's places (the game: game.py's own)."""
        m2, pl = self.matchup_after(out)
        from tools.label import session as SE
        st = SE.Battle(m2, self.s.P, self.s.device).st
        spacing = self.s.P.sim["formation"]["spacing_m"]
        n = int(DP.errors(st, self.s.b.own, pl["group"], self.mine, spacing)[0])
        return [f"расстановка: {n} ошибок (отряд вне зоны расстановки или строи наезжают)"] if n else []

    def apply(self, act, out):
        """The battle rebuilt on the places; returns the sample's numbers."""
        s = self.s
        m2, pl = self.matchup_after(out)
        plan = out["plan"]
        s.m_deployed = m2
        s._build(m2)
        b = s.b
        verb, pace = DC.VERBS[int(plan["verb"][0])], DC.PACES[int(plan["pace"][0])]
        b.verb, b.pace = verb, pace
        b.command = DC.order_vec(1, verb, pace, s.device)
        b.destroy = torch.tensor([verb == "destroy"], device=s.device)
        N, G = b.st.N, int(scfg()["groups"])
        g0 = GR.Groups.empty(1, N, G, s.device)
        g0.member = pl["group"]
        g0.on = plan["on"].clone()
        g0.cx, g0.cz, g0.heading = plan["cx"].clone(), plan["cz"].clone(), plan["heading"].clone()
        g0.line = plan["line"].clone()
        g0.verb = torch.full((1, G), DC.VERBS.index("hold"), dtype=torch.long, device=s.device)
        g0.pace = torch.full((1, G), DC.PACES.index("any"), dtype=torch.long, device=s.device)
        g0.n0 = plan["n"].clone()
        s.start_groups = g0
        return {"level": "strategist", "sec": 0, "raw": self.raw0, "card": s.b.card, "side": s.b.side,
                "command": self.inp["command"], "mine_zone": self.mine, "enemy_zone": self.enemy,
                "bounds": s.b.st.bounds, "K": self.K, "act": act}


def rebuild(smp, device="cpu"):
    """(inp, ctx, act) of a strategist sample: inp exactly as strategist/inputs.build gave it."""
    d = lambda v: v.to(device) if torch.is_tensor(v) else v                       # noqa: E731
    mine = {k: d(v) for k, v in smp["mine_zone"].items()}
    enemy = {k: d(v) for k, v in smp["enemy_zone"].items()}
    raw = d(smp["raw"])
    axes = ZS.axes(mine, enemy, raw, smp["bounds"])
    inp = SIN.build(raw, d(smp["card"]), d(smp["side"]), axes, mine, enemy, d(smp["command"]), bounds=smp["bounds"],
                    K=smp["K"])
    return inp, SNN.Ctx(axes, mine, enemy), {k: d(v) for k, v in smp["act"].items()}

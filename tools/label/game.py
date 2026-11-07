"""Разметка в игре: the person is the strategist (once, at the deployment) and the operative (every 15 s) of a live
battle in the game; the tacticians (formula) and the unit network command the units through the bridge, by the chain of
tools/nn/companion/chain.py (docs/ru/training/label.md "Разметка в игре").

The game side: tools.build nn-arena --own-ai net --deploy strategist --label-game (src/apps/battle/label_pause.lua):
the player's orders, pause and speed controls off; the deployment waits for the plan (up to an hour); on every
operative second the bridge pauses the battle (speed 0) and waits on real-time timers for the answer to that move.
The host's tools/label/game.ps1 runs the server with the game folder mounted, waits for the page's request
(REQUEST: the task, the speed), builds the arena and launches the game, and writes OVER when the game has closed.

GameSession - session.Session's face for the page and the server (pending, preview, decide, stop, state, advance):
    a worker thread answers the bridge's files:
      the deployment document   the strategist's input (companion strategist_input, one to one with the simulator) ->
                                pending 'strategist' -> the person's decision through tools/label/strategist.Level's
                                walk and checks -> the plan file (companion deploy: the same places, the same points
                                sent) and the chain's starting groups
      a state document          the chain's second (Chain.decide): on the operative's seconds Chain.op_decide is this
                                session's: pending 'operative' (session.Session._op_pending: what the operative sees,
                                the formula's hint) -> the person's decision through human.op_act / op_walk / op_parcels
                                -> the chain goes on (tacticians, units) -> the orders file -> the bridge resumes
    the record: record.Recorder, the simulator's format (samples.pt, decisions.jsonl, replay.json, result.json) with
    "where": "game" in session.json and in every sample; chain.jsonl - every second's orders and the chain's fields.
"""
import json
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

from tools import config as project
from tools.label import human as HU
from tools.label import record as RC
from tools.label import session as SE
from tools.label import strategist as LS
from tools.label import view as VW
from tools.nn.companion import chain as CH
from tools.nn.companion import deploy as CD
from tools.nn.companion import exchange
from tools.nn.companion import strategist_input as SI
from tools.nn.operative import cfg as ocfg
from tools.nn.operative import teacher as OTC
from tools.nn.strategist import battle as SB
from tools.nn.strategist import cfg as scfg
from tools.nn.strategist import deploy as DP
from tools.nn.strategist import teacher as STC
from tools.nn.tactic import decide as DC
from tools.nn.tactic import raw as TR
from tools.nn.unit import view as V

ROOT = project.BUILD / "label" / "game"
REQUEST = ROOT / "request.json"       # the page's «Запустить игру»: {task, speed, at} - tools/label/game.ps1 reads it
OVER = ROOT / "over.json"             # game.ps1: the game has closed ({code, run, at})
TASKS = ("hold", "destroy")
SPEEDS = (1, 3, 10)
EMPTY = {"t": 0, "units": [], "groups": [], "frames": [], "men": None}
OVERLAY = "tww3_bai_nn_overlay.txt"   # src/apps/battle/label_overlay.lua FILE, CLICK, FORMAT
CLICK = "tww3_bai_nn_label_click.txt"
OVERLAY_FORMAT = "tww3_bai_nn_overlay 1"
TEXT = {"strategist": "Ты — стратег. Расстановка: реши в окне разметки (группы и зоны — на поле)",
        "operative": "Ты — оперативник. Пауза, t = {sec} с. Реши в окне разметки"}


def overlay_text(version, names, groups=(), text=None, button=False, camera=None, catalogue=None, ping=None):
    """The overlay file (label_overlay.lua's format): every group's members ('g', its number), its target ('t') and its
    zone; the line of text, the in-game button, the camera, the probe of the ground pings' types."""
    lines = [OVERLAY_FORMAT, f"version {int(version)}"]
    if text:
        lines.append("text " + " ".join(str(text).split()))
    marked = set()                     # one icon a unit: an enemy several groups target shows the first group's number
    for g in groups:
        k = int(g["g"]) + 1
        for kind, slots in (("g", g.get("members") or ()), ("t", g.get("target") or ())):
            for s in slots:
                if s not in marked:
                    marked.add(s)
                    lines.append(f"mark {names[s]} {kind} {k}")
        if g.get("front"):
            lines.append(f"zone {k} {g['cx']:.1f} {g['cz']:.1f} {g['heading']:.2f} {g['front']:.1f} {g['depth']:.1f}")
    if ping is not None:
        lines.append(f"ping {int(ping)}")
    if button:
        lines.append("button 1")
    if camera is not None:
        lines.append("camera " + " ".join(f"{v:.1f}" for v in camera))
    if catalogue is not None:
        lines.append("catalogue " + " ".join(f"{v:.1f}" if isinstance(v, float) else str(v) for v in catalogue))
    lines.append("end")
    return "\n".join(lines) + "\n"


def camera_of(raw, back_m=170.0, height_m=110.0):
    """(x, z, tx, tz, h): the camera behind my army's centre looking at it, along the way to the seen enemy's centre
    (none seen: +z)."""
    I = TR.I
    r = raw[0]
    alive = r[:, I["alive"]] > 0
    mine = alive & (r[:, I["side"]] == 1)
    foe = alive & (r[:, I["side"]] == 2) & (r[:, I["vis"]] > 0)
    if not bool(mine.any()):
        return None
    mx, mz = float(r[mine, I["x"]].mean()), float(r[mine, I["z"]].mean())
    dx, dz = 0.0, 1.0
    if bool(foe.any()):
        ex, ez = float(r[foe, I["x"]].mean()), float(r[foe, I["z"]].mean())
        d = ((ex - mx) ** 2 + (ez - mz) ** 2) ** 0.5
        if d > 1:
            dx, dz = (ex - mx) / d, (ez - mz) / d
    return (mx - dx * back_m, mz - dz * back_m, mx + dx * 40.0, mz + dz * 40.0, height_m)


def units_info(doc, params):
    """view.units_info of a bridge document: per slot (the document's order) the page's entry of the unit."""
    out = []
    for i, u in enumerate(doc["units"]):
        key = u.get("key") or ""
        e = VW.unit_entry(key)
        s = params.static(key) if key in params.units else {}
        e.update({"slot": i, "side": int(u["side"]), "men0": int(s.get("men0", u.get("men") or 0)),
                  "h": round(float(s.get("sp_h") or 1.5), 2), "v": round(float(s.get("sp_v") or 1.5), 2),
                  "ammo0": float(s.get("ammo0", 0.0))})
        out.append(e)
    return out


def unit_rows(raw_u, doc, places=None):
    """The snapshot's units (view.snapshot's row) from the unit level's raw state of a document; places {slot: (x, z,
    b)} overrides a unit's place (the deployment: its real place, the raw state's is the strategist's laid-out one)."""
    I = V.I
    r = raw_u[0]
    out = []
    for s, u in enumerate(doc["units"]):
        if float(r[s, I["side"]]) <= 0:
            continue
        x, z, b = (places or {}).get(s, (float(r[s, I["x"]]), float(r[s, I["z"]]), float(r[s, I["b"]])))
        ammo = float(r[s, I["ammo"]]) if float(r[s, I["range"]]) > 0 else None
        out.append([s, int(r[s, I["side"]]), round(x, 1), round(z, 1), round(b, 1), round(float(r[s, I["front"]]), 1),
                    round(float(r[s, I["depth"]]), 1), int(round(float(u.get("men") or 0))),
                    round(float(r[s, I["hp"]]), 3), int(float(r[s, I["r"]]) > 0), int(float(r[s, I["alive"]]) > 0),
                    int(float(r[s, I["vis"]]) > 0), None if ammo is None else round(ammo, 2),
                    int(float(r[s, I["m"]]) > 0), int(float(r[s, I["mv"]]) > 0), int(float(r[s, I["fire"]]) > 0)])
    return out


class GameStrategist(LS.Level):
    """tools/label/strategist.Level on the game's deployment document: the input by the companion's StrategistSeer (as
    the companion's strategist, one to one with the simulator), the places and the deployment's errors by the companion's
    deploy.Strategist (its places_of: the same code the plan of the network or the formula goes through)."""

    def __init__(self, doc, cd, params, device="cpu"):
        self.s = SimpleNamespace(device=device, P=params)
        self.doc, self.cd = doc, cd
        command = DC.order_vec(1, cd.task, "any", device)
        self.inp, self.ctx, self.laid = cd.seer.input(doc, command)
        self.raw0 = cd.seer.raw
        self.mine, self.enemy, self.axes = self.ctx.mine, self.ctx.enemy, self.ctx.axes
        self.K = self.inp["slot"].shape[1]
        self.hint_out = STC.play_formula(self.inp, self.ctx)
        self.bounds = params.map_half

    def places(self, out):
        raw = self.raw0
        own = raw[..., TR.I["side"]] == 1
        shooter = (raw[..., TR.I["range"]] > 0) & own
        width0 = torch.tensor([[float(u.get("ow") or 0.0) for u in self.laid["units"]]], device=raw.device)
        plan = out["plan"]
        mem = SB.slots_of(plan, self.inp, raw.shape[1])
        return DP.places(mem, shooter, plan["cx"], plan["cz"], plan["heading"], plan["line"], width0, DP.slot_of(plan))

    def deploy_errors(self, out):
        sent = getattr(self.cd, "sent", None)
        _, n, _ = self.cd.places_of(self.doc, self.laid, self.inp, out)
        self.cd.sent = sent
        return [f"расстановка: {n} ошибок (отряд вне зоны расстановки или строи наезжают)"] if n else []

    def pending(self):
        r = self.result(self.hint_out)
        units = self.doc["units"]
        own = [i for i, u in enumerate(units) if int(u["side"]) == 1]
        foes = [u.get("key") or "" for u in units if int(u["side"]) == 2]
        mz = {k: float(v[0]) for k, v in self.mine.items()}
        ez = {k: float(v[0]) for k, v in self.enemy.items()}
        return {"level": "strategist", "sec": 0, "seen": own, "own": own, "enemy_keys": foes, "zones": [mz, ez],
                "edit": r["edit"], "hint": r["edit"], "edit_from": "hint", "can_keep": False,
                "sectors": int(scfg()["sectors"]), "verbs_open": scfg()["verbs_open"],
                "paces_open": scfg()["paces_open"], "base": float(self.ctx.base[0]),
                "command": {"verb": self.cd.task, "pace": "any"}, "_hint_act": self.hint_out["act"]}

    def plan(self, out, think_ms=None):
        """(the plan file's text, the chain's starting plan): companion deploy.Strategist.decide's tail on out."""
        cd, doc = self.cd, self.doc
        places, errors, groups = cd.places_of(doc, self.laid, self.inp, out)
        cd.plans[doc["batch"]] = cd.chain_plan
        rows = [(n, *cd.sent[n], p[2], p[3], p[0], p[1]) for n, p in places.items()]
        return CD.plan_text(doc["batch"], rows, think_ms), {"errors": errors, "groups": groups,
                                                            "places": {n: [round(v, 2) for v in p] for n, p in
                                                                       places.items()}}

    def sample(self, act):
        return {"level": "strategist", "sec": 0, "raw": self.raw0, "card": self.cd.seer.tactic.card,
                "side": torch.ones(1, dtype=torch.long, device=self.raw0.device), "command": self.inp["command"],
                "mine_zone": self.mine, "enemy_zone": self.enemy, "bounds": self.bounds, "K": self.K, "act": act}


class GameSession(SE.Session):
    """The labelling battle in the game (module doc); session.Session's methods for the operative's pause and checks."""

    def __init__(self, game, nets, task="hold", speed=3, device="cpu", name="игра", out=True, out_root=None,
                 request=True, poll_s=0.005, map_root=None, allow_flat=True, camera=False, catalogue=False):
        if task not in TASKS:
            raise HU.Refused([f"задача {task!r}: одна из {TASKS}"])
        self.game, self.device, self.task, self.speed = Path(game), device, task, int(speed)
        from tools.nn.sim.params import load
        self.P = load()
        self.level, self.men, self.poll_s = "operative", False, poll_s
        self.op_every = int(ocfg()["decide_s"])
        self.period = self.op_every
        self.G = int(ocfg()["groups"])
        kw = {} if map_root is None else {"map_root": map_root}
        self.cd = CD.Strategist("formula", task, device=device, params=self.P, allow_flat=allow_flat, **kw)
        self.chain = CH.Chain(nets["unit"], plans=self.cd.plans, task=task, params=self.P, device=device,
                              allow_flat=allow_flat, **kw)
        self.chain.op_decide = self._op_decide
        self.m = {"name": name, "map": "flat", "where": "game", "task": task, "speed": self.speed}
        self.b = SimpleNamespace(verb=task, pace="any", destroy=torch.tensor([task == "destroy"], device=device),
                                 limit=None, st=SimpleNamespace(bounds=self.P.map_half))
        self.sec, self.over, self.result, self.pending, self.why = 0, False, None, None, None
        self.n_decisions, self.parcels, self.parcel_errors = 0, 0, []
        self.raw, self.op, self.strat, self.batch = None, None, None, None
        self.units_info, self.last_snap, self.replay, self.new_snaps = [], dict(EMPTY), [], []
        self.phase = "waiting for the game"
        self.error = None
        self.info = {"keys": []}
        self.cv = threading.Condition()
        self._answer = None
        self._stop = False
        self.last_doc = None
        self.folder, self.rec = None, None
        self.camera, self.catalogue = bool(camera), bool(catalogue)   # (the scripted check: the camera on my army at
        #                                                               a pause; the ground pings' types laid out once)
        self.ov_version, self.ov_groups, self.names = 0, None, []
        self.via = "page"
        if out:
            self.folder = RC.new_folder(name, out_root)
        self.started_at = time.time()
        if request:
            ROOT.mkdir(parents=True, exist_ok=True)
            if OVER.exists():
                OVER.unlink()
            REQUEST.write_text(json.dumps({"task": task, "speed": self.speed, "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                           "folder": self._rel(self.folder)}, ensure_ascii=False), encoding="utf-8")
        self.worker = threading.Thread(target=self._run, name="label-game", daemon=True)
        self.worker.start()

    # ------------------------------------------------------------ what Session's methods read
    @property
    def groups(self):
        return self.chain.groups

    @property
    def axes(self):
        return getattr(self.chain, "axes", None)

    @property
    def tac(self):
        return self.chain.tac

    # ------------------------------------------------------------ the worker
    def _run(self):
        st_f, dep_f = self.game / exchange.STATE, self.game / CD.DOC
        old = exchange.read_state(st_f)
        last = (old["batch"], old["move"], bool(old.get("done"))) if old else None
        old = exchange.read_state(dep_f)
        last_dep = old["batch"] if old else None
        try:
            while not self._stop:
                new = False
                d = exchange.read_state(dep_f)
                if d is not None and d.get("deploy") and d["batch"] != last_dep:
                    last_dep, new = d["batch"], True
                    self._deploy(d)
                s = exchange.read_state(st_f)
                # (the bridge's last document, done, keeps the last move's number: told apart by its flag)
                if s is not None and (s["batch"], s["move"], bool(s.get("done"))) != last and not self._stop:
                    last, new = (s["batch"], s["move"], bool(s.get("done"))), True
                    if s.get("done"):
                        self._end(s, "the battle's end")
                        return
                    self._second(s)
                if OVER.exists() and OVER.stat().st_mtime >= self.started_at and not self._stop:
                    self._end(self.last_doc, "the game closed")
                    return
                if not new:
                    time.sleep(self.poll_s)
        except Exception as e:                       # shown on the page; the game resumes with no answer (nn_miss)
            traceback.print_exc()
            self.error = f"{type(e).__name__}: {e}"
            self._end(self.last_doc, f"error: {self.error}")

    def _deploy(self, doc):
        self.batch = doc["batch"]
        self.phase = "deployment"
        self.units_info = units_info(doc, self.P)
        self.info = {"keys": [u.get("key") or "" for u in doc["units"]]}
        self.strat = GameStrategist(doc, self.cd, self.P, self.device)
        real = {i: (float(u["x"]), float(u["z"]), float(u["b"])) for i, u in enumerate(doc["units"])
                if int(u["side"]) == 1 and u.get("x") is not None}
        raw_u = self.strat.raw0[..., :TR.BASE]
        units = [u for u in unit_rows(raw_u, self.strat.laid, real) if u[1] == 1]
        self._snap({"t": 0, "units": units, "groups": [], "frames": [], "men": None})
        self._start_record(doc)
        self.names = [u["n"] for u in doc["units"]]
        import math
        z = doc["zones"]["own"]
        a = math.radians(float(z["b"]))
        fx, fz = math.sin(a), math.cos(a)
        rx, rz = fz, -fx
        cam = (z["x"] - fx * 260, z["z"] - fz * 260, z["x"] + fx * 40, z["z"] + fz * 40, 200.0) if self.camera else None
        cat = None
        if self.catalogue:
            x0 = z["x"] + fx * (z["depth"] / 2 - 25) - rx * 105
            z0 = z["z"] + fz * (z["depth"] / 2 - 25) - rz * 105
            cat = (round(x0, 1), round(z0, 1), round(rx * 14, 2), round(rz * 14, 2), 0, 15)
        with self.cv:
            self.pending = self.strat.pending()
            hint = self.strat.result(self.strat.hint_out)
            self._overlay(hint["groups"], TEXT["strategist"], camera=cam, catalogue=cat)
            while self.pending is not None and not self._stop:
                self.cv.wait(0.2)
                self._click()

    def _start_record(self, doc):
        if self.folder is None or self.rec is not None:
            return
        info = {"where": "game", "level": "operative + strategist", "period_s": self.op_every,
                "period_config_s": int(ocfg()["decide_s"]), "operative_every_s": self.op_every,
                "tactic_every_s": int(self.chain.t_every), "upper": "the strategist: the person (deployment)",
                "lower": "tactic formula + unit network (companion chain)", "unit": "frozen unit network (sampling, "
                "as the simulator's chain)", "nets": {"unit": "config/nn/tactic.json train.unit_ckpt"},
                "keys": [u.get("key") or "" for u in doc["units"]], "names": [u["n"] for u in doc["units"]],
                "sides": [int(u["side"]) for u in doc["units"]], "batch": doc["batch"], "map": doc.get("map"),
                "task": self.task, "speed": self.speed, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                "fighters_lab": False}
        self.info = dict(info)
        self.m.update(map=doc.get("map"), zones=doc.get("zones"), factions=doc.get("factions"),
                      units=[{k: u.get(k) for k in ("n", "side", "key", "men")} for u in doc["units"]])
        self.rec = RC.Recorder(self.folder, self.m, info)

    def _second(self, doc):
        self.last_doc = doc
        if doc["batch"] != self.batch or self.rec is None:
            self.batch = doc["batch"]
            self._start_record(doc)
        if self.chain.batch != doc["batch"]:
            self.units_info = units_info(doc, self.P)
            self.info["keys"] = [u.get("key") or "" for u in doc["units"]]
            self.phase = "battle"
        self.sec = int(round(float(doc.get("t", 0)) / 1000.0))
        t0 = time.perf_counter()
        if not self.names:
            self.names = [u["n"] for u in doc["units"]]
        orders, think_ms, uses = self.chain.decide(doc)
        gs, _ = VW.groups_frames(self.chain.groups, self.chain.tac)
        sig = json.dumps([[g["g"], g["members"], g["target"], g["cx"], g["cz"], g["heading"]] for g in gs])
        if sig != self.ov_groups:
            self.ov_groups = sig
            self._overlay(gs)
        self.b.verb, self.b.pace = self.chain.verb, self.chain.pace_cmd
        self.b.destroy = self.chain.destroy
        attempts = exchange.write_atomic(self.game / exchange.ORDERS,
                                         exchange.orders_text(doc["batch"], doc["move"], orders, think_ms, uses))
        groups, frames = VW.groups_frames(self.chain.groups, self.chain.tac)
        self._snap({"t": self.sec, "units": unit_rows(self.chain.raw_u, doc), "groups": groups, "frames": frames,
                    "men": None})
        if self.folder is not None:
            row = {"move": doc["move"], "t": doc.get("t"), "think_ms": round(think_ms, 1),
                   "turn_ms": round((time.perf_counter() - t0) * 1000, 1), "write_attempts": attempts,
                   "orders": orders, "chain": self.chain.log_extra()}
            with open(self.folder / "chain.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _snap(self, s):
        with self.cv:
            self.last_snap = s
            self.replay.append(s)
            self.new_snaps.append(s)
            self.cv.notify_all()

    def _op_decide(self, inp, ctx, raw):
        """Chain.op_decide: the person's decision on the operative's second (waits for decide())."""
        self.op, self.raw = (inp, ctx), raw
        self.b.verb, self.b.pace, self.b.destroy = self.chain.verb, self.chain.pace_cmd, self.chain.destroy
        if self._stop or not bool(ctx.own.any(1)[0]):
            return OTC.play_formula(inp, ctx, self.b.destroy)
        with self.cv:
            self._answer = None
            self.pending = self._op_pending(inp, ctx)
            gs, _ = VW.groups_frames(self.chain.groups, self.chain.tac)
            self._overlay(gs, TEXT["operative"].format(sec=self.sec), button=self.pending["can_keep"],
                          camera=camera_of(raw) if self.camera else None)
            self.cv.notify_all()
            while self._answer is None and not self._stop:
                self.cv.wait(0.2)
                self._click()
            out, self._answer = self._answer, None
        return out if out is not None else OTC.play_formula(inp, ctx, self.b.destroy)

    def _overlay(self, groups, text=None, button=False, camera=None, catalogue=None):
        """The overlay file for the game (label_overlay.lua): a new version each call."""
        if not self.names:
            return
        with self.cv:
            self.ov_version += 1
            exchange.write_atomic(self.game / OVERLAY, overlay_text(self.ov_version, self.names, groups, text, button,
                                                                    camera, catalogue))

    def _click(self):
        """The in-game button (label_overlay.lua's dialogue box: its tick) -> «Оставить как есть», or the formula's hint
        where nothing can be kept; the click file removed."""
        f = self.game / CLICK
        if not f.exists():
            return
        try:
            f.unlink()
        except OSError:
            return
        p = self.pending
        if p is None:
            return
        self.via = "game button"
        try:
            self.decide({"keep": True} if p.get("can_keep") else p["hint"])
        finally:
            self.via = "page"

    def preview(self, edit):
        """session.Session.preview; the groups it gives drawn in the game too (the person's edit seen on the field)."""
        out = super().preview(edit)
        p = self.pending
        if out.get("ok") and out.get("groups") is not None and p is not None:
            text = TEXT["strategist"] if p["level"] == "strategist" else TEXT["operative"].format(sec=p["sec"])
            self._overlay(out["groups"], text, button=bool(p.get("can_keep")))
        return out

    # ------------------------------------------------------------ the person's calls (the server's thread)
    def advance(self, seconds=None, wait_s=0.6):
        """The snapshots since the last call (waits up to wait_s for one when there is none and nothing is pending)."""
        with self.cv:
            if not self.new_snaps and self.pending is None and not self.over:
                self.cv.wait(wait_s)
            out, self.new_snaps = self.new_snaps, []
        return out

    def decide(self, edit):
        with self.cv:
            pend = self.pending
            if pend is None:
                return {"ok": False, "errors": ["решения сейчас не ждут"]}
            try:
                if pend["level"] == "strategist":
                    act = self.strat.act(edit)
                    out, errs = self.strat.walk(act)
                    filled = edit
                else:
                    act, out, filled, errs = self._op_try(edit)
            except HU.Refused as e:
                return {"ok": False, "errors": e.errors}
            if errs:
                return {"ok": False, "errors": errs}
            source = self._source(edit)
            if pend["level"] == "strategist":
                res = self.strat.result(out)
                text_plan, made = self.strat.plan(out)
                sample = dict(self.strat.sample(act), formula_act=pend["_hint_act"], source=source, where="game")
                text = {"sec": 0, "level": "strategist", "source": source, "edit": filled, "groups": res["groups"],
                        "places": res["places"], "where": "game", "plan_errors": made["errors"], "via": self.via}
                self._overlay(res["groups"])
                exchange.write_atomic(self.game / CD.PLAN, text_plan)
                if self.folder is not None:
                    (self.folder / "plan.txt").write_text(text_plan, encoding="utf-8")
                self.phase = "deployed"
            else:
                inp, ctx = self.op
                sample = {"level": "operative", "sec": self.sec, "raw": self.raw,
                          "groups": RC.groups_dict(self.groups.clone()), "card": self.chain.card,
                          "side": self.chain.side, "command": self.chain.order, "axes": RC.axes_dict(self.axes),
                          "bounds": self.chain.bounds, "K": self.chain.K, "act": act, "formula_act": pend["_hint_act"],
                          "source": source, "where": "game"}
                res = HU.op_result(out, ctx, self.axes)
                text = {"sec": self.sec, "level": "operative", "source": source, "edit": filled, "groups": res["groups"],
                        "where": "game", "via": self.via}
                self._overlay(res["groups"])
                self._answer = out
            if self.rec is not None:
                self.rec.add(sample, text)
            self.n_decisions += 1
            self.pending = None
            self.cv.notify_all()
        return {"ok": True, "source": source, "decisions": self.n_decisions}

    def stop(self):
        with self.cv:
            self._stop = True
            self.pending = None
            self.cv.notify_all()
        if not self.over:
            self._end(self.last_doc, "stopped by the person")
        return self.result

    def _end(self, doc, why):
        with self.cv:
            if self.over:
                return
            self._stop = True
            self.pending = None
            self.over, self.why, self.phase = True, why, "over"
            try:
                self._overlay([])
            except OSError:
                pass
            left, men = {}, {}
            units = (doc or {}).get("units") or []
            for s, name in ((1, "mine"), (2, "enemy")):
                rows = [u for u in units if int(u["side"]) == s]
                stand = [u for u in rows if (u.get("men") or 0) > 0 and not u.get("r") and not u.get("s")]
                left[name] = {"units": len(rows), "standing": len(stand)}
                men0 = sum(int((self.P.static(u["key"]) if u.get("key") in self.P.units else {}).get("men0", 0))
                           for u in rows)
                men[name] = {"men0": men0, "men": int(sum(u.get("men") or 0 for u in rows))}
            winner = 2 if units and left["mine"]["standing"] == 0 else (1 if units and left["enemy"]["standing"] == 0
                                                                          else 0)
            st = getattr(self.chain, "stats", {}) or {}
            self.result = {"why": why, "where": "game", "seconds": self.sec, "winner": winner, "left": left, "men": men,
                           "decisions": self.n_decisions, "parcels": st.get("parcels", 0),
                           "parcel_errors": st.get("parcel_errors", 0), "level": self.level, "period_s": self.period,
                           "error": self.error, "batch": self.batch}
            if OVER.exists():
                try:
                    self.result["game"] = json.loads(OVER.read_text(encoding="utf-8-sig"))
                except ValueError:
                    pass
            if self.rec is not None:
                self.rec.finish(self.result, {"keys": self.info.get("keys"), "names": self.info.get("names"),
                                              "snaps": self.replay})
            self.cv.notify_all()

    def state(self):
        with self.cv:
            out = super().state()
        out.update(game=True, phase=self.phase, error=self.error, task=self.task, speed=self.speed)
        return out

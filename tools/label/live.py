"""Разметка вживую (build/diag_chain/live_label.md, docs/ru/training/label.md "Разметка вживую"): a person plays a battle in
the game as usual - no pause, his own orders - and his groups are the operative's labels. The game side:
tools.build human --live-label (src/apps/battle/live_label.lua) writes every second of battle the state document with
groups = {own = {{k, {names}}} - the game's own groups (Ctrl+G), enemy = {{k, {names}}} - the groups made on the enemy
cards}. This recorder cuts the battle into windows of WINDOW_S (15 s, the operative's period):

    a window's start   the operative's input then: the raw state (tactic_input.TacticSeer, one to one with the simulator),
                       the person's groups then as the operative's last groups (own group k -> group k-1, its target the
                       enemy group k, its zone where its units stand), the field's axes
    a window's end     the groups changed within it -> a decision (source 'human'): the walk's choices of the groups now
                       (human.op_act: members, target seed - the first unit of the enemy group, the zone where the units
                       stand), checked by the walk (human.op_walk); the same groups -> a skip ('keep'); no groups at all
                       -> 'none' (written as text only)
    every second       the player's orders (orders.jsonl: each own unit's place, ordered point, target, running, moving,
                       melee, width - the bridge's row)

The record: record.Recorder's folder (samples.pt, decisions.jsonl, result.json) with "mode": "live", "where": "game";
a sample keeps the window's raw input (record.rebuild gives the operative's input) and the person's groups as they were
("human_groups"), the walk's act when it took them (None and the errors when not: e.g. a unit in no group).

    python -m tools.label.live --game /game [--out build/label] [--name live]    (the container: torch)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

from tools.label import game as GM
from tools.label import human as HU
from tools.label import record as RC
from tools.nn.companion import exchange
from tools.nn.companion import tactic_input as TI
from tools.nn.operative import cfg as ocfg
from tools.nn.operative import frame as FR
from tools.nn.operative import groups as GR
from tools.nn.operative import inputs as OIN
from tools.nn.operative import net as ONN
from tools.nn.tactic import decide as DC
from tools.nn.tactic import raw as TR

WINDOW_S = int(ocfg()["decide_s"])
ROW_KEYS = ("x", "z", "b", "ox", "oz", "t", "f", "mv", "m", "r", "ow", "men")


def groups_of(doc):
    """({k: [names]} own, {k: [names]} enemy) of a state document's groups."""
    g = doc.get("groups") or {}
    own = {int(k): list(n) for k, n in (g.get("own") or [])}
    enemy = {int(k): list(n) for k, n in (g.get("enemy") or [])}
    return own, enemy


def with_rest(own, doc, G):
    """The person's own groups and, when some own units are in none, one group more of them (the operative's walk wants
    every own unit in a group: the first free number <= G; recorded as 'rest' in the decision)."""
    grouped = {n for names in own.values() for n in names}
    rest = [u["n"] for u in doc["units"] if int(u["side"]) == 1 and (u.get("men") or 0) > 0 and not u.get("r")
            and not u.get("s") and u["n"] not in grouped]
    out = dict(own)
    k = next((k for k in range(1, G + 1) if k not in out), None)
    if rest and k is not None:
        out[k] = rest
    return out, (k if rest else None)


def signature(own, enemy):
    return json.dumps([sorted((k, sorted(v)) for k, v in own.items()), sorted((k, sorted(v)) for k, v in enemy.items())])


def operative_groups(own, enemy, slot_of, raw, axes, G):
    """GR.Groups [1, G] of the person's groups (module doc)."""
    N = raw.shape[1]
    I = TR.I
    g = GR.Groups.empty(1, N, G, raw.device)
    for k, names in own.items():
        if not 1 <= k <= G:
            continue
        j = k - 1
        slots = [slot_of[n] for n in names if n in slot_of]
        if not slots:
            continue
        for s in slots:
            g.member[0, s] = j
        g.on[0, j] = True
        g.n0[0, j] = len(slots)
        cx = float(raw[0, slots, I["x"]].mean())
        cz = float(raw[0, slots, I["z"]].mean())
        g.cx[0, j], g.cz[0, j] = cx, cz
        foes = [slot_of[n] for n in enemy.get(k, []) if n in slot_of]
        heading = float(axes.heading[0])
        if foes:
            for s in foes:
                g.target[0, j, s] = True
            g.seed[0, j] = foes[0]
            tx, tz = float(raw[0, foes, I["x"]].mean()), float(raw[0, foes, I["z"]].mean())
            heading = float(torch.rad2deg(torch.atan2(torch.tensor(tx - cx), torch.tensor(tz - cz)))) % 360.0
        g.heading[0, j] = heading
        g.verb[0, j] = DC.VERBS.index("hold")
        g.pace[0, j] = DC.PACES.index("any")
    return g


class LiveRecorder:
    def __init__(self, game, out_root=None, name="live", device="cpu", window_s=WINDOW_S, poll_s=0.02, task="hold",
                 map_root=None, allow_flat=True):
        self.game, self.device, self.window_s, self.poll_s = Path(game), device, int(window_s), poll_s
        kw = {} if map_root is None else {"map_root": map_root}
        self.seer = TI.TacticSeer(device=device, allow_flat=allow_flat, **kw)
        self.seer_kw = dict(device=device, allow_flat=allow_flat, **kw)     # (the strategist's seer, made at the deployment)
        self.P = self.seer.P
        self.G = int(ocfg()["groups"])
        self.command = DC.order_vec(1, task, "any", device)
        self.side = torch.ones(1, dtype=torch.long, device=device)
        self.folder = RC.new_folder(name, out_root)
        self.rec, self.batch, self.win, self.sec = None, None, None, 0
        self.stats = {"windows": 0, "human": 0, "keep": 0, "none": 0, "refused": 0, "seconds": 0, "orders_rows": 0,
                      "chat": 0, "commands": 0, "strategist": 0, "strategist_refused": 0}
        self.deploy_doc, self.deploy_done, self.deploy_chat = None, False, []
        self.chat_seen = 0
        self.cmd_seen = 0
        self.ov_version, self.ov_sig = 0, None
        self.last_doc = None

    # ------------------------------------------------------------ the battle
    def start(self, doc):
        self.batch = doc["batch"]
        units = doc["units"]
        self.names = [u["n"] for u in units]
        self.sides = [int(u["side"]) for u in units]
        self.slot_of = {n: i for i, n in enumerate(self.names)}
        self.axes = None
        info = {"where": "game", "mode": "live", "level": "operative", "period_s": self.window_s,
                "lower": "the person's own orders (orders.jsonl)", "batch": doc["batch"], "map": doc.get("map"),
                "keys": [u.get("key") or "" for u in units], "names": self.names, "sides": self.sides,
                "started": time.strftime("%Y-%m-%d %H:%M:%S"), "fighters_lab": False}
        m = {"name": "live", "where": "game", "mode": "live", "map": doc.get("map"), "factions": doc.get("factions"),
             "units": [{k: u.get(k) for k in ("n", "side", "key", "men")} for u in units]}
        self.rec = RC.Recorder(self.folder, m, info)

    def second(self, doc):
        self.last_doc = doc
        if doc["batch"] != self.batch:
            self.start(doc)
        if doc.get("deploy"):
            # the game's deployment phase (live_label.lua: the deployment document every second): the last one is the
            # person's deployment; 'Start battle' (deploy_final, or the battle's first document) makes the strategist's
            # sample of it
            self.deploy_doc = doc
            self.chat(doc)
            self.commands(doc)
            if doc.get("deploy_final"):
                self.deployed()
            return
        if self.deploy_doc is not None and not self.deploy_done:
            self.deployed()
        r = self.seer.raw(doc)
        if self.axes is None:
            self.axes = FR.Axes.of(r, self.side, self.P.map_half)
        self.sec = int(round(float(doc.get("t", 0)) / 1000.0))
        own, enemy = groups_of(doc)
        sig = signature(own, enemy)
        self.stats["seconds"] += 1
        # the player's orders, every second
        rows = {u["n"]: {k: u.get(k) for k in ROW_KEYS} for u in doc["units"] if int(u["side"]) == 1}
        with open(self.folder / "orders.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"sec": self.sec, "t": doc.get("t"), "selection": doc.get("selection"),
                                "units": rows}, ensure_ascii=False) + "\n")
        self.stats["orders_rows"] += 1
        # (no marks over the units on the field: the person's word 07.11 - "not that"; the groups show as the game shows
        # its own: the frames over the cards, the outlines - src/apps/battle/enemy_cards.lua)
        if self.win is None:
            self.open(r, own, enemy, sig, doc)
        elif self.sec >= self.win["sec"] + self.window_s:
            self.close(r, own, enemy, sig, doc)
            self.open(r, own, enemy, sig, doc)
        self.chat(doc)
        self.commands(doc)

    def deployed(self):
        """The strategist's sample of the person's deployment (tools/label/live_deploy.py) into the record, once."""
        if self.deploy_done or self.deploy_doc is None:
            return
        self.deploy_done = True
        from tools.label import live_deploy as LD
        from tools.nn.companion import strategist_input as SI
        try:
            seer = SI.StrategistSeer(**self.seer_kw)
            smp, text = LD.sample(self.deploy_doc, self.command, seer, chat=list(self.deploy_chat))
        except Exception as e:                          # noqa: BLE001 - a broken deployment must not stop the battle's record
            text = {"sec": 0, "until": 0, "level": "strategist", "mode": "live", "where": "game", "source": "none",
                    "errors": [f"the sample: {type(e).__name__}: {e}"], "chat": list(self.deploy_chat)}
            with open(self.folder / "decisions.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(text, ensure_ascii=False) + chr(10))
            self.stats["strategist_refused"] += 1
            return
        self.rec.add(smp, text)
        self.stats["strategist"] += 1
        if text["errors"]:
            self.stats["strategist_refused"] += 1

    def commands(self, doc):
        """The player's orders as the game handed them to the bridge (src/apps/battle/live_label.lua M.COMMANDS; doc.commands,
        numbered n): each to commands.jsonl once it carries what the order set (after: each ordered unit's ordered
        position ox, oz, bearing ob, width ow, fast) or once it is 2 s old - with the battle's second and the window:
        name, target (an attack's unit), bool1 / string1 (the command's own values), units (the own selection then);
        phase 'deployment' (the deployment's orders: sec from the deployment's start, window 0) or 'battle'."""
        for m in doc.get("commands") or []:
            n = int(m.get("n") or 0)
            if n <= self.cmd_seen:
                continue
            t = float(m.get("t") or 0)
            if "after" not in m and float(doc.get("t") or 0) - t < 2000 and not doc.get("deploy"):
                break                          # (in order: wait for it)
            if "after" not in m and doc.get("deploy"):
                break                          # (the deployment's clock stands: the next document brings it)
            self.cmd_seen = n
            deploy = m.get("phase") == "deployment"
            row = {"n": n, "phase": "deployment" if deploy else "battle",
                   "sec": round(float(m.get("deploy_ms") or 0) / 1000.0, 1) if deploy else round(t / 1000.0, 1),
                   "window": 0 if deploy else self.stats["windows"] + 1,
                   "window_sec": None if deploy else (self.win["sec"] if self.win else None),
                   "name": m.get("name"), "target": m.get("unit"),
                   "bool1": m.get("bool1"), "string1": m.get("string1"), "units": m.get("selection") or [],
                   "ordered": m.get("after") or {}}
            with open(self.folder / "commands.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + chr(10))
            self.stats["commands"] += 1

    def chat(self, doc):
        """The person's chat messages (src/apps/battle/label_chat.lua; doc.chat, numbered n): each new one to chat.jsonl
        with the battle's second, the window it falls in (its number and start) and the groups, the selection and the
        enemy's pick at the moment it was sent; the window's decision keeps their numbers and texts."""
        for m in doc.get("chat") or []:
            n = int(m.get("n") or 0)
            if n <= self.chat_seen:
                continue
            self.chat_seen = n
            deploy = bool(doc.get("deploy"))
            row = {"n": n, "sec": 0.0 if deploy else round(float(m.get("t") or 0) / 1000.0, 1),
                   "window": 0 if deploy else self.stats["windows"] + 1,
                   "window_sec": None if deploy else (self.win["sec"] if self.win else None), "deploy": deploy,
                   "text": m.get("text"), "groups": m.get("groups"), "selection": m.get("selection"),
                   "enemy_pick": m.get("enemy_pick")}
            with open(self.folder / "chat.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + chr(10))
            if deploy:
                self.deploy_chat.append({"n": n, "text": row["text"]})
            elif self.win is not None:
                self.win.setdefault("chat", []).append({"n": n, "sec": row["sec"], "text": row["text"]})
            self.stats["chat"] += 1

    def open(self, r, own, enemy, sig, doc):
        full, _ = with_rest(own, doc, self.G)
        groups = operative_groups(full, enemy, self.slot_of, r, self.axes, self.G) if own else             GR.Groups.empty(1, r.shape[1], self.G, r.device)
        K = min(int(ocfg()["rows"]), r.shape[1])
        inp = OIN.build(r, self.seer.card, self.side, self.axes, groups, self.command, bounds=self.P.map_half, K=K)
        ctx = ONN.ctx_of(inp, r, self.axes, groups)
        self.win = {"sec": self.sec, "raw": r, "groups": groups, "inp": inp, "ctx": ctx, "K": K, "sig": sig,
                    "own": own, "enemy": enemy}

    def close(self, r, own, enemy, sig, doc):
        w = self.win
        self.stats["windows"] += 1
        changed = sig != w["sig"]
        human = {"own": {k: v for k, v in sorted(own.items())}, "enemy": {k: v for k, v in sorted(enemy.items())}}
        if not own and not w["own"]:
            source, act, errs = "none", None, []
        else:
            source = "human" if changed else "keep"
            # the rest: the own units the window's input has (its rows) in none of the person's groups
            slots, mine, _ = HU.op_rows(w["ctx"])
            grouped = {n for names in own.values() for n in names}
            rest = [self.names[sl] for sl, o in zip(slots, mine) if o and self.names[sl] not in grouped]
            full = dict(own)
            # (in groups of at most max_group, the tactician's limit, in the first free numbers)
            from tools.nn.tactic import cfg as tcfg
            size = int(tcfg()["max_group"])
            rest_ks = []
            for i in range(0, len(rest), size):
                k = next((k for k in range(1, self.G + 1) if k not in full), None)
                if k is None:
                    break
                full[k] = rest[i:i + size]
                rest_ks.append(k)
            human["rest"] = rest_ks
            alive = {sl for sl, o in zip(slots, mine) if o}
            edit = {"keep": True} if not changed else {"groups": [
                {"g": k - 1, "members": [self.slot_of[n] for n in names if self.slot_of.get(n) in alive],
                 "task": {"seed": next((self.slot_of[n] for n in enemy.get(k, []) if n in self.slot_of), None),
                          "goal": None, "form": "square", "turn": 0, "verb": "hold", "pace": "any"}}
                for k, names in sorted(full.items())
                if 1 <= k <= self.G and any(self.slot_of.get(n) in alive for n in names)]}
            act, errs = None, []
            try:
                act, plan, _ = HU.op_act(edit, w["ctx"], self.axes)
                _, errs = HU.op_walk(w["inp"], w["ctx"], act, plan)
            except HU.Refused as e:
                errs = e.errors
            if errs:
                act = None
                self.stats["refused"] += 1
        self.stats[source] += 1
        text = {"sec": w["sec"], "until": self.sec, "level": "operative", "mode": "live", "where": "game",
                "source": source, "changed": changed, "groups": human, "errors": errs, "chat": w.get("chat") or []}
        if source == "none":
            with open(self.folder / "decisions.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(text, ensure_ascii=False) + "\n")
            return
        sample = {"level": "operative", "mode": "live", "where": "game", "sec": w["sec"], "raw": w["raw"],
                  "groups": RC.groups_dict(w["groups"]), "card": self.seer.card, "side": self.side,
                  "command": self.command, "axes": RC.axes_dict(self.axes), "bounds": self.P.map_half, "K": w["K"],
                  "act": act if act is not None else {}, "source": source, "human_groups": human, "errors": errs}
        self.rec.add(sample, text)

    def overlay(self, own, enemy):
        groups = []
        for k, names in sorted(own.items()):
            groups.append({"g": k - 1, "members": [self.slot_of[n] for n in names if n in self.slot_of],
                           "target": [self.slot_of[n] for n in enemy.get(k, []) if n in self.slot_of]})
        for k, names in sorted(enemy.items()):
            if k not in own:
                groups.append({"g": k - 1, "members": [],
                               "target": [self.slot_of[n] for n in names if n in self.slot_of]})
        self.ov_version += 1
        exchange.write_atomic(self.game / GM.OVERLAY, GM.overlay_text(self.ov_version, self.names, groups))

    def finish(self, why):
        if self.rec is None or getattr(self, "finished", None):
            return getattr(self, "finished", None)
        if self.deploy_doc is not None and not self.deploy_done:
            self.deployed()
        result = {"why": why, "mode": "live", "where": "game", "seconds": self.sec, **self.stats}
        self.finished = result
        self.rec.finish(result, {"names": getattr(self, "names", []), "snaps": []})
        try:
            exchange.write_atomic(self.game / GM.OVERLAY, GM.overlay_text(self.ov_version + 1, [], []))
        except OSError:
            pass
        return result

    def run(self, idle_exit_s=0, out=print):
        st_f = self.game / exchange.STATE
        old = exchange.read_state(st_f)
        last = (old["batch"], old["move"], bool(old.get("done"))) if old else None
        last_new = time.monotonic()
        while True:
            d = exchange.read_state(st_f)
            if d is not None and d.get("live") and (d["batch"], d["move"], bool(d.get("done"))) != last:
                last, last_new = (d["batch"], d["move"], bool(d.get("done"))), time.monotonic()
                if d.get("done"):
                    if self.win is not None and self.last_doc is not None:
                        # the last, partial window: a decision only when the groups changed in it
                        own, enemy = groups_of(self.last_doc)
                        if signature(own, enemy) != self.win["sig"]:
                            self.close(self.win["raw"], own, enemy, signature(own, enemy), self.last_doc)
                    res = self.finish("the battle's end")
                    out(f"live: done {res}")
                    return res
                self.second(d)
                if self.sec % self.window_s == 0:
                    own, enemy = groups_of(d)
                    out(f"live: t {self.sec} s, own groups {own}, enemy groups {enemy}, {self.stats}")
                continue
            if idle_exit_s and time.monotonic() - last_new > idle_exit_s:
                res = self.finish("no state for a while")
                out(f"live: stopped {res}")
                return res
            time.sleep(self.poll_s)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--game", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--name", default="live")
    ap.add_argument("--idle-exit", type=float, default=0)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    torch.set_num_threads(2)
    rec = LiveRecorder(a.game, Path(a.out) if a.out else None, a.name)
    print(f"live: watching {Path(a.game) / exchange.STATE}; the record {rec.folder}")

    def stop(*_):
        # docker stop (live.ps1 after the game closed): the record's result before going
        print(f"live: stopped {rec.finish('stopped')}")
        sys.exit(0)
    import signal
    signal.signal(signal.SIGTERM, stop)
    rec.run(a.idle_exit)


if __name__ == "__main__":
    main()

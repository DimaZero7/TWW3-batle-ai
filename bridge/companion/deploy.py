"""The strategist's deployment in the game (bridge/lua/apps/bridge/deploy_adapter.lua, bridge/docs/apps/bridge.md):
the deployment document -> the strategist's input (strategist_input.StrategistSeer, one to one with the simulator's) ->
its choice (its network, greedy, or its formula: strategist/teacher.play_formula) -> every own unit's place by the same
code the simulator places with (strategist/deploy.places: x, z, the engine's facing, the width) -> the plan file. The point
sent is the formation's centre + half its depth ahead (teleport_point: the game stands the centre that far behind it).

Before the plan is written the places are checked as in the simulator (strategist/deploy.errors on a simulator state built
with my units there: a unit in no group, a formation not wholly inside my zone, two overlapping; plus a group with no open
cell): the count goes into the record and the log line, the plan is written anyway (the game logs its own check of the
places it reads back). The operative's starting command (verb, pace) is in the record only: no operative in the game yet.

    Strategist(who='formula' | <checkpoint>, task='hold' | 'destroy').decide(doc) -> (plan text, record)
    plan_text(batch, places, think_ms); teleport_point(x, z, b, depth, men0)
"""
import time

import torch

from bridge.companion import strategist_input as SI
from bridge.companion import unit_input as UI
from sim import scenario
from net.strategist import battle as BT
from net.strategist import deploy as DP
from net.strategist import teacher as TC
from net.tactic import decide as DC
from net.tactic import raw as TR

DOC = "tww3_bai_nn_deploy.json"       # bridge/lua/apps/bridge/deploy_adapter.lua DOC_FILE, PLAN_FILE
PLAN = "tww3_bai_nn_plan.txt"
FORMAT = "tww3_bai_nn_plan 1"         # bridge/lua/apps/bridge/services.lua PLAN_FORMAT
TASKS = ("hold", "destroy")


def plan_text(batch, places, think_ms=None):
    """places: [(name, x, z, b, width, cx, cz)] - x, z the point sent to teleport_to_location, cx, cz the formation's
    centre the simulator has (the game checks the place it reads against the centre)."""
    lines = [FORMAT, f"batch {batch}"]
    if think_ms is not None:
        lines.append(f"think_ms {think_ms:.1f}")
    for name, x, z, b, w, cx, cz in places:
        lines.append(f"place {name} {x:.2f} {z:.2f} {b:.4f} {w:.2f} {cx:.2f} {cz:.2f}")
    lines.append("end")
    return "\n".join(lines) + "\n"


def teleport_point(x, z, b, depth, men0):
    """The point to send teleport_to_location for a formation centred at (x, z) facing b, depth deep (m): half the depth
    ahead along the facing - the game stands the formation's centre about half its depth behind the point it is sent
    (07.11.2026, run 20261107-014403: spearmen 3.2-4.8 m back at a simulator half depth of 4.8, archers 4.0-4.1 at
    5.25). A unit of one (a lord) has no ranks: the point is its centre (0.0 m there). Floats or tensors."""
    depth, men0 = torch.as_tensor(float(depth) if not torch.is_tensor(depth) else depth), torch.as_tensor(men0)
    half = torch.where(men0 > 1, depth / 2, torch.zeros_like(depth))
    a = torch.deg2rad(torch.as_tensor(float(b) if not torch.is_tensor(b) else b))
    return torch.as_tensor(x) + half * torch.sin(a), torch.as_tensor(z) + half * torch.cos(a)


def state_with(laid, places, params, device="cpu"):
    """(st, own [1, N'], index [N] the document's unit -> its slot): a simulator state of the laid-out document with my
    units at places {name: (x, z, b, width)} - deploy.errors' geometry."""
    fac = laid.get("factions") or {}
    units = laid["units"]
    sides = {1: [], 2: []}
    for u in units:
        s = int(u["side"])
        p = places.get(u["n"]) if s == 1 else None
        row = {"key": u.get("key") or "", "x": u["x"], "z": u["z"], "b": u["b"]}
        if p is not None:
            row.update(x=p[0], z=p[1], b=p[2], width=p[3])
        sides[s].append(row)
    H = max(len(sides[1]), len(sides[2]), 1)
    st = scenario.build([{"attacker": int(laid.get("attacker", 2)),
                          "sides": {1: {"faction": fac.get("own"), "units": sides[1]},
                                    2: {"faction": fac.get("enemy"), "units": sides[2]}}}], params, device=device,
                        per_side=H)
    index, k = [], {1: 0, 2: 0}
    for u in units:
        s = int(u["side"])
        index.append((s - 1) * H + k[s])
        k[s] += 1
    return st, st.u["side"] == 1, index


class Strategist:
    """The deployment's decision of one battle (module doc). who: 'default' (net/config/strategist.json default - the
    simulator's chain takes the same), 'formula' or a strategist checkpoint (net.strategist.ckpt.restore); task: the battle's task
    from above (no strategist above yet: the companion's --strategist-task)."""

    def __init__(self, who="default", task="hold", device="cpu", params=None, map_root=UI.MAPS, allow_flat=True):
        assert task in TASKS, task
        if who == "default":
            from net.strategist import default
            who = default()
        self.seer = SI.StrategistSeer(map_root, device, params, allow_flat)
        self.P, self.device, self.task, self.who = self.seer.P, device, task, who
        self.net = None
        self.plans = {}          # {batch: the plan the chain starts from} (chain.groups_of_plan; companion --chain)
        if who != "formula":
            from bridge.config import resolve as _path
            from net.strategist import ckpt as SC
            self.net, _, _ = SC.restore(str(_path(who)), device)
            self.net.eval()

    @torch.no_grad()
    def choose(self, inp, ctx, net):
        return TC.play_formula(inp, ctx) if net is None else net(inp, ctx, greedy=True)

    def places_of(self, doc, laid, inp, out):
        """({name: (x, z, b, width)}, deployment errors, groups [{units, cx, cz, heading, line}])."""
        raw = self.seer.raw
        names = [u["n"] for u in laid["units"]]
        N = len(names)
        I = TR.I
        own = raw[..., I["side"]] == 1
        shooter = (raw[..., I["range"]] > 0) & own
        width0 = torch.tensor([[float(u.get("ow") or 0.0) for u in laid["units"]]], device=self.device)
        plan = out["plan"]
        mem = BT.slots_of(plan, inp, N)
        pl = DP.places(mem, shooter, plan["cx"], plan["cz"], plan["heading"], plan["line"], width0, DP.slot_of(plan))
        places = {names[i]: (float(pl["x"][0, i]), float(pl["z"][0, i]), float(pl["b"][0, i]), float(pl["width"][0, i]))
                  for i in range(N) if bool(own[0, i]) and bool(pl["on"][0, i])}
        self.chain_plan = {"group": {names[i]: int(pl["group"][0, i]) for i in range(N) if bool(own[0, i])},
                           **{k: plan[k][0].tolist() for k in ("on", "cx", "cz", "heading", "line", "n")},
                           "verb": DC.VERBS[int(plan["verb"][0])], "pace": DC.PACES[int(plan["pace"][0])]}
        st, own_st, index = state_with(laid, places, self.P, self.device)
        member = torch.full(own_st.shape, -1, dtype=torch.long, device=self.device)
        for i in range(N):
            if bool(own[0, i]):
                member[0, index[i]] = pl["group"][0, i]
        spacing = self.P.sim["formation"]["spacing_m"]
        zone = SI.zone_of(doc["zones"]["own"], self.device)
        errors = int(DP.errors(st, own_st, member, zone, spacing).sum()) + int(plan["no_cell"].sum())
        # the points sent: each formation's centre + half its depth ahead (teleport_point; the depth: geometry.dims,
        # the simulator's formation of that width and men)
        from sim import geometry
        _, depth = geometry.dims(st.u, spacing)
        sent = {}
        for i in range(N):
            if names[i] in places:
                k = index[i]
                x, z, b, _ = places[names[i]]
                px, pz = teleport_point(x, z, b, depth[0, k], st.u["men0"][0, k])
                sent[names[i]] = (float(px), float(pz))
        self.sent = sent
        groups = []
        for g in range(mem.shape[1]):
            if bool(plan["on"][0, g]):
                groups.append({"units": [names[i] for i in range(N) if bool(mem[0, g, i])],
                               "cx": round(float(plan["cx"][0, g]), 1), "cz": round(float(plan["cz"][0, g]), 1),
                               "heading": round(float(plan["heading"][0, g]), 2), "line": bool(plan["line"][0, g])})
        return places, errors, groups

    def decide(self, doc):
        t0 = time.perf_counter()
        command = DC.order_vec(1, self.task, "any", self.device)
        inp, ctx, laid = self.seer.input(doc, command)
        out = self.choose(inp, ctx, self.net)
        places, errors, groups = self.places_of(doc, laid, inp, out)
        self.plans[doc["batch"]] = self.chain_plan      # the groups the chain's operative starts from (strategist/battle)
        think_ms = (time.perf_counter() - t0) * 1000
        rows = [(n, *self.sent[n], p[2], p[3], p[0], p[1]) for n, p in places.items()]
        record = {"batch": doc["batch"], "who": "formula" if self.net is None else str(self.who), "task": self.task,
                  "think_ms": round(think_ms, 1), "errors": errors, "groups": groups,
                  "verb": DC.VERBS[int(out["plan"]["verb"][0])], "pace": DC.PACES[int(out["plan"]["pace"][0])],
                  "places": {n: [round(p[0], 2), round(p[1], 2), round(p[2], 4), round(p[3], 2)]
                             for n, p in places.items()},
                  "sent": {n: [round(v, 2) for v in xz] for n, xz in self.sent.items()},
                  "own": sum(1 for u in doc["units"] if int(u["side"]) == 1),
                  "enemy": sum(1 for u in doc["units"] if int(u["side"]) == 2)}
        if self.net is not None:
            # the formula on the same input: what the teacher would have done (the log only)
            sent = self.sent
            f_places, f_errors, f_groups = self.places_of(doc, laid, inp, self.choose(inp, ctx, None))
            self.sent = sent
            record["formula"] = {"errors": f_errors, "groups": f_groups}
        return plan_text(doc["batch"], rows, think_ms), record


class LinePlace:
    """The place switch (companion --strategist <a .json>; drill S0-1's crest8 check, tmp/drill_s1/design_merged.md
    section 6 (b)): no strategist - my infantry on the line the file gives, by the simulator's rule (net/strategist/
    crest.places: the stage's rows, row_m apart, left to right in slot order own_u1, own_u2, ...), the lord lord_back_m
    behind their middle (net/stage0/battle._lord_place's rule), each at the simulator's default frontage (laid_out).
    The file: {"line": {"x", "z", "b"}, "n": my infantry count (asserted), ...} in the map's world metres (the finder's,
    = the battle file's after its shift). The plan and the record as Strategist.decide's; the check (deploy.errors: a
    formation not wholly in my zone, two overlapping) counted in the record, the plan written anyway."""

    def __init__(self, path, device="cpu", params=None):
        import json
        from pathlib import Path
        from sim.params import load
        self.path = str(path)
        self.spec = json.loads(Path(path).read_text(encoding="utf-8"))
        self.P = params or load()
        self.device, self.task, self.who = device, "hold", f"line:{Path(path).name}"
        self.plans = {}

    def places_of(self, doc):
        from net.strategist import crest as CR
        laid = laid_out_doc(doc, self.P, self.device)
        own = [u for u in laid["units"] if int(u["side"]) == 1]
        lords = [u for u in own if u["n"].endswith("_lord")]
        inf = sorted([u for u in own if not u["n"].endswith("_lord")], key=lambda u: _slot_no(u["n"]))
        line = self.spec["line"]
        n = int(self.spec.get("n", len(inf)))
        assert n == len(inf), f"the line file is for {n} units, the battle has {len(inf)} of mine (no lord)"
        pts = CR.places({"x": float(line["x"]), "z": float(line["z"]), "b": float(line["b"])}, n)
        places = {u["n"]: (x, z, b, float(u["ow"])) for u, (x, z, b) in zip(inf, pts)}
        # (the lord: crest.lord_place - lord_back_m behind the line, kept inside my zone's rear; the simulator's rule too)
        lx, lz, lb = CR.lord_place(pts, float(line["b"]), doc["zones"]["own"])
        for u in lords:
            places[u["n"]] = (lx, lz, lb, float(u["ow"]))
        # (refused, not logged: a unit's footprint on blocked ground - crest.layout_ok on the line file's map)
        mp = self.spec.get("map")
        if mp and not CR.layout_ok(CR.ground(mp), pts):
            raise ValueError(f"{self.who}: a unit of the line stands on blocked ground of {mp} - the plan is refused")
        return laid, places

    def decide(self, doc):
        from sim import geometry
        t0 = time.perf_counter()
        laid, places = self.places_of(doc)
        st, own_st, index = state_with(laid, places, self.P, self.device)
        names = [u["n"] for u in laid["units"]]
        member = torch.full(own_st.shape, -1, dtype=torch.long, device=self.device)
        for i, nm in enumerate(names):
            if nm in places:
                member[0, index[i]] = 0          # (one group; the lord too: out of the zone counts, as the formula's)
        zone = SI.zone_of(doc["zones"]["own"], self.device)
        spacing = self.P.sim["formation"]["spacing_m"]
        errors = int(DP.errors(st, own_st, member, zone, spacing).sum())
        if errors:                                       # (the line's geometry breaks the deployment's rules: refused)
            raise ValueError(f"{self.who}: {errors} deployment errors (net/strategist/deploy.errors) - the plan is refused")
        _, depth = geometry.dims(st.u, spacing)
        sent = {}
        for i, nm in enumerate(names):
            if nm in places:
                k = index[i]
                x, z, b, _ = places[nm]
                px, pz = teleport_point(x, z, b, depth[0, k], st.u["men0"][0, k])
                sent[nm] = (float(px), float(pz))
        think_ms = (time.perf_counter() - t0) * 1000
        rows = [(nm, *sent[nm], p[2], p[3], p[0], p[1]) for nm, p in places.items()]
        record = {"batch": doc["batch"], "who": self.who, "task": self.task, "think_ms": round(think_ms, 1),
                  "errors": errors, "line": self.spec.get("line"), "site": self.spec.get("site"),
                  "groups": [{"units": [nm for nm in places if not nm.endswith("_lord")],
                              "cx": round(float(self.spec["line"]["x"]), 1), "cz": round(float(self.spec["line"]["z"]), 1),
                              "heading": round(float(self.spec["line"]["b"]), 2), "line": True}],
                  "verb": "hold", "pace": "any",
                  "places": {nm: [round(p[0], 2), round(p[1], 2), round(p[2], 4), round(p[3], 2)] for nm, p in places.items()},
                  "sent": {nm: [round(v, 2) for v in xz] for nm, xz in sent.items()},
                  "own": sum(1 for u in doc["units"] if int(u["side"]) == 1),
                  "enemy": sum(1 for u in doc["units"] if int(u["side"]) == 2)}
        return plan_text(doc["batch"], rows, think_ms), record


def _slot_no(name):
    import re
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else 10 ** 6


def laid_out_doc(doc, params, device="cpu"):
    return SI.laid_out(doc, params, device)


def make(who="default", task="hold", device="cpu"):
    """The deployment's answerer of companion --strategist: a .json - the place switch (LinePlace), else Strategist."""
    if str(who).endswith(".json"):
        from bridge.config import resolve as _path
        return LinePlace(str(_path(who)), device)
    return Strategist(who, task, device)


def describe(record):
    gs = " | ".join(f"g{i} [{', '.join(n.removeprefix('own_') for n in g['units'])}] at ({g['cx']:.0f}, {g['cz']:.0f})"
                    f" facing {g['heading']:.0f}{' line' if g['line'] else ''}" for i, g in enumerate(record["groups"]))
    return (f"strategist {record['who']} task {record['task']}: {len(record['places'])} of {record['own']} units "
            f"placed, errors {record['errors']}, verb {record['verb']}, think {record['think_ms']:.0f} ms | {gs}")

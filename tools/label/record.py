"""The battle's folder build/label/<date_time>_<name>/ and reading it back for training (torch).

    matchup.json     the matchup as it started
    session.json     who played which level, the period, the levels above and below, the nets
    samples.pt       the samples: a list of dicts of tensors, one a decision of the person's level - the raw numbers its
                     input rebuilds from without a difference (the way PPO keeps them: operative/env.rebuild,
                     tactic/env.rebuild), the person's choices (act: the walk's format), the formula's choices then
    decisions.jsonl  the same decisions as text: the second, the source (human / keep / hint: the formula's hint taken
                     unchanged), the edit, what the code made of it (groups, zones, frames)
    replay.json      a snapshot a second (view.snapshot: every unit, hidden ones marked), for watching the battle again
    result.json      the end: why, the time, the winner, the gold trade, the units left, the decisions

    load(folder)            the samples
    rebuild(sample)         operative: (inp, ctx, act) - inp exactly as operative/inputs.build gave it then;
                            tactician: (inp, order, parcel, tiles, act, prev) - inp as tactic/inputs.build gave it
    labels(sample)          operative: the labels of net.choose(labels=...) (the teacher's term with the person as the
                            teacher: the target a one-hot of the seed's row)
"""
import datetime
import json
from dataclasses import fields

import torch

from tools import config as project
from tools.label import cfg
from tools.label import matchup as MU
from tools.nn.operative import frame as FR
from tools.nn.operative import groups as GR
from tools.nn.operative import inputs as IN
from tools.nn.operative import net as NN
from tools.nn.tactic import inputs as TIN
from tools.nn.tactic import parcel as PC
from tools.nn.tactic import uzone as U
from tools.nn.tactic import zone as ZN

FORMAT = "label_v1"


def new_folder(name, root=None):
    root = project.ROOT / cfg()["out"] if root is None else root
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    f = root / f"{stamp}_{MU.safe(name)}"
    k = 1
    while f.exists():
        k += 1
        f = root / f"{stamp}_{MU.safe(name)}_{k}"
    f.mkdir(parents=True)
    return f


def cpu(x):
    if torch.is_tensor(x):
        return x.detach().cpu().clone()
    if isinstance(x, dict):
        return {k: cpu(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(cpu(v) for v in x)
    return x


def groups_dict(g):
    return {f.name: getattr(g, f.name) for f in fields(g)}


def axes_dict(a):
    return {"cx": a.cx, "cz": a.cz, "heading": a.heading, "field": dict(a.field)}


def parcel_dict(p):
    return {n: getattr(p, n) for n in p.__dataclass_fields__}


def tiles_dict(t):
    return {"cx": t.cx, "cz": t.cz, "heading": t.heading, "rows": t.rows, "cols": t.cols, "mask": t.mask,
            "tile_m": t.tile_m}


class Recorder:
    def __init__(self, folder, matchup, session):
        self.folder = folder
        self.samples = []
        (folder / "matchup.json").write_text(json.dumps(matchup, ensure_ascii=False, indent=1), encoding="utf-8")
        (folder / "session.json").write_text(json.dumps({"format": FORMAT, **session}, ensure_ascii=False, indent=1),
                                             encoding="utf-8")

    def add(self, sample, text):
        self.samples.append(cpu({"format": FORMAT, **sample}))
        torch.save(self.samples, self.folder / "samples.pt")
        with open(self.folder / "decisions.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(text, ensure_ascii=False) + "\n")

    def finish(self, result, replay):
        (self.folder / "replay.json").write_text(json.dumps(replay, ensure_ascii=False), encoding="utf-8")
        (self.folder / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


def load(folder):
    return torch.load(folder / "samples.pt", map_location="cpu", weights_only=False)


def rebuild(s, device="cpu"):
    """The level's input of a sample (module doc)."""
    d = lambda v: v.to(device) if torch.is_tensor(v) else v                       # noqa: E731
    if s["level"] == "operative":
        g = GR.Groups(**{k: d(v) for k, v in s["groups"].items()})
        a = s["axes"]
        axes = FR.Axes(d(a["cx"]), d(a["cz"]), d(a["heading"]), {k: d(v) for k, v in a["field"].items()})
        raw = d(s["raw"])
        inp = IN.build(raw, d(s["card"]), d(s["side"]), axes, g, d(s["command"]), bounds=s["bounds"], K=s["K"])
        ctx = NN.ctx_of(inp, raw, axes, g)
        return inp, ctx, {k: d(v) for k, v in s["act"].items()}
    if s["level"] == "tactic":
        p = PC.Parcel(**{k: d(v) for k, v in s["parcel"].items()})
        t = s["tiles"]
        tiles = ZN.Tiles(d(t["cx"]), d(t["cz"]), d(t["heading"]), d(t["rows"]), d(t["cols"]), d(t["mask"]),
                         tile_m=t["tile_m"])
        prev = None if s["prev"] is None else {k: d(v) for k, v in s["prev"].items()}
        areas = None if prev is None else U.world(prev, tiles)
        inp = TIN.build(d(s["raw"]), d(s["card"]), p, tiles, areas, bounds=s["bounds"], rows=s["K"])
        return inp, d(s["order"]), p, tiles, {k: d(v) for k, v in s["act"].items()}, prev
    raise ValueError(f"no rebuild for the level {s['level']!r}")


def labels(s, K=None):
    """The operative sample's labels for net.choose(labels=...): the act, the target a one-hot [1, G, K + 1]."""
    act = s["act"]
    K = int(s["K"] if K is None else K)
    lb = dict(act)
    lb["target"] = torch.nn.functional.one_hot(act["target"].clamp(max=K), K + 1).float()
    return lb

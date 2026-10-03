"""The drills' adaptive teacher (run.py --drill-teach auto; docs/en/training/training.md "Drills",
"Teacher"): how much of each drill's battles the skilled script labels, from how far the network is
behind the script on that drill.

Plain python (no torch): runs in the project's .venv.

After every evaluation (test5's drill block: the network's and the drill's skilled script's win rates
on the same DRILL_EVAL_SEEDS battles), per taught drill d:

    deficit_d = max(0, skilled_d - net_d) / skilled_d
    share_d   = 0                              if deficit_d <= MATCH (the network matches the script)
                min(cap, k x deficit_d)        else

share_d is the share of drill d's battles whose units the script labels (drills/teach.py: a battle is
drawn when it starts, so a labelled battle is labelled whole); the imitation weight on a labelled unit is
a fixed w (--drill-teach-weight). The term is normalised over ALL the drill's units (labelled or not),
so the pull on drill d is w x share_d: share 1 is the manual teacher at weight w. Battles of no drill
are never labelled.

The defaults, from the kiting run (build/nn-train/test5/w5_kiting_teach, manual teacher w 0.15 -> 0
over 15 min on every kiting battle): the drill flipped from win 0.0 to 1.0 between updates 13 and 19
(~3.5 min, w ~0.13: ~0.47 weight-minutes of pull) and stayed 1.00 after the weight reached 0 - a push,
not a leash, is all a drill needs, and the manual run gave ~2.4x the pull it needed. WEIGHT 0.15 (what
worked); CAP 0.25: a drill as far behind as kiting was gets 0.0375 of pull, ~0.47 weight-minutes in
~12.5 min (within one 15-min run), while >= 75 % of every drill's battles stay pure PPO; K 0.5: the cap
from a deficit of 0.5 on, below it the pull falls with the gap (counter's 0.30 -> 0.15); MATCH 0.05:
~one standard error of a 128-battle win rate, so noise around a matched drill does not switch it on.

Until the run's first evaluation the shares come from the previous drill numbers: test5's "before"
evaluation of the starting network, else the init checkpoint's own test5 evaluation (prior()), else
UNKNOWN (half the cap) for every taught drill.
"""
import json
import re
from pathlib import Path

WEIGHT = 0.15
K = 0.5
CAP = 0.25
MATCH = 0.05
ROOT = Path(__file__).resolve().parents[3]


def deficit(net, skilled):
    """How far the network is behind the skilled script, relative to the script: max(0, skilled - net) /
    skilled; None without both numbers or with a script that never wins."""
    if net is None or skilled is None or skilled <= 0:
        return None
    return max(0.0, float(skilled) - float(net)) / float(skilled)


def share(d, k=K, cap=CAP, match=MATCH):
    """The share of a drill's battles the teacher labels for deficit d: 0 at or below `match`, else
    k x d within [0, cap]; None for an unknown deficit."""
    if d is None:
        return None
    if d <= match:
        return 0.0
    return min(float(cap), max(0.0, float(k) * d))


def numbers(drills_eval, name):
    """(net win rate, skilled win rate) of a drill in an evaluation's drill block (evaluate.play_drills;
    test5's eval json "drills"); (None, None) without it."""
    d = (drills_eval or {}).get(name) or {}
    sk = ((d.get("scripts") or {}).get("skilled") or {}).get("win_rate")
    return d.get("win_rate"), sk


def _load(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _path(p):
    p = Path(p)
    if str(p).startswith("/repo/"):
        p = ROOT / str(p)[len("/repo/"):]
    return p if p.is_absolute() or p.exists() else ROOT / p


def prior(init):
    """(the drill block, where from) of the last evaluation of the init checkpoint, or (None, None): a
    test5 point m<minute>.pt -> its eval_m<minute>.json, else report.json's trend at that minute (the last
    minute is after.json's); a run's latest.pt of a test5 run (runs/test5_<label>) -> that test5 folder's
    after.json."""
    if not init:
        return None, None
    p = _path(init)
    cands = []
    m = re.fullmatch(r"m(\d+(?:\.\d+)?)", p.stem)
    if m:
        minute = float(m.group(1))
        cands.append(p.parent / f"eval_m{minute:g}.json")
        rep = _load(p.parent / "report.json") or {}
        point = (rep.get("trend") or {}).get(f"{minute:g}")
        if point and point.get("drills"):
            return point["drills"], f"{p.parent / 'report.json'} trend minute {minute:g}"
        if (rep.get("train_args") or {}).get("minutes") == minute:
            cands.append(p.parent / "after.json")
    if p.stem == "latest" and p.parent.name.startswith("test5_"):
        cands.append(p.parent.parent.parent / "test5" / p.parent.name[len("test5_"):] / "after.json")
    for c in cands:
        ev = _load(c)
        if ev and ev.get("drills"):
            return ev["drills"], str(c)
    return None, None


class Auto:
    """The adaptive teacher's state: shares {drill: share of its battles labelled} and weights {drill: w}.
    bind() by the training (the drills it teaches), observe() after every evaluation; the training reads
    `shares` before every update."""

    def __init__(self, k=K, cap=CAP, weight=WEIGHT, prior=None, source=None, match=MATCH):
        self.k, self.cap, self.weight, self.match = float(k), float(cap), float(weight), float(match)
        self.prior, self.source = prior, source
        self.names = ()
        self.shares = {}
        self.rows = []
        self.history = []                        # the rows of every observe(), the first bind()'s

    @property
    def unknown(self):
        """The share of a drill with no numbers yet: half the cap."""
        return self.cap / 2

    def bind(self, names):
        """The taught drills (the READY ones the run plays); their shares from the prior numbers (UNKNOWN
        without) -> the table rows."""
        self.names = tuple(names)
        self.shares = {n: self.unknown for n in self.names}
        return self.observe(self.prior, source=self.source or ("default" if self.prior is None else "prior"))

    def weights(self):
        """{drill: the imitation weight on its labelled units} (fixed)."""
        return {n: self.weight for n in self.names}

    def observe(self, drills_eval, agreement=None, source="evaluation"):
        """New shares from an evaluation's drill block -> rows [{drill, net, skilled, deficit, share_was,
        share, agree}] (agree: {drill: the agreement since the last point}, the training log's; a drill
        without numbers keeps its share)."""
        rows = []
        for n in self.names:
            net, sk = numbers(drills_eval, n)
            d = deficit(net, sk)
            s = share(d, self.k, self.cap, self.match)
            was = self.shares.get(n, self.unknown)
            if s is not None:
                self.shares[n] = s
            rows.append({"drill": n, "net": net, "skilled": sk, "deficit": None if d is None else round(d, 4),
                         "share_was": round(was, 4), "share": round(self.shares[n], 4),
                         "agree": (agreement or {}).get(n), "source": source})
        self.rows = rows
        self.history.append(rows)
        return rows

    def meta(self):
        return {"k": self.k, "cap": self.cap, "weight": self.weight, "match": self.match}


def table(rows, title=None):
    """Markdown lines: drill | net win | skilled win | deficit | teacher share (was -> now) | agreement."""
    if not rows:
        return []
    f = (lambda v, fmt="{:.3f}": "-" if v is None else fmt.format(v))
    out = ([title, ""] if title else []) + [
        "| drill | net win | skilled win | deficit | teacher share (was → now) | agreement |",
        "|---|---|---|---|---|---|"]
    out += [f"| {r['drill']} | {f(r['net'])} | {f(r['skilled'])} | {f(r['deficit'])} | "
            f"{f(r['share_was'], '{:.2f}')} → {f(r['share'], '{:.2f}')} | {f(r.get('agree'))} |" for r in rows]
    return out

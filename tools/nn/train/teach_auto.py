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

A drill evaluated mostly on its EMBEDDED frame (drills.EMBED; the drill block's "frames") is measured by
its gold trade instead of the win rate (the rest of a normal battle blurs the win rate): with the naive
script's trade beside, the network's place between the two scripts, score = (net - naive) / (skilled - naive),
against the skilled script's 1: deficit = max(0, 1 - score) (numbers()).

Until the run's first evaluation the shares come from the previous drill numbers: test5's "before"
evaluation of the starting network, else the init checkpoint's own test5 evaluation (prior()), else
UNKNOWN (half the cap) for every taught drill.

The teacher in NORMAL battles (run.py --teach-normal; Transfer below; rollout.Battles teach_normal): the
drill's teacher script labels our units in the ordinary training battles at the drill's moments only (its
transfer detector's situation), as much as the network still needs there. After every evaluation (test5's
normal battles: the transfer block, drills/transfer.py), per drill d with a transfer detector:

    gap_d   = max(0, ref_d - net_d) / ref_d    net_d: the network's APPLIED share (the situation's unit-seconds
                                              where it uses the skill), ref_d: ai_like's on the same battles
    share_d = 0                    if gap_d <= NORMAL_MATCH (the network applies the skill as often as ai_like)
              min(cap, k x gap_d)  else

A drill may set its own reference, cap and stop (drills.Drill transfer_ref, teach_cap, teach_stop; direct_fire:
ai_like never uses that skill, so its reference is a fixed applied share 0.6, its cap 0.05, and the overall rating
falling 0.15 below the run's first evaluation, or the network's mistake share of the drill rising 0.1 above it,
switches it off for the rest of the run - the kiting teacher of step s45 sat at its cap all the step and the network
took "the shooter runs" wider than taught).
share_d is the share of the NORMAL battles whose units the script labels at the moments (under the name
"<drill>@normal"); the weight a fixed --teach-normal-weight; the pull weight x share. Smaller than the drills'
(NORMAL_CAP, NORMAL_WEIGHT): it acts on the battles the network is judged by; it switches itself off as the
transfer catches up (and stays off for a skill the network already applies more than ai_like: hold_fire).
"""
import json
import re
from pathlib import Path

WEIGHT = 0.15
K = 0.5
CAP = 0.25
MATCH = 0.05
# the teacher in normal battles (Transfer): a smaller pull than the drills' - at the cap 0.1 x 0.15 = 0.015
NORMAL_WEIGHT = 0.1
NORMAL_K = 0.5
NORMAL_CAP = 0.15
NORMAL = "@normal"       # a drill's teacher in normal battles: "<drill>@normal" (= drills.NORMAL; no torch here)
NORMAL_MATCH = 0.1       # the applied share within 10 % of ai_like's: matched (a 512-battle evaluation's noise ~0.02-0.05)
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


def embedded_mostly(d):
    """Whether a drill block's battles are mostly from the embedded frame ("frames": {frame: battles})."""
    fr = d.get("frames") or {}
    return sum(fr.values()) > 0 and fr.get("embedded", 0) >= 0.5 * sum(fr.values())


def numbers(drills_eval, name):
    """(net, skilled) of a drill in an evaluation's drill block (evaluate.play_drills; test5's eval json
    "drills"): the win rates; for a drill evaluated mostly on its embedded frame the gold-trade scores
    ((trade - naive's) / (skilled's - naive's): the skilled script 1; None when skilled does not trade above
    naive); (None, None) without it."""
    d = (drills_eval or {}).get(name) or {}
    sc = d.get("scripts") or {}
    if embedded_mostly(d):
        net, nv, sk = d.get("gold_trade"), (sc.get("naive") or {}).get("gold_trade"), (sc.get("skilled") or {}).get("gold_trade")
        if net is None or nv is None or sk is None or sk <= nv:
            return None, None
        return (float(net) - float(nv)) / (float(sk) - float(nv)), 1.0
    sk = (sc.get("skilled") or {}).get("win_rate")
    return d.get("win_rate"), sk


def transfer_numbers(transfer_eval, name):
    """(network's applied share, ai_like's) of a drill in an evaluation's transfer block (drills/transfer.py;
    test5's eval json "transfer"); (None, None) without it."""
    x = (transfer_eval or {}).get(name) or {}
    return (x.get("network") or {}).get("share"), (x.get("ai_like") or {}).get("share")


def transfer_mistake(transfer_eval, name):
    """The network's mistake share of a drill in an evaluation's transfer block, or None."""
    return (((transfer_eval or {}).get(name) or {}).get("network") or {}).get("mistake")


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


def prior(init, key="drills"):
    """(the evaluation's `key` block - "drills", or "transfer" for the teacher in normal battles - , where
    from) of the last evaluation of the init checkpoint, or (None, None): a
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
        if point and point.get(key):
            return point[key], f"{p.parent / 'report.json'} trend minute {minute:g}"
        if (rep.get("train_args") or {}).get("minutes") == minute:
            cands.append(p.parent / "after.json")
    if p.stem == "latest" and p.parent.name.startswith("test5_"):
        cands.append(p.parent.parent.parent / "test5" / p.parent.name[len("test5_"):] / "after.json")
    for c in cands:
        ev = _load(c)
        if ev and ev.get(key):
            return ev[key], str(c)
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
        self.shares = {n: self._cap(n) / 2 for n in self.names}
        return self.observe(self.prior, source=self.source or ("default" if self.prior is None else "prior"))

    def _cap(self, name):
        """The share's cap of one taught name (Transfer: a drill's own teach_cap)."""
        return self.cap

    def _stopped(self, name):
        """Whether this name's teacher is off for the rest of the run (Transfer: a drill's teach_stop)."""
        return False

    def weights(self):
        """{drill: the imitation weight on its labelled units} (fixed)."""
        return {n: self.weight for n in self.names}

    def _numbers(self, evaluation, name):
        return numbers(evaluation, name)

    def _measure(self, evaluation, name):
        """What net and skilled are: "win rate", or "trade score" (an embedded frame's, numbers())."""
        return "trade score" if embedded_mostly((evaluation or {}).get(name) or {}) else "win rate"

    def observe(self, drills_eval, agreement=None, source="evaluation"):
        """New shares from an evaluation's drill block -> rows [{drill, net, skilled, deficit, share_was,
        share, agree}] (agree: {drill: the agreement since the last point}, the training log's; a drill
        without numbers keeps its share)."""
        rows = []
        for n in self.names:
            net, sk = self._numbers(drills_eval, n)
            d = deficit(net, sk)
            s = share(d, self.k, self._cap(n), self.match)
            was = self.shares.get(n, self._cap(n) / 2)
            if s is not None:
                self.shares[n] = s
            stopped = self._stopped(n)
            if stopped:
                self.shares[n] = 0.0
            rows.append({"drill": n, "net": net, "skilled": sk, "deficit": None if d is None else round(d, 4),
                         "share_was": round(was, 4), "share": round(self.shares[n], 4),
                         "agree": (agreement or {}).get(n), "source": source,
                         "measure": self._measure(drills_eval, n), **({"stopped": True} if stopped else {})})
        self.rows = rows
        self.history.append(rows)
        return rows

    def meta(self):
        return {"k": self.k, "cap": self.cap, "weight": self.weight, "match": self.match}


class Transfer(Auto):
    """The teacher in normal battles (run.py --teach-normal): shares {"<drill>@normal": share of the normal
    battles labelled at the drill's moments} from the transfer gap (the network's applied share against
    ai_like's, transfer_numbers); bind() with the drills' names, observe() with an evaluation's transfer
    block. Rows as Auto's: net = the network's applied share, skilled = ai_like's."""

    def __init__(self, k=NORMAL_K, cap=NORMAL_CAP, weight=NORMAL_WEIGHT, prior=None, source=None, match=NORMAL_MATCH):
        super().__init__(k, cap, weight, prior, source, match)
        self.caps, self.refs, self.stops, self.stops_mistake = {}, {}, {}, {}
        self.rating0 = None                      # the run's first evaluation's rating (the stops' reference)
        self.mistake0 = {}                       # the first evaluation's mistake share of each drill (its stop's)
        self.stopped = set()

    def bind(self, drills):
        """The drills taught in normal battles; a drill's own transfer_ref, teach_cap, teach_stop (drills.Drill)
        replace ai_like's applied share as the reference and the cap, and add the stop."""
        from tools.nn.train import drills as D
        loaded = D.load(list(drills))
        for n in drills:
            d, key = loaded.get(n), D.normal_name(n)
            for attr, store in (("teach_cap", self.caps), ("transfer_ref", self.refs), ("teach_stop", self.stops),
                                ("teach_stop_mistake", self.stops_mistake)):
                if d is not None and getattr(d, attr, None) is not None:
                    store[key] = float(getattr(d, attr))
        return super().bind([D.normal_name(d) for d in drills])

    def _cap(self, name):
        return self.caps.get(name, self.cap)

    def _stopped(self, name):
        return name in self.stopped

    def _numbers(self, evaluation, name):
        net, ref = transfer_numbers(evaluation, name[:-len(NORMAL)] if name.endswith(NORMAL) else name)
        if name in self.refs and net is not None:
            ref = self.refs[name]
        return net, ref

    def observe(self, drills_eval, agreement=None, source="evaluation", rating=None):
        """Auto.observe with the stops: `rating` (the evaluation's overall rating) below the first one seen
        (rating0) by more than a drill's teach_stop switches that drill's teacher off for the rest of the run."""
        if rating is not None:
            if self.rating0 is None:
                self.rating0 = float(rating)
            for n, drop in self.stops.items():
                if float(rating) < self.rating0 - drop:
                    self.stopped.add(n)
        for n, rise in self.stops_mistake.items():
            m = transfer_mistake(drills_eval, n[:-len(NORMAL)] if n.endswith(NORMAL) else n)
            if m is None:
                continue
            self.mistake0.setdefault(n, m)
            if m > self.mistake0[n] + rise:
                self.stopped.add(n)
        return super().observe(drills_eval, agreement, source)

    def _measure(self, evaluation, name):
        return "applied share"


def table(rows, title=None, ref="skilled win", net="net win"):
    """Markdown lines: drill | net win | skilled win | deficit | teacher share (was -> now) | agreement (the
    teacher in normal battles: ref "ai_like applied", net "net applied")."""
    if not rows:
        return []
    f = (lambda v, fmt="{:.3f}": "-" if v is None else fmt.format(v))
    out = ([title, ""] if title else []) + [
        f"| drill | {net} | {ref} | deficit | teacher share (was → now) | agreement |",
        "|---|---|---|---|---|---|"]
    m = (lambda r: f" ({r['measure']})" if r.get("measure") not in (None, "win rate") else "")
    out += [f"| {r['drill']}{m(r)} | {f(r['net'])} | {f(r['skilled'])} | {f(r['deficit'])} | "
            f"{f(r['share_was'], '{:.2f}')} → {f(r['share'], '{:.2f}')} | {f(r.get('agree'))} |" for r in rows]
    return out

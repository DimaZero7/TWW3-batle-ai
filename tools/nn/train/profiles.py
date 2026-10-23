"""Metric profiles: which metrics a run computes (docs/en/training/workflow.md "Metric profiles").

Every evaluation, gate summary and gap card computes a small MANDATORY set: in test5 the rating, the
pair gold against each script (with the script baselines) and own lord deaths (test5 --no-script-metrics: own lord
deaths only, in the exam's proxy and the training window; no script battles, no baselines); in a gate summary the
outcomes, the pairs, the pair gold and own lord deaths; in a gap card the result, the trade, the gold
each side lost and own lord deaths. Everything else is a named profile chosen per run
(`--profile drills,transfer`); "full" is every profile of that tool but the opt-in ones (OPT_IN: named
on their own, "full,exam"). What a metric means never depends on the profile: a profile only says whether
it is computed.

Plain python (no torch): tools/nn/gate.py and tools/ops use it on the host.
"""
MANDATORY = "mandatory"
FULL = "full"
# test5 / evaluate.play: behaviour = behaviour.Tracker's sums (missile time in melee, flanks, piles,
# ability uses); liveliness = its liveliness counts too; fatigue = behaviour.Fatigue; transfer = the
# drills' transfer detectors in the normal battles; drills = the drill evaluations (evaluate.play_drills);
# capacity = the capacity-and-forgetting report (capacity.py); exam = the exam's proxy (exam_sim.py: the network
# against itS27_4 on the exam's seeds in the fighters lab, ~7 min a point; test5 --exam-at: the step's last point).
TEST5 = ("behaviour", "liveliness", "fatigue", "transfer", "drills", "capacity", "exam")
# Profiles "full" leaves out (costly, for one line of networks): only when named, "mandatory,exam" or "full,exam".
OPT_IN = ("exam",)
TEST5_ALIASES = {"shooters": "behaviour"}
# the game's recordings and the gap card (tools/nn/battle_metrics.py GROUPS; gate.py liveliness)
GAME = ("liveliness", "routs", "lords", "fatigue", "activity", "shooters", "wrap")
GAP = ("routs", "lords", "fatigue", "activity", "shooters", "wrap")


def parse(text, known, aliases=None):
    """The profiles named in `text` ("a,b", "full", "mandatory" or "" / None) as a tuple in `known`'s
    order; "full" = all of `known` but OPT_IN ("full,exam" adds one); mandatory alone = (). ValueError on a
    name not in known / aliases."""
    aliases = aliases or {}
    names = [n.strip().lower() for n in (text or "").replace("+", ",").split(",") if n.strip()]
    out = set(k for k in known if k not in OPT_IN) if FULL in names else set()
    for n in names:
        if n in (MANDATORY, FULL):
            continue
        n = aliases.get(n, n)
        if n not in known:
            raise ValueError(f"unknown metric profile {n!r}; known: {MANDATORY}, {', '.join(known)}, {FULL}"
                             + (f" (aliases: {', '.join(f'{a}={b}' for a, b in aliases.items())})" if aliases else ""))
        out.add(n)
    return tuple(k for k in known if k in out)


def text(chosen, known):
    """'full' when every known profile but the opt-in ones is chosen ('full+exam' with one of those), else
    'mandatory' plus the chosen ones (parse reads both back)."""
    chosen = tuple(chosen)
    base = set(k for k in known if k not in OPT_IN)
    if chosen and base and set(chosen) >= base:
        return "+".join((FULL,) + tuple(k for k in chosen if k in OPT_IN))
    return "+".join((MANDATORY,) + chosen)


def test5_needs(drill_teach, normal_adaptive):
    """{profile: why} the training's adaptive teachers read from every evaluation: run.py --drill-teach
    auto reads the drills, an adaptive --teach-normal (run.normal_drills: not fixed shares) the transfer."""
    out = {}
    if drill_teach == "auto":
        out["drills"] = "--drill-teach auto reads each evaluation's drills"
    if normal_adaptive:
        out["transfer"] = "the adaptive --teach-normal reads each evaluation's transfer"
    return out

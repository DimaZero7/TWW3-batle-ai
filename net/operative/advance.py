"""Drill O0-1 'advance on the holder' - the players of the operative level (tmp/op1/design_merged.md FINAL: the teacher
is ADVANCE / KEEP, no HALT at this level; the numbers net/config/operative.json o1). A player answers on the level's
facts of one battle (lines.Level.facts: plain scalars) with None (KEEP) or a pointer (lines.POINTERS).

    teach(f, c)     the teacher, no memory: ADVANCE -> Hold(L60) when att = 1, the second >= start_s, no member in melee,
                    no member has a comer, E stands (no unit of E closing on our centroid at >= comes_ms), the
                    front-to-front gap > far_gap_m, and the order in force is not already that line (l60_same); KEEP
                    otherwise. -> (pointer | None, the reason: the first condition that failed, or 'advance')
                    The cause from the generator: "our mission is attack and the enemy holds".
    PLAYERS         DEFAULT (the stack without the operative, as today: always KEEP), DUMB (march onto his line once:
                    opside's arm M - Hold(LM) at the first tick with att = 1, then KEEP), TEACH (teach)
    table()         the teacher's rule on hand-made states (python -m net.operative.advance)
"""
REASONS = ("advance", "role", "early", "melee", "comer", "comes", "near", "same")


def _c():
    from net.operative.lines import conf
    return conf()


def teach(f, c=None):
    """(module doc)."""
    c = c or _c()
    if int(f["att"]) != 1:
        return None, "role"
    if float(f["sec"]) < float(c["start_s"]):
        return None, "early"
    if f["melee"]:
        return None, "melee"
    if f["comer"]:
        return None, "comer"
    if float(f["e_close_max"]) >= float(c["comes_ms"]):
        return None, "comes"
    if float(f["gap"]) <= float(c["far_gap_m"]):
        return None, "near"
    if f["l60_same"]:
        return None, "same"
    return "L60", "advance"


class Default:
    name = "DEFAULT"

    def __call__(self, f, order, c=None):
        return None


class Dumb:
    """opside's arm M as the level's dumb move: the line onto E once, at the first tick with att = 1 (none in melee)."""
    name = "DUMB"

    def __call__(self, f, order, c=None):
        if int(f["att"]) == 1 and order["ptr"] == "deploy" and not f["melee"]:
            return "LM"
        return None


class Teach:
    name = "TEACH"

    def __call__(self, f, order, c=None):
        return teach(f, c)[0]


PLAYERS = {"DEFAULT": Default, "DUMB": Dumb, "TEACH": Teach}


def table():
    """The teacher on hand-made states: [(case, facts, want, got, reason)]; asserts each."""
    base = {"att": 1, "sec": 30.0, "melee": False, "comer": False, "e_close_max": 0.0, "gap": 250.0, "l60_same": False}
    cases = [("the cause: attack, he stands 250 m off", {}, "L60"),
             ("defend (att 0), he stands", {"att": 0}, None),
             ("before the first tick (15 s)", {"sec": 15.0}, None),
             ("a member in melee", {"melee": True}, None),
             ("a member has a comer", {"comer": True}, None),
             ("he comes at 1.0 m/s", {"e_close_max": 1.0}, None),
             ("he drifts at 0.9 m/s (not 'comes')", {"e_close_max": 0.9}, "L60"),
             ("he walks away (-2 m/s)", {"e_close_max": -2.0}, "L60"),
             ("front gap 100 m (= far gap)", {"gap": 100.0}, None),
             ("front gap 101 m", {"gap": 101.0}, "L60"),
             ("front gap 60 m (arrived at L60)", {"gap": 60.0}, None),
             ("already ordered to that L60", {"l60_same": True}, None),
             ("late in the battle (400 s), he stands far", {"sec": 400.0, "gap": 300.0}, "L60"),
             ("gap 460 m (held), he stands", {"gap": 460.0}, "L60")]
    out = []
    for name, d, want in cases:
        f = {**base, **d}
        got, why = teach(f)
        assert got == want, (name, got, want, why)
        out.append((name, d, want, got, why))
    return out


if __name__ == "__main__":
    for name, d, want, got, why in table():
        print(f"| {name} | {d or 'base'} | {want or 'KEEP'} | {got or 'KEEP'} | {why} |")

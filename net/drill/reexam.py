"""The short re-exam of the accepted drills after a simulator change (METHOD.md "Порядок": the weekly merge of simulator
fixes, then the night's re-check of every drill's behavioural marks). For each row of net/drills.json whose `passed`
says yes it plays a small fixed set of that drill's own scenes on the re-exam's registered seeds (split 'reexam' of
net/drill/seeds.py) with the row's weights (sha checked) on the network's own path, through the drill's own exam code
(no new measure), and compares with the row's accepted numbers by METHOD step 6's tolerances for prior drills:

    B1      does the move at the cause          not lower than the accepted by more than 0.05
    quiet   false moves without the cause        not higher than the accepted by more than 0.02
    aim     the move reaches its aim             not lower than the accepted by more than 0.10 (step 6: teacher - 0.10)
    orders  orders to the game a unit-minute     <= 1.5 x the human's 1.0 (step 6)
A line is 'pass' when the value is on the right side of its limit, 'noise' when it is past the limit by less than two
errors (the error from (k + 1) / (n + 2), exam_o1.share's rule: 0 of n is not 'no error'), 'fail' beyond. A drill:
FAIL on any fail, INSIDE NOISE on any noise, else PASS.

What each drill's lines read (its own exam code, its own set's recipe):
    T0-1  tactic.imitate.read on path N (the network drives, TEACH labels the same states) of set S's recipe mix scaled
          to n scenes: B1 = followed_N pooled (a label followed within -1..+2 ticks, imitate.marks), quiet = false_N a
          tick, aim = M4 (the target is the teacher's), orders = the N file's game_orders_min; accepted: the row's
          after.* on S. B1's two errors by battle (labels of one battle are not independent: METHOD step 5)
    O0-1  operative.exam_o1.play of the NET arm on taught's recipe (n cause scenes + n twins, fair starts, the level's
          facts on the seen foes - as the acceptance ran): B1 = _b1_own, aim = contact before the timer on the causes,
          quiet = changed on the twins, orders = orders_with_lord_min; accepted: held_out.taught (B1), after.*
    S0-1  strategist.exam_s1 on taught's recipe (n cause scenes + n twins): B1 = the network's crest = the teacher's
          (no play - the pointer is chosen before the battle: it does not move with the simulator), quiet = k != 0 on
          the twins, aim = units meeting their first enemy higher (played), orders = the rows' orders; accepted:
          held_out.taught (B1), after.*

    python -m net.drill.reexam [--drills T0-1,O0-1,S0-1] [--n N] [--device cuda] [--out tmp/reexam/<stamp>]
                               [--sim-set contact.lord_strikers=0 ...] [--lord-mode off|post|t2] [--buttons on|off]
                               [--ckpt <a candidate's weights>]
--sim-set changes sim.json's value in this process only (the cached sim.params.load() object; the file untouched).
--lord-mode: the re-exam with lords in play - net/config/stage0.json lord_script = the mode (net/unit/lord.MODES: post -
my lord at the T2 post, never fighting; t2 - the T2 lord script; group - my lord a group of one under the tactic
level's own network, tactic/side.py) and enemy_lord_fights on, lord_buttons by --buttons,
in this process only; the scene lords of T0-1 / O0-1 / S0-1 then start at the T2 post (stage0/battle.lord_back), not
lord_back_m back; off (the default) - the lords as the rows were accepted. --lords on = --lord-mode t2 --buttons on.
With a lord mode the report adds the aura exposure: the share of the tactic level's infantry decisions (my able units
at its ticks) with lord_near on and with hold_the_line on (net/tactic/raw.py effects).
--ckpt: a candidate's weights in place of the journal row's (one drill; its sha256 printed, the row's sha not checked).
Exit 0: every drill PASS; 1: a FAIL; 2: only INSIDE NOISE.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

DRILLS = ("T0-1", "O0-1", "S0-1")
SEEDS = {"T0-1": {"scene": 8_000_000, "torch": 8_900_000},
         "O0-1": {"scene": 30_000_000, "torch": 8_910_000},
         "S0-1": {"scene": 99_000_000, "torch": 8_920_000}}
_SEEDS_WHY = ("the re-exam's own ranges, split 'reexam' in net/drill/seeds.py _reexam (checked against every range of "
              "the same generator): T0-1 tactic.make 8,000,000 + 10,000 x the recipe's place (the registry's highest "
              "tactic.make was 6,860,460); O0-1 o1.scene 30,000,000 (substitutes + k x 10^6 to 35,000,512: between "
              "the exam's 20-25 x 10^6 and training's 40 x 10^6); S0-1 s1.scene 99,000,000 (above the exam's 96-98 x "
              "10^6); torch chunk seeds 8,900,000 / 8,910,000 / 8,920,000. S0-1's ridges are taught's ground pool "
              "(s1_ground.POOLS 'taught': the exam's maps, never training's)")
N_MAX = {"T0-1": 1024, "O0-1": 512, "S0-1": 1024}          # (the registered ranges' size: --n above it halts)
N = {"T0-1": 512, "O0-1": 192, "S0-1": 128}
CHUNK = {"T0-1": 256, "O0-1": 384, "S0-1": 256}
_N_WHY = ("B1's two errors 2 sqrt(p (1 - p) / n) <= 0.05 at the row's accepted p: T0-1 p 0.815 per label, 1.37 labels a "
          "scene of S (after.json of 20.11: 3,418 / 2,496) -> 512 scenes ~700 labels, 0.029 binomial, ~0.04 by battle; "
          "O0-1 p 0.995 needs 8 cause scenes, but on CUDA a chunk is padded to its full 384 (exam_o1.play): 192 + 192 "
          "twins cost what 8 do and keep the exam's compiled shape; S0-1 p 0.879 (taught) -> 2 sqrt(0.879 x 0.121 / "
          "128) = 0.058 - its B1 needs no play, so the band is the scene count's; 128 + 128 twins = one chunk of 256 "
          "(the exam's shape). The chunks are the acceptance exams' (compiled kernels in the cache)")
MIN_CHUNK = {"T0-1": 1.6, "O0-1": 11.4, "S0-1": 4.5}
_MIN_WHY = ("GPU minutes a chunk, measured: T0-1 tmp/drill1_w/before/after.log S 32.1 min = 2 paths x 10 chunks of 256; "
            "O0-1 tmp/engine/o1/chain_a2/NET_taught.log 11.4 min for one chunk of 384; S0-1 tmp/drill_s1/c2 taught NET "
            "512 battles 545 s = 2 chunks of 256; + 1 min a drill to start (imports, maps, weights)")
TOL = {"B1": 0.05, "quiet": 0.02, "aim": 0.10}
ORDERS_CAP = 1.5 * 1.0
_TOL_WHY = ("METHOD step 6: prior drills - 'does' not lower than the accepted by more than 0.05, false moves not higher "
            "by more than 0.02; the move reaches its aim not lower than the teacher's - 0.10; orders <= 1.5 x the human's "
            "(1.0 a unit-minute: exam_o1.ACCEPT human_orders_min, game/vision/README.md)")


# ------------------------------------------------------------------ sets and seeds
def t1_counts(n):
    """{recipe: count} of set S's recipe mix (net/config/stage0.json drills.attack.sets.S) scaled to n scenes by the
    largest remainder (sum n; a recipe may get 0 at a small n)."""
    from net.tactic import scenes as SN
    full = SN.c()["sets"]["S"]["counts"]
    tot = sum(full.values())
    raw = {r: n * m / tot for r, m in full.items()}
    out = {r: int(math.floor(v)) for r, v in raw.items()}
    for r in sorted(raw, key=lambda r: (out[r] - raw[r], list(full).index(r)))[:n - sum(out.values())]:
        out[r] += 1
    return out


def t1_specs(n):
    from net.tactic import scenes as SN
    full = list(SN.c()["sets"]["S"]["counts"])
    s0 = SEEDS["T0-1"]["scene"]
    out = []
    for rec, m in t1_counts(n).items():
        j = full.index(rec)
        out += [SN.make(rec, s0 + 10000 * j + i) for i in range(m)]
    return out


def o1_specs(n):
    from net.operative import scenes_fair as SF
    from net.operative import scenes_o1 as SO
    from net.tactic import zfit
    zfit._CACHE.clear()          # (as exam_o1.sets_of: a set's starts from its own seeds only, not the build order)
    s0 = SEEDS["O0-1"]["scene"]
    sp =[SO.scene(s0 + i) for i in range(n)] + [SO.scene(s0 + i, twin="come" if i % 2 == 0 else "role")
                                                 for i in range(n)]
    return [SF.fair(x) for x in sp]


def s1_specs(n):
    from net.strategist import s1_scenes as SS
    return SS.scenes_at(SEEDS["S0-1"]["scene"], n, "taught")


def registry_entries():
    """The re-exam's ranges for seeds.registry() (split 'reexam'), at N_MAX."""
    from net.drill import seeds as SD
    from net.tactic import scenes as SN
    out = []
    s0 = SEEDS["T0-1"]["scene"]
    full = list(SN.c()["sets"]["S"]["counts"])
    for rec, m in t1_counts(N_MAX["T0-1"]).items():
        lo = s0 + 10000 * full.index(rec)
        out.append(SD.E(f"T0-1 reexam {rec}", "t0_1", "reexam", "tactic.make", lo, lo + int(m), key=rec,
                        why="net/drill/reexam.py t1_specs: set S's mix scaled to N_MAX"))
        out += SD._twins(f"T0-1 reexam {rec}", "reexam", rec, lo, int(m))
    out.append(SD.E("T0-1 reexam torch chunks", "t0_1", "reexam", "torch.chunk", SEEDS["T0-1"]["torch"],
                    SEEDS["T0-1"]["torch"] + 100, why="reexam.py: chunk k under torch seed + k"))
    from net.operative import scenes_o1 as SO
    subs = int(SO.cs()["substitutes"])
    s0 = SEEDS["O0-1"]["scene"]
    out.append(SD.E("O0-1 reexam", "o0_1", "reexam", "o1.scene", s0, s0 + N_MAX["O0-1"], subs=subs,
                    why="reexam.py o1_specs (twins share the cause scene's seed)"))
    out.append(SD.E("O0-1 reexam torch chunks", "o0_1", "reexam", "torch.chunk", SEEDS["O0-1"]["torch"],
                    SEEDS["O0-1"]["torch"] + 100, why="reexam.py"))
    s0 = SEEDS["S0-1"]["scene"]
    out.append(SD.E("S0-1 reexam", "s0_1", "reexam", "s1.scene", s0, s0 + N_MAX["S0-1"],
                    why="reexam.py s1_specs (s1_scenes.scenes_at, taught's ground pool)"))
    out.append(SD.E("S0-1 reexam torch chunks", "s0_1", "reexam", "torch.chunk", SEEDS["S0-1"]["torch"],
                    SEEDS["S0-1"]["torch"] + 100, why="reexam.py"))
    return out


def check_starts(drill, specs, tseeds):
    """Every spec's identity and family and every torch seed inside the registry file's 'reexam' ranges of the card."""
    from net import ROOT
    from net.drill import seeds as SD
    E = SD.load(ROOT / "net" / "drill" / "registry.json")
    card = {"T0-1": "t0_1", "O0-1": "o0_1", "S0-1": "s0_1"}[drill]
    starts = []
    for s in specs:
        for g, k, v in (SD.identity(s), SD.family(s)) if drill == "T0-1" else (SD.identity(s),):
            starts.append((g, k, int(v)))
    starts += [("torch.chunk", None, int(t)) for t in tseeds]
    return SD.assert_starts(E, [card], "reexam", starts)


# ------------------------------------------------------------------ the lines
def _share(k, n):
    """{p, two_errors, k, n}: the share and its two errors from (k + 1) / (n + 2) (exam_o1.share)."""
    if not n:
        return {"p": None, "two_errors": None, "k": k, "n": 0}
    q = (k + 1) / (n + 2)
    return {"p": k / n, "two_errors": 2 * math.sqrt(q * (1 - q) / n), "k": int(k), "n": int(n)}


def _of(c):
    """_share of a learn.share cell {share, n}."""
    n = int(c.get("n") or 0)
    return _share(int(round((c.get("share") or 0.0) * n)), n)


def _ratio_by_battle(f, l):
    """{p, two_errors, k, n, battles}: sum f / sum l with the error by battle (the ratio's cluster error)."""
    pairs = [(a, b) for a, b in zip(f, l) if b]
    k, n = sum(a for a, _ in pairs), sum(b for _, b in pairs)
    m = len(pairs)
    if not n:
        return {"p": None, "two_errors": None, "k": 0, "n": 0, "battles": 0}
    p = k / n
    v = sum((a - p * b) ** 2 for a, b in pairs) * m / max(m - 1, 1) / n ** 2
    q = (k + 1) / (n + 2)
    return {"p": p, "two_errors": 2 * max(math.sqrt(v), math.sqrt(q * (1 - q) / n)), "k": int(k), "n": int(n),
            "battles": m}


def _mean(xs):
    xs = [x for x in xs if x is not None and isinstance(x, (int, float)) and math.isfinite(x)]
    n = len(xs)
    if not n:
        return {"p": None, "two_errors": None, "n": 0}
    mu = sum(xs) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return {"p": mu, "two_errors": 2 * sd / math.sqrt(n), "n": n}


def judge(line, now, acc):
    """(status, limit) of a line: pass / noise / fail / NOT JUDGED (module doc)."""
    if now is None or now.get("p") is None:
        return "NOT JUDGED", None
    p, e = now["p"], now["two_errors"] or 0.0
    if line == "orders":
        lim, up = ORDERS_CAP, False
    elif acc is None:
        return "NOT JUDGED", None
    elif line == "quiet":
        lim, up = acc + TOL["quiet"], False
    else:
        lim, up = acc - TOL[line], True
    if (p >= lim) if up else (p <= lim):
        return "pass", lim
    return ("noise" if ((p + e >= lim) if up else (p - e <= lim)) else "fail"), lim


# ------------------------------------------------------------------ the drills
def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_t01(row, n, device, chunk, ckpt=None):
    from types import SimpleNamespace
    from sim.params import load
    from net.tactic import ckpt as TK
    from net.tactic import exam_pin as EP
    from net.tactic import imitate as IM
    from net.tactic import learn as LN
    P = load()
    net, _, _ = TK.restore(ckpt or row["weights"]["path"], device)
    net.eval()
    specs = t1_specs(n)
    ch = min(chunk, len(specs))
    seeds = LN._seeds(SEEDS["T0-1"]["torch"], len(specs), ch)
    check_starts("T0-1", specs, seeds)
    files, mN, _ = IM.read(net, specs, True, seeds, ch, SimpleNamespace(device=device, sim=""), P, "scene",
                           paths="N", watch=EP.HoleWatch)
    t = IM.marks(mN, None, "S")
    pb = t["per_battle"]
    a = row["after"]
    return {"lines": {"B1": _ratio_by_battle(pb["followed"], pb["labels"]),
                      "quiet": _of(t["false_N"]["tick"]), "aim": _of(t["M4"]),
                      "orders": _mean(files["N"]["per_battle"]["game_orders_min"])},
            "accepted": {"B1": a["followed"]["own_path"]["S"], "quiet": a["false_attack_tick"]["S"],
                         "aim": a["target_is_teachers"]["S"], "orders": a["orders_unit_min"]},
            "reads": {"B1": "followed_N pooled (imitate.marks), error by battle", "quiet": "false_N a tick",
                      "aim": "M4: target = the teacher's", "orders": "game_orders_min of path N"},
            "printed": {"nocause_false": t["nocause_false"], "uncaused": t["uncaused"],
                        "recipes": t1_counts(n)},
            "battles": len(specs)}


def run_o01(row, n, device, chunk, ckpt=None):
    from net.operative import ckpt as CK
    from net.operative import exam_o1 as EX
    from net.operative import heads as HD
    from net.operative import lines as LL
    specs = o1_specs(n)
    ch = min(chunk, len(specs))
    seeds = [SEEDS["O0-1"]["torch"] + k for k in range((len(specs) + ch - 1) // ch)]
    check_starts("O0-1", specs, seeds)
    net = CK.restore_o1(ckpt or row["weights"]["path"], device)[0]
    d = EX.play("NET", specs, seeds, ch, device, tactic="TEACH", make_player=lambda ci, ii: HD.NetPlayer(net),
                visible=True)
    R = EX.reads(d, int(LL.conf()["limit_s"]))
    cause = [b for b, s in enumerate(specs) if s["o1"]["cause"]]
    nocause = [b for b, s in enumerate(specs) if not s["o1"]["cause"]]
    hits, cnt = EX._b1_own(d, cause)
    a = row["after"]
    return {"lines": {"B1": _share(sum(hits), len(hits)),
                      "quiet": _share(sum(R["changed"][b] for b in nocause), len(nocause)),
                      "aim": _share(sum(R["contact"][b] for b in cause), len(cause)),
                      "orders": _mean(R["orders"])},
            "accepted": {"B1": row["held_out"]["taught"][0], "quiet": a["quiet"]["twins_moved"]["NET"][0],
                         "aim": a["aim_contact"]["NET"][0], "orders": a["orders_unit_min"]["taught"]},
            "reads": {"B1": "exam_o1._b1_own (first LS within a tick of the teacher's label on the path)",
                      "quiet": "the order changed on the come / role twins", "aim": "contact before the timer, causes",
                      "orders": "orders_with_lord_min"},
            "printed": {"B1_misses": cnt}, "battles": len(specs)}


def run_s01(row, n, device, chunk, ckpt=None):
    from net.strategist import exam_s1 as EX
    from net.strategist import s1_net as SN
    specs = s1_specs(n)
    ch = min(chunk, len(specs))
    tseed = SEEDS["S0-1"]["torch"]
    check_starts("S0-1", specs, [tseed + k for k in range((len(specs) + ch - 1) // ch)])
    net, _ = SN.restore(ckpt or row["weights"]["path"], device)
    k = [int(x) for x in SN.choose(net, SN.observe(specs, device), device)]
    sr = EX.spec_rows(specs)
    cause = [i for i, r in enumerate(sr) if r["cause"]]
    twins = [i for i, r in enumerate(sr) if not r["cause"]]
    rows = EX.play(specs, "NET", tseed, device, ch, k)
    by = {r["key"]: r for r in rows}
    up, nu = EX._units_above(by, [sr[i]["key"] for i in cause])
    a = row["after"]
    return {"lines": {"B1": _share(sum(int(k[i]) == sr[i]["teach"] for i in cause), len(cause)),
                      "quiet": _share(sum(int(k[i]) != 0 for i in twins), len(twins)),
                      "aim": _share(up, nu), "orders": _mean([r["orders"] for r in rows])},
            "accepted": {"B1": row["held_out"]["taught"][0], "quiet": a["quiet_twins_moved"]["NET"][0],
                         "aim": a["aim_higher_at_contact"]["NET"][0], "orders": None},
            "accepted_text": {"orders": a.get("orders_unit_min")},
            "reads": {"B1": "the network's crest = the teacher's (chosen before the battle: no play)",
                      "quiet": "k != 0 on the twins", "aim": "units meeting their first enemy higher, causes",
                      "orders": "the NET rows' orders"},
            "battles": len(specs)}


RUN = {"T0-1": run_t01, "O0-1": run_o01, "S0-1": run_s01}


class Aura:
    """The aura exposure (module doc): lower.Tacticians._tick_player wrapped for the run - after each tick, my able
    units of its rows counted with the raw's e_lord_near / e_hold_the_line on (the level's own raw: the lord's row
    hidden, the effects observed before)."""

    def __init__(self):
        self.n = {"decisions": 0, "lord_near": 0, "hold_the_line": 0}

    def __enter__(self):
        from net.operative import lower as LW
        from net.tactic import raw as TR
        self.orig = orig = LW.Tacticians._tick_player
        n = self.n

        def wrapped(tac, *a, **k):
            before = tac.view
            out = orig(tac, *a, **k)
            v = (tac.view or {}).get("view") if tac.view is not before else None   # (a lord group's tick: not ours)
            if v is not None and v.raw is not None:
                fx = v.fx
                my = fx.my.clamp(min=0)
                ok = fx.able & (fx.my >= 0)
                for key, col in (("lord_near", "e_lord_near"), ("hold_the_line", "e_hold_the_line")):
                    on = v.raw[..., TR.I[col]].gather(1, my) > 0
                    n[key] += int((on & ok).sum())
                n["decisions"] += int(ok.sum())
            return out
        LW.Tacticians._tick_player = wrapped
        return self

    def __exit__(self, *e):
        from net.operative import lower as LW
        LW.Tacticians._tick_player = self.orig

    def shares(self):
        d = self.n["decisions"]
        return {"decisions": d, **{k: (round(self.n[k] / d, 4) if d else None) for k in ("lord_near", "hold_the_line")}}


def lords(mode, buttons):
    """stage0.json's lord keys for the run (module doc --lord-mode); {mode, buttons} for the report."""
    from net.stage0 import cfg as scfg
    if mode == "off":
        return {"mode": "off"}
    scfg()["lord_script"] = mode
    scfg()["enemy_lord_fights"] = True
    scfg()["lord_buttons"] = buttons == "on"
    return {"mode": mode, "buttons": buttons}


# ------------------------------------------------------------------ the command
def accepted_rows():
    from net import ROOT
    J = json.loads((ROOT / "net" / "drills.json").read_text(encoding="utf-8"))
    return {r["id"]: r for r in J["drills"] if str(r.get("passed", "")).strip().lower().startswith("да")}


def sim_set(pairs):
    """Apply 'a.b=value' (value JSON) to the cached sim.params.load()'s sim dict; {path: (old, new)}."""
    from sim.params import load
    P = load()
    out = {}
    for s in pairs:
        path, _, v = s.partition("=")
        ks = path.split(".")
        d = P.sim
        for k in ks[:-1]:
            d = d[k]
        if ks[-1] not in d:
            raise SystemExit(f"--sim-set {path}: no such key in sim.json")
        old, d[ks[-1]] = d[ks[-1]], json.loads(v)
        out[path] = [old, d[ks[-1]]]
    return out


def estimate(drills, n_of, chunk_of, device):
    m = 0.0
    for dr in drills:
        ch = chunk_of[dr]
        b = n_of[dr] * (1 if dr == "T0-1" else 2)
        m += 1.0 + MIN_CHUNK[dr] * math.ceil(b / ch)
    return round(m, 1)


def table_lines(res):
    L = [f"{'drill':6} {'line':7} {'now':>8} {'2 err':>7} {'n':>6} {'accepted':>9} {'limit':>7}  status"]
    for dr, r in res.items():
        if "error" in r:
            L.append(f"{dr:6} ERROR {r['error']}")
            continue
        for ln in ("B1", "quiet", "aim", "orders"):
            x = r["lines"][ln]
            acc = r["accepted"].get(ln)
            st, lim = r["status"][ln]
            f = lambda v, w=8, d=3: (f"{v:{w}.{d}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}")  # noqa: E731
            acc_s = f(acc, 9) if acc is not None else f"{str((r.get('accepted_text') or {}).get(ln, '-'))[:9]:>9}"
            L.append(f"{dr:6} {ln:7} {f(x.get('p'), 8, 4)} {f(x.get('two_errors'), 7, 4)} {x.get('n') or 0:6d} "
                     f"{acc_s} {f(lim, 7)}  {st}")
    L.append("")
    for dr, r in res.items():
        L.append(f"{dr}: {r.get('verdict', 'ERROR')}  ({r.get('battles', 0)} battles, {r.get('min', '?')} min)")
        if r.get("aura"):
            au = r["aura"]
            L.append(f"{dr}: aura exposure - lord_near {au['lord_near']}, hold_the_line {au['hold_the_line']} of "
                     f"{au['decisions']} infantry decisions")
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drills", default=",".join(DRILLS))
    ap.add_argument("--n", type=int, default=0, help="every drill's n (0: N; T0-1 scenes, O0-1 / S0-1 cause scenes)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="", help="default tmp/reexam/<date-time>")
    ap.add_argument("--sim-set", action="append", default=[], help="sim.json path=value in this process only")
    ap.add_argument("--lords", default=None, choices=("off", "on"),
                    help="on: = --lord-mode t2 --buttons on (the old switch)")
    ap.add_argument("--lord-mode", default=None, choices=("off", "post", "t2", "group"),
                    help="stage0.json lord_script in this process only (module doc)")
    ap.add_argument("--buttons", default=None, choices=("on", "off"), help="the lord's buttons (default on)")
    ap.add_argument("--ckpt", default="", help="a candidate's weights instead of the journal row's (one drill)")
    a = ap.parse_args(argv)
    if a.lords is not None:
        if a.lord_mode is not None or a.buttons is not None:
            raise SystemExit("--lords is the alias of --lord-mode t2 --buttons on: not with them")
        a.lord_mode, a.buttons = ("t2", "on") if a.lords == "on" else ("off", "on")
    a.lord_mode, a.buttons = a.lord_mode or "off", a.buttons or "on"
    import torch
    from net import ROOT
    from net.tactic import exam_pin as EP
    D = Path(a.out or ROOT / "tmp" / "reexam" / time.strftime("%Y%m%d-%H%M%S"))
    D.mkdir(parents=True, exist_ok=True)
    changed = sim_set(a.sim_set)
    lk = lords(a.lord_mode, a.buttons)
    rows = accepted_rows()
    want = [x for x in a.drills.split(",") if x]
    if a.lord_mode == "group" and any(dr != "T0-1" for dr in want):
        # (lord mode group commands the lord through the tactic level only: the operative's O0-1 side and the
        # strategist's S0-1 read the level as one group a battle - not built for his group)
        raise SystemExit("--lord-mode group: T0-1 only (--drills T0-1); O0-1 / S0-1 are not built for the lord's group")
    ck_sha = None
    if a.ckpt:
        if len(want) != 1:
            raise SystemExit("--ckpt: one drill (--drills X)")
        ck_sha = _sha(a.ckpt)
        print(json.dumps({"ckpt": a.ckpt, "sha256": ck_sha}), flush=True)
    not_covered = sorted(set(rows) - set(RUN))
    n_of = {dr: (a.n or N[dr]) for dr in want}
    for dr in want:
        if n_of[dr] > N_MAX[dr]:
            raise SystemExit(f"{dr}: n {n_of[dr]} > the registered {N_MAX[dr]}")
    chunk_of = {dr: CHUNK[dr] for dr in want}
    est = estimate(want, n_of, chunk_of, a.device)
    head = {"start": time.strftime("%Y-%m-%d %H:%M"), "device": a.device, "sim_key": EP.sim_key(), "sim_set": changed,
            "lords": lk, "ckpt": {"path": a.ckpt, "sha256": ck_sha} if a.ckpt else None,
            "n": n_of, "chunk": chunk_of, "gpu_minutes_estimate": est, "estimate_why": _MIN_WHY, "n_why": _N_WHY,
            "tolerances": {**TOL, "orders_cap": ORDERS_CAP}, "tol_why": _TOL_WHY, "seeds": SEEDS,
            "accepted_rows": sorted(rows), "not_covered": not_covered}
    print(json.dumps({k: head[k] for k in ("sim_key", "sim_set", "lords", "n", "gpu_minutes_estimate", "not_covered")}),
          flush=True)
    res = {}
    for dr in want:
        if dr not in rows:
            res[dr] = {"error": "not an accepted row of net/drills.json"}
            continue
        row = rows[dr]
        sha = ck_sha if a.ckpt else _sha(ROOT / row["weights"]["path"])
        if not a.ckpt and sha != row["weights"]["sha256"]:
            res[dr] = {"error": f"weights {row['weights']['path']} sha {sha[:16]} != the row's"}
            continue
        t0 = time.time()
        torch.manual_seed(0)
        with Aura() as au:
            r = RUN[dr](row, n_of[dr], a.device, chunk_of[dr], ckpt=a.ckpt or None)
        if a.lord_mode != "off":
            r["aura"] = au.shares()
        r["status"] = {ln: judge(ln, r["lines"][ln], r["accepted"].get(ln)) for ln in r["lines"]}
        sts = [s for s, _ in r["status"].values()]
        r["verdict"] = ("FAIL" if "fail" in sts else "NOT JUDGED" if "NOT JUDGED" in sts else
                        "INSIDE NOISE" if "noise" in sts else "PASS")
        r["min"] = round((time.time() - t0) / 60, 1)
        r["weights_sha256"] = sha
        res[dr] = r
        print(json.dumps({"drill": dr, "verdict": r["verdict"], "min": r["min"], "aura": r.get("aura")}), flush=True)
        (D / "reexam.json").write_text(json.dumps({**head, "drills": res}, indent=1, default=str), encoding="utf-8")
        if a.device.startswith("cuda"):
            torch.cuda.empty_cache()
    out = {**head, "drills": res, "end": time.strftime("%Y-%m-%d %H:%M")}
    (D / "reexam.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    L = table_lines(res)
    if not_covered:
        L.append(f"accepted rows without a re-exam: {', '.join(not_covered)}")
    L.append(f"GPU minutes estimate {est}; written {D / 'reexam.json'}")
    (D / "table.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)
    v = [r.get("verdict") for r in res.values()]
    return 0 if all(x == "PASS" for x in v) else (1 if any(x in ("FAIL", "NOT JUDGED", None) for x in v) else 2)


if __name__ == "__main__":
    raise SystemExit(main())

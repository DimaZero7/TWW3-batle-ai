"""Drill S0-1 'the line on the crest' - the teacher's exam before training (METHOD.md recipe step 4;
tmp/drill_s1/design_opus.md section 6 as amended by design_merged.md section 5 and Astra's s1_step2_review,
tmp/drill_s1/brief_C3.md). The arms stand my units on a line (s1_scenes.line) and the stack below plays the battle as
it is: the operative is out (our mission hold: O0-1 is not trained yet, its default keeps the zone); the tactic level is
T0-1's teacher (net/tactic/exam_pin.build_under('TEACH')) and the unit script (stage0/battle.TacticSide): the places ->
battle.play_tactic(battles=) -> the tactic's zone fitted on the places (tactic/zfit). NOT modelled here (said so, Astra
4c): O0-1 in the stack; the stage's 30 % chain opponents in G (scripts only: play_tactic's general battles); the role
twins' realism (mission attack, but no attacking operative - their TEACH = KEEP = DEFAULT by construction).

    DEFAULT  the centre line (KEEP)          TEACH  the teacher's line (crest.teach)
    DUMB     the foot: the legal line of the lowest band (cause scenes; DEFAULT on the twins)
    PLACE    the cause scene's teacher line on its flat twin (the set 'flat'; report: the height apart from the move)
    NET      a strategist checkpoint (--net): the network's choice (the exam after)
EVERY arm plays the WHOLE set in the same chunks and lanes with the same torch chunk seeds (an arm without its own line
on a battle stands DEFAULT there): a pair differs by the line only (Astra 1). The arms' rows are joined by the battle's
key (s1_scenes.identity), never by index. The flat set's chunks are half filled with carriers (CARRIED: a batch of
flat battles alone traces a graph that Triton fails to compile).
Sets (s1_scenes; seeds in tmp/drill_s1/c/NEEDS_A.md): taught (n cause + their n twins), flat (the flat twin of each of
taught's n cause scenes: PLACE and DEFAULT), held:<axis> (n cause + n twins each of the seven held axes), G (n general
battles of the stage, att 0).
THE FIRST OPPONENT (Astra 2): a unit's first contact is the first second at which it and an enemy touch in EITHER
direction (hit_sec or its transpose, sim/battle); several enemies in that second -> the nearest centre; per unit kept:
(slot, the enemy's slot, the second, dh = my height - the enemy's); a unit that never touched stays as 'no contact'
(dh None). LORDS: out of both the contact line and the height line on both sides (stage0/watch's mine / foe: side less
the lord) - the battle's contact is my infantry's first such contact before the timer.
Per battle: contact, the units' rows, dh_c (the mean dh of the touched units), the line's band_slot_min and every
front-row slot's band (crest.slot_bands on the arm's line), hp_trade at the end (stage0/watch.hp_trade, lords out;
with lords beside), the gold trade, orders to the game a unit-minute.

LINES (design_opus 6 'Teacher's exam before training' + design_merged 5; written before any run - do not move; the
minimum n of each is its ACTUAL n - pairs present in both arms - and below it the line is NOT JUDGED, which fails the
table; the table exits 1 on any fail or NOT JUDGED):
    exec     taught cause: dh_c TEACH - DEFAULT lower bound >= +0.8 m; >= 0.75 of TEACH's units (non-contacts in the
             denominator) dh > 0; TEACH contact >= 0.9                                    (n >= 256)
    sign     taught cause: hp_trade TEACH - DUMB lower bound > 0                           (n >= 256)
    quiet    taught twins: TEACH's placement changed <= 3 %; hp_trade TEACH - DEFAULT lower bound >= -0.005 (n >= 256;
             the held sets' twins reported beside)
    held     each of the seven held axes: binds 100 %; dh_c TEACH - DEFAULT lower bound > 0 on its cause scenes (n >= 64)
    noharm   G: hp_trade TEACH - DEFAULT lower bound >= -0.005 (n >= 512); the flat field's battles bit for bit (n >= 1)
    orders   taught (causes and twins): TEACH's orders a unit-minute <= DEFAULT's + 0.1 and <= 1.2  (n >= 512)
    freeze   taught cause: hp_trade TEACH - DEFAULT lower bound < 0 -> FAIL (the design's FREEZE)   (n >= 256)
    place    flat: report; |PLACE - DEFAULT| hp_trade beyond two errors -> STOP (fails the table: not the height) (n >= 256)
    bins     the cause scenes of taught by Delta band (0.05-0.08, 0.08-0.12, 0.12-0.2, >= 0.2): TEACH - DEFAULT hp_trade;
             every bin n >= 30; the lowest bin with a negative mean fails the teacher
    value    the value gate (4.0 errors, 0.02): taught cause hp_trade TEACH - DEFAULT >= 4 standard errors and >= 0.02
             -> 'half the gap' asked of the network, else 'not worse than the teacher' (report; n >= 256)
    strata   (report; tmp/astra/KEY.md s1_step2_review AMENDED): the cause scenes of taught and held by the TEACH line's
             band_slot_min < 0.05 / 0.05-0.08 / >= 0.08, captured apart from synthetic: n, the weakest slot's dh (its
             no-contacts), contact, paired trade TEACH - DEFAULT. A stratum with n >= 30 whose weakest-slot dh mean <= 0
             or whose trade mean < 0 -> 'veto band_slot_min >= 0.08' for the teacher (only the touched scenes replayed)
Two errors = 2 sd / sqrt(n), paired by battle.
    python -m net.strategist.exam_s1 run --out DIR [--sets taught,flat,held,G] [--arms ...] [--n 256] [--n-g 512]
                                        [--net CKPT] [--device cuda] [--chunk 256]
    python -m net.strategist.exam_s1 table --out DIR [--arm TEACH]          (exit 1: a line failed or not judged)
    python -m net.strategist.exam_s1 preflight [--n 16]      (every start of the sets built; counts; binds)
THE NEW RECIPE (METHOD steps 4 and 6; the lines above are the OLD ones, kept printed by `table`): tmp/drill_s1/c2/
LINES_s1.md. `run` takes --specs-cache DIR (a set's specs built once, --build-only before the GPU) and --bridge FILE
(the key bridge: a set whose starts or full specs differ from the proven ones is refused); rows played from 23.11 add
hp_trade_300 (the + 300 s window) and, for NET, k (the network's pointer).
    python -m net.strategist.exam_s1 teacher --out tmp/drill_s1/c/exam --bridge B [--verdict F]   (exit 0/1/2/3)
    python -m net.strategist.exam_s1 accept --out E --run T --bridge B --specs-cache E/specs     (exit 0/1/2/3)
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

SEEDS = {"taught": 96_000_000, "held": 96_100_000, "G": 97_000_000}
HELD_STRIDE = 10_000
TSEED = {"taught": 98_000_000, "held": 98_100_000, "G": 98_200_000, "flat": 98_300_000}
_SEEDS_WHY = "the exam's fresh ranges of generator s1.scene / s1.general and torch.chunk (tmp/drill_s1/c/NEEDS_A.md)"
MIN_N = {"exec": 256, "sign": 256, "quiet": 256, "held": 64, "noharm": 512, "flat_bit": 1, "orders": 512,
         "freeze": 256, "place": 256, "bins": 30, "value": 256, "strata": 30}
BINS = ((0.05, 0.08), (0.08, 0.12), (0.12, 0.2), (0.2, 9.0))
STRATA = ((-9.0, 0.05), (0.05, 0.08), (0.08, 9.0))
LINES = {"exec_dh_lo": 0.8, "exec_units_above": 0.75, "exec_contact": 0.9, "quiet_changed": 0.03, "quiet_lo": -0.005,
         "noharm_lo": -0.005, "orders_over": 0.1, "orders_max": 1.2, "value_errors": 4.0, "value_abs": 0.02}
WINDOW_S = 300
_WINDOW_WHY = ("METHOD step 6: the trade read at + 300 s and at the battle's end (600 s here); rows played from 23.11 carry "
               "hp_trade_300 - the exam's stored arms (22-23.11) only the end")
NEW_FIELDS = ("hp_trade_300", "k")
CARRIED = {"flat": "taught"}
_CARRIED_WHY = ("the flat set alone is a batch without any height map (every battle on 'flat': sim/terrain.set_map adds no "
                "'terr'), so the compiled step traces another graph whose fused persistent reduction (~120 pointer "
                "arguments, triton_per_fused_..._59) fails in Triton's make_ttgir on this GPU (tmp/drill_s1/c/"
                "gpu_s1.try3.log, 22.11). Each of its chunks is half filled with carriers - the first battles of the "
                "taught set, standing DEFAULT in every arm - so the batch has its maps and runs the graph the taught "
                "set compiled; a flat battle is the same flat battle as anywhere (terr -1 sees height 0 and the layers' "
                "empty values), its chunk, lane and chunk seed the same in every arm; the carriers' rows are dropped")
ARMS_OF = {"taught": ("DEFAULT", "TEACH", "DUMB"), "flat": ("DEFAULT", "PLACE"), "held": ("DEFAULT", "TEACH"),
           "G": ("DEFAULT", "TEACH")}
KEY_TREES = ("net/strategist", "net/tactic", "net/unit", "net/stage0", "net/operative/heads.py",
             "net/operative/gframe.py", "net/matchup/duel.py", "net/config/strategist.json", "net/config/stage0.json",
             "net/config/tactic.json", "net/config/unit.json")
_KEY_WHY = ("Astra s1_step2_review 7c: a result's key carries the lower levels' policies (net/tactic: T0-1's teacher, "
            "net/unit: the unit script, net/stage0: the battle and its watch), the input builder (net/strategist/s1_net, "
            "net/operative/heads) and the simulator (exam_pin.sim_key)")


def set_specs(name, n, n_g=None):
    from net.strategist import s1_scenes as SS
    if name == "taught":
        return SS.scenes_at(SEEDS["taught"], n, "taught")
    if name == "flat":
        return [SS.scene(SEEDS["taught"] + i, "taught", twin="flat") for i in range(n)]
    if name == "G":
        return [SS.general(SEEDS["G"] + i) for i in range(n if n_g is None else n_g)]
    if name.startswith("held:"):
        ax = name.split(":", 1)[1]
        j = SS.HELD.index(ax)
        return SS.scenes_at(SEEDS["held"] + HELD_STRIDE * j, n, "held", held=ax)
    raise KeyError(name)


def set_names(sets):
    from net.strategist import s1_scenes as SS
    out = []
    for s in sets:
        out += [f"held:{a}" for a in SS.HELD] if s == "held" else [s]
    return out


def _kind(name):
    return "held" if name.startswith("held:") else name


def _tseed(name):
    from net.strategist import s1_scenes as SS
    if name.startswith("held:"):
        return TSEED["held"] + 1000 * SS.HELD.index(name.split(":", 1)[1])
    return TSEED[name]


def bkey(spec):
    """The battle's key the arms are joined by."""
    from net.strategist import s1_scenes as SS
    return "|".join(str(x) for x in SS.identity(spec))


# ------------------------------------------------------------------ the play
def battles(specs):
    from net.stage0 import battle as BA
    from net.stage0 import cfg as scfg
    from net.strategist import crest as CR
    out = []
    for s in specs:
        mine = [BA._unit(k, *p) for k, p in zip(s["mine"], s["places_mine"])]
        theirs = [BA._unit(k, *p) for k, p in zip(s["theirs"], s["places_theirs"])]
        l1 = BA._unit(scfg()["lords"][s["fac"][1]], *CR.lord_place(s["places_mine"], s["places_mine"][0][2],
                                                                   s["mine_zone"]), general=True)
        l2 = BA._unit(scfg()["lords"][s["fac"][2]], *CR.lord_place(s["places_theirs"], s["enemy_zone"]["b"],
                                                                   s["enemy_zone"]), general=True)
        out.append({1: [l1] + mine, 2: [l2] + theirs, "factions": dict(s["factions"])})
    return out


def first_contacts(hs, mine, foe, x, z, done_mask, live):
    """(new [B, N] bool, j [B, N] long): my units (mine) touching an enemy (foe) for the first time this second in
    either direction (hs [B, N, N] hit_sec or its transpose > 0), the nearest centre among the enemies touched (module
    doc). done_mask [B, N]: units already recorded; live [B]: battles running."""
    import torch
    touch = (hs > 0) | (hs.transpose(1, 2) > 0)
    pair = touch & mine[:, :, None] & foe[:, None, :]
    new = pair.any(2) & ~done_mask & live[:, None]
    d = torch.hypot(x[:, :, None] - x[:, None, :], z[:, :, None] - z[:, None, :])
    d = torch.where(pair, d, torch.full_like(d, float("inf")))
    return new, d.argmin(2)


def watch_cls():
    import torch
    from net.stage0 import watch as W
    from sim import terrain

    class CrestWatch(W.Watch):
        """Watch + each of my units' first contact by the module doc's rule: dh_u, opp_u, sec_u [B, N]."""

        def __call__(self, sec, st):
            if self.st is None:
                self._init(st)
                self.dh_u = torch.full((self.B, self.N), float("nan"), device=self.dev)
                self.opp_u = torch.full((self.B, self.N), -1, dtype=torch.long, device=self.dev)
                self.sec_u = torch.full((self.B, self.N), float("nan"), device=self.dev)
                self.c_done = torch.zeros((self.B, self.N), dtype=torch.bool, device=self.dev)
            u = st.u
            if sec + 1 == WINDOW_S:            # (the window's reading, METHOD step 6: + 300 s; a read, nothing changed)
                self.hp_300 = W.hp_trade(u, self.lord).clone()
            hs = u["hit_sec"].view(self.B, self.N, self.N)
            new, j = first_contacts(hs, self.mine, self.foe, u["x"], u["z"], self.c_done, ~st.done)
            if bool(new.any()):
                h = terrain.height(u, u["x"], u["z"])
                self.dh_u = torch.where(new, h - h.gather(1, j), self.dh_u)
                self.opp_u = torch.where(new, j, self.opp_u)
                self.sec_u = torch.where(new, torch.full_like(self.sec_u, float(sec + 1)), self.sec_u)
                self.c_done |= new
            super().__call__(sec, st)
    return CrestWatch


def arm_line(spec, arm, k_net=None):
    """The arm's line on this battle: its own where it has one, else DEFAULT's (module doc)."""
    if arm == "DUMB" and not spec["s1"].get("cause"):
        return "DEFAULT"
    if arm == "PLACE" and spec["s1"].get("twin_made") != "flat":
        return "DEFAULT"
    return arm


def arm_specs(specs, arm, k_net=None):
    import copy
    from net.strategist import crest as CR
    from net.strategist import s1_ground as SG
    from net.strategist import s1_scenes as SS
    from net.tactic import zfit
    out = []
    for i, s in enumerate(specs):
        sp = copy.deepcopy(s)
        a = arm_line(sp, arm)
        kn = None if k_net is None else k_net[i]
        ln = SS.line(sp, a, kn)
        sp["places_mine"] = SS.places(sp, a, kn)
        assert CR.layout_ok(SG.ground(sp["map"]), sp["places_mine"]), f"{bkey(sp)} {arm}: a unit on blocked ground"
        zn = zfit.zone_for([p[:2] for p in sp["places_mine"]], float(ln["b"]))
        sp["zone"] = None if zn is None else {k: (float(v) if isinstance(v, float) else v) for k, v in zn.items()}
        sp["line"] = {k: round(float(ln[k]), 3) for k in ("x", "z", "b", "band")}
        sp["arm_line"] = a
        sp["slot_bands"] = [round(v, 4) for v in CR.slot_bands(SG.ground(sp["map"]), ln, len(sp["mine"]))]
        out.append(sp)
    return out


def play(specs, arm, tseed, device="cpu", chunk=256, k_net=None):
    """[row] per spec (module doc) of the arm."""
    import torch
    from net.stage0 import battle as BA
    from net.stage0 import watch as W
    from net.strategist import s1_ground as SG
    from net.tactic import exam_pin as EP
    from sim.params import load
    P = load()
    sp = arm_specs(specs, arm, k_net)
    SG.register([s["map"] for s in sp])
    ends = {}

    def on_chunk(ii, watch, side):
        u = watch.st.u
        hp = W.hp_trade(u, watch.lord)
        keep = u["side"] > 0
        c = u["cost"] * keep
        lost = c * (1 - u["hp_abs"] / u["hp0"].clamp(min=1e-6)).clamp(0, 1)
        lo = [(lost * (u["side"] == s_)).sum(1) for s_ in (1, 2)]
        bud = torch.stack([(c * (u["side"] == s_)).sum(1) for s_ in (1, 2)], 1).mean(1).clamp(min=1.0)
        hpl = (lo[1] - lo[0]) / bud
        for r, i in enumerate(ii):
            mi = watch.mine[r].nonzero().flatten().tolist()
            fi = watch.foe[r].nonzero().flatten().tolist()
            units = []
            for slot, q in enumerate(mi):
                d = float(watch.dh_u[r, q])
                o = int(watch.opp_u[r, q])
                units.append({"slot": slot, "dh": None if d != d else round(d, 3),
                              "opp": fi.index(o) if o in fi else None,
                              "sec": None if d != d else float(watch.sec_u[r, q])})
            h3 = getattr(watch, "hp_300", None)
            ends[i] = {"hp_trade": float(hp[r]), "hp_trade_lords": float(hpl[r]), "units": units,
                       "hp_trade_300": float(h3[r]) if h3 is not None else float(hp[r])}
    compiled = torch.device(device).type == "cuda"
    m, _ = BA.play_tactic(sp, lambda: EP.build_under("TEACH"), P, device, chunk=chunk, seeds=int(tseed), battles=battles,
                          watch_cls=watch_cls(), on_chunk=on_chunk, full=compiled)
    out = []
    for i, s in enumerate(sp):
        lim = int(s["limit"])
        us = ends[i]["units"]
        secs = [x["sec"] for x in us if x["sec"] is not None]
        dhs = [x["dh"] for x in us if x["dh"] is not None]
        for x, b in zip(us, s["slot_bands"] + [None] * len(us)):
            x["band"] = b
        out.append({"key": bkey(s), "seed": s["seed"], "twin": s["twin"], "held": s["held"], "line": s["line"],
                    "arm_line": s["arm_line"], "ground": s["s1"].get("ground"), "cause": s["s1"].get("cause"),
                    "slot_min": min(s["slot_bands"]) if s["slot_bands"] else None,
                    "contact": bool(secs) and min(secs) < lim, "t_contact": min(secs) if secs else None,
                    "dh": sum(dhs) / len(dhs) if dhs else None, "units": us, "hp_trade": ends[i]["hp_trade"],
                    "hp_trade_lords": ends[i]["hp_trade_lords"], "trade": float(m["trade"][i]),
                    "orders": float(m["game_orders_min"][i]) if "game_orders_min" in m else None,
                    "hp_trade_300": ends[i]["hp_trade_300"]})
        if k_net is not None:
            out[-1]["k"] = int(k_net[i])
    return out


def spec_rows(specs):
    from net.strategist import s1_scenes as SS
    out = []
    for s in specs:
        fd = SS.found(s)
        k = SS.label(s)
        out.append({"key": bkey(s), "seed": s["seed"], "twin": s["twin"], "twin_made": s["s1"].get("twin_made"),
                    "held": s["held"], "map": s["map"], "att": s["att"], "general": bool(s["s1"].get("general")),
                    "n": len(s["mine"]), "teach": k, "cause": s["s1"].get("cause"), "ground": s["s1"].get("ground"),
                    "delta_band": None if not k else round(fd["cands"][k - 1]["band"] - fd["centre"]["band"], 4),
                    "slot_min": None if not k else round(fd["cands"][k - 1]["band_slot_min"], 4),
                    "axes": s["s1"].get("axes"), "binds": SS.binds(s), "rejects": s["s1"].get("rejects")})
    return out


def _tree_sha(paths):
    from net import ROOT
    h = hashlib.sha256()
    for p in paths:
        q = ROOT / p
        files = sorted(q.rglob("*.py")) if q.is_dir() else [q]
        for f in files:
            if "__pycache__" in f.parts or not f.exists():
                continue
            h.update(f.relative_to(ROOT).as_posix().encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def _carriers(n):
    """The first n battles of the taught set (its cause scenes: synthetic ridges and captured maps)."""
    from net.strategist import s1_scenes as SS
    return [SS.scene(SEEDS["taught"] + i, "taught") for i in range(n)]


def _play_carried(specs, carriers, arm, tseed, device, chunk):
    """The set's rows played with carriers (CARRIED's why): each chunk = chunk / 2 of the set's battles + as many
    carriers (standing DEFAULT: arm_line gives a non-flat battle DEFAULT under PLACE; other arms are refused)."""
    assert arm in ("DEFAULT", "PLACE"), arm
    half = chunk // 2
    assert len(carriers) >= len(specs), (len(carriers), len(specs))
    mixed = []
    for a in range(0, len(specs), half):
        part = specs[a:a + half]
        mixed += part + carriers[a:a + len(part)]
    keys = {bkey(s) for s in specs}
    rows = [r for r in play(mixed, arm, tseed, device, chunk) if r["key"] in keys]
    assert len(rows) == len(specs), (len(rows), len(specs))
    return rows


def _key(name, specs, arm, chunk, tseed, net_sha):
    from net.strategist import s1_scenes as SS
    from net.tactic import exam_pin as EP
    h = hashlib.sha256(json.dumps([SS.identity(s) for s in specs], default=str).encode()).hexdigest()[:16]
    return {"set": name, "arm": arm, "starts": h, "n": len(specs), "chunk": chunk, "tseed": tseed, "sim": EP.sim_key(),
            "code": _tree_sha(KEY_TREES), "net": net_sha, "carriers": CARRIED.get(name)}


def spec_sha(specs):
    """The sha of a set's full specs (every field; tmp/drill_s1/c2/bridge.py writes the same digest)."""
    return hashlib.sha256(json.dumps(specs, sort_keys=True, default=str).encode()).hexdigest()[:16]


def starts_sha(specs):
    from net.strategist import s1_scenes as SS
    return hashlib.sha256(json.dumps([SS.identity(s) for s in specs], default=str).encode()).hexdigest()[:16]


def specs_of(name, n, n_g, cache=None, bridge=None):
    """set_specs, kept in cache/<stem>__specs.pkl (built once on the CPU before the GPU lock; a file of another set /
    size / digest is refused); bridge - the key bridge's {set: {starts, specs_sha}}: a set whose starts or full specs
    differ from the proven ones is refused (tmp/drill_s1/c2/key_bridge.json)."""
    import pickle
    n, n_g = (None, n_g) if name == "G" else (n, None)        # (a set's size is n, G's is n_g: the other one unread)
    f = None if not cache else Path(cache) / f"{_stem(name)}__specs.pkl"
    specs = None
    if f is not None and f.exists():
        d = pickle.loads(f.read_bytes())
        if d.get("name") != name or d.get("n") != n or d.get("n_g") != n_g or spec_sha(d["specs"]) != d.get("sha"):
            raise SystemExit(f"{f}: a specs cache of another set / size / digest - refused")
        specs = d["specs"]
    if specs is None:
        specs = set_specs(name, n, n_g)
        if f is not None:
            f.parent.mkdir(parents=True, exist_ok=True)
            part = f.with_suffix(".part")
            part.write_bytes(pickle.dumps({"name": name, "n": n, "n_g": n_g, "sha": spec_sha(specs), "specs": specs}))
            part.replace(f)
    if bridge is not None:
        b = bridge.get(name)
        if b is None or b["starts"] != starts_sha(specs) or b["specs_sha"] != spec_sha(specs):
            raise SystemExit(f"{name}: starts {starts_sha(specs)} / specs {spec_sha(specs)} differ from the key bridge's "
                             f"{None if b is None else (b['starts'], b['specs_sha'])} - refused")
    return specs


def bridge_sets(path):
    """{set: {starts, specs_sha, ...}} of the key bridge file (tmp/drill_s1/c2/key_bridge.json); SystemExit when it does
    not say ok."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if not d.get("ok"):
        raise SystemExit(f"{path}: the key bridge is not ok - refused")
    return d["sets"]


def run(a):
    import torch
    from net.drill import store as ST
    D = Path(a.out)
    D.mkdir(parents=True, exist_ok=True)
    only = set(a.arms.split(",")) if a.arms else None
    net_sha, k_of = None, None
    br = bridge_sets(a.bridge) if a.bridge else None
    if a.net:
        from net.strategist import s1_net as SN
        net, _ = SN.restore(a.net, a.device)
        net_sha = ST.sha256(a.net)[:16]
    for name in set_names(a.sets.split(",")):
        t0 = time.time()
        specs = specs_of(name, a.n, a.n_g, a.specs_cache or None, br)
        if a.build_only:
            print(name, len(specs), "specs built", spec_sha(specs), round(time.time() - t0), "s", flush=True)
            continue
        ST.atomic_text(json.dumps(spec_rows(specs), indent=1, default=str), D / f"{_stem(name)}__specs.json")
        arms = [x for x in ARMS_OF[_kind(name)] if only is None or x in only]
        if a.net and name != "flat":
            from net.strategist import s1_net as SN
            k_of = SN.choose(net, SN.observe(specs, a.device), a.device)
            arms.append("NET")
        for arm in arms:
            ts = _tseed(name)
            key = _key(name, specs, arm, a.chunk, ts, net_sha if arm == "NET" else None)
            f = D / f"{_stem(name)}__{arm}.json"
            if f.exists() and json.loads(f.read_text(encoding="utf-8")).get("key") == key:
                print(name, arm, "kept (same key)", flush=True)
                continue
            if name in CARRIED:
                rows = _play_carried(specs, _carriers(len(specs)), arm, ts, a.device, a.chunk)
            else:
                rows = play(specs, arm, ts, a.device, a.chunk, k_of if arm == "NET" else None)
            ST.atomic_text(json.dumps({"key": key, "rows": rows}, indent=1), f)
            print(name, arm, len(rows), "battles", round(time.time() - t0), "s", flush=True)
            if a.device.startswith("cuda"):
                torch.cuda.empty_cache()


# ------------------------------------------------------------------ the table
def stat(xs):
    xs = [x for x in xs if x is not None and x == x]
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": None, "two_errors": None, "lo": None, "hi": None}
    mu = sum(xs) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    e = 2 * sd / math.sqrt(n)
    return {"n": n, "mean": round(mu, 4), "two_errors": round(e, 4), "lo": round(mu - e, 4), "hi": round(mu + e, 4)}


def _stem(name):
    """A set's file stem: 'held:x' -> 'held-x' (a colon is not a Windows file name's character)."""
    return name.replace(":", "-")


def _name(stem):
    return "held:" + stem[5:] if stem.startswith("held-") else stem


def _load(D, name, arm):
    """{battle key: row} of the arm's file, or None."""
    f = Path(D) / f"{_stem(name)}__{arm}.json"
    if not f.exists():
        return None
    return {r["key"]: r for r in json.loads(f.read_text(encoding="utf-8"))["rows"]}


def spec_keys(D, name):
    """{battle key: spec row} of a set's specs file in D ({} when absent)."""
    f = Path(D) / f"{_stem(name)}__specs.json"
    return {r["key"]: r for r in json.loads(f.read_text(encoding="utf-8"))} if f.exists() else {}


def _paired(A, B, field, keys):
    if A is None or B is None:
        return stat([])
    return stat([A[k][field] - B[k][field] for k in keys if k in A and k in B
                 and A[k][field] is not None and B[k][field] is not None])


def _judge(ok, n, nmin):
    if n < nmin:
        return "NOT JUDGED"
    return "pass" if ok else "fail"


def _line(status, n, nmin, **cells):
    return {"status": status, "n": n, "min": nmin, **cells}


def _weakest(row):
    """(the weakest front-row slot's dh or None, touched?) of a row."""
    us = [x for x in row["units"] if x.get("band") is not None]
    if not us:
        return None, False
    w = min(us, key=lambda x: x["band"])
    return w["dh"], w["dh"] is not None


def table(D, arm="TEACH"):
    """{line: {status, n, min, ...}} (module doc's lines) of the arm (TEACH before the training, NET after); "failed":
    the lines failed or not judged."""
    D = Path(D)
    out = {"arm": arm, "lines": LINES, "min_n": MIN_N}
    sp = {}
    for f in D.glob("*__specs.json"):
        sp[_name(f.name.split("__")[0])] = {r["key"]: r for r in json.loads(f.read_text(encoding="utf-8"))}
    tau = sp.get("taught") or {}
    cause = [k for k, r in tau.items() if r["cause"]]
    twins = [k for k, r in tau.items() if not r["cause"]]
    A, Dd, Du = (_load(D, "taught", x) for x in (arm, "DEFAULT", "DUMB"))
    both = lambda X, Y, ks: [k for k in ks if X and Y and k in X and k in Y]          # noqa: E731
    cA = both(A, Dd, cause)
    dh = _paired(A, Dd, "dh", cA)
    units = [x for k in cA for x in A[k]["units"]]
    above = sum(x["dh"] is not None and x["dh"] > 0 for x in units) / max(len(units), 1)
    cont = sum(A[k]["contact"] for k in cA) / max(len(cA), 1)
    ok = (dh["lo"] is not None and dh["lo"] >= LINES["exec_dh_lo"] and above >= LINES["exec_units_above"]
          and cont >= LINES["exec_contact"])
    out["exec"] = _line(_judge(ok, min(len(cA), dh["n"]), MIN_N["exec"]), min(len(cA), dh["n"]), MIN_N["exec"],
                        dh_c=dh, units_above=round(above, 4), units_n=len(units),
                        units_no_contact=sum(x["dh"] is None for x in units), contact=round(cont, 4),
                        contact_DEFAULT=round(sum(Dd[k]["contact"] for k in cA) / max(len(cA), 1), 4) if Dd else None,
                        dh_c_arm=stat([A[k]["dh"] for k in cA]), dh_c_DEFAULT=stat([Dd[k]["dh"] for k in cA]))
    tr = _paired(A, Dd, "hp_trade", cA)
    out["freeze"] = _line(_judge(tr["lo"] is not None and tr["lo"] >= 0, tr["n"], MIN_N["freeze"]), tr["n"],
                          MIN_N["freeze"], hp_trade=tr, rule="lower bound < 0 -> FAIL")
    gate = None
    if tr["n"] >= MIN_N["value"]:
        gate = ("half the gap" if tr["mean"] >= LINES["value_errors"] * tr["two_errors"] / 2
                and tr["mean"] >= LINES["value_abs"] else "not worse than the teacher")
    out["value"] = _line("reported" if gate else "NOT JUDGED", tr["n"], MIN_N["value"], hp_trade=tr,
                         hp_trade_lords=_paired(A, Dd, "hp_trade_lords", cA), gate=gate)
    tw = both(A, Dd, twins)
    ch = [k for k in tw if A[k]["line"] != Dd[k]["line"]]
    q = _paired(A, Dd, "hp_trade", tw)
    sh = len(ch) / max(len(tw), 1)
    held_tw = []
    for name in sp:
        if name.startswith("held:"):
            Ah, Dh = _load(D, name, arm), _load(D, name, "DEFAULT")
            held_tw += [Ah[k]["hp_trade"] - Dh[k]["hp_trade"] for k, r in sp[name].items()
                        if not r["cause"] and Ah and Dh and k in Ah and k in Dh]
    out["quiet"] = _line(_judge(sh <= LINES["quiet_changed"] and q["lo"] is not None and q["lo"] >= LINES["quiet_lo"],
                                min(len(tw), q["n"]), MIN_N["quiet"]), min(len(tw), q["n"]), MIN_N["quiet"],
                         changed=round(sh, 4), changed_n=len(ch), hp_trade=q, held_twins_hp_trade=stat(held_tw))
    allk = both(A, Dd, list(tau))
    oa = stat([A[k]["orders"] for k in allk])
    od = stat([Dd[k]["orders"] for k in allk])
    ok = oa["mean"] is not None and od["mean"] is not None and oa["mean"] <= od["mean"] + LINES["orders_over"] \
        and oa["mean"] <= LINES["orders_max"]
    out["orders"] = _line(_judge(ok, oa["n"], MIN_N["orders"]), oa["n"], MIN_N["orders"], arm_orders=oa, DEFAULT=od)
    s = _paired(A, Du, "hp_trade", both(A, Du, cause))
    out["sign"] = _line(_judge(s["lo"] is not None and s["lo"] > 0, s["n"], MIN_N["sign"]), s["n"], MIN_N["sign"],
                        hp_trade=s, dh_c=_paired(A, Du, "dh", both(A, Du, cause)))
    # PLACE on the flat set
    fl = sp.get("flat") or {}
    Pl, Df = _load(D, "flat", "PLACE"), _load(D, "flat", "DEFAULT")
    p = _paired(Pl, Df, "hp_trade", list(fl))
    stop = p["n"] >= 2 and abs(p["mean"]) > p["two_errors"]
    out["place"] = _line("NOT JUDGED" if p["n"] < MIN_N["place"] else ("fail: STOP, not the height" if stop
                                                                         else "reported (within two errors)"),
                         p["n"], MIN_N["place"], hp_trade=p, dh_c=_paired(Pl, Df, "dh", list(fl)))
    # held axes
    from net.strategist import s1_scenes as SS
    held = {}
    for ax in SS.HELD:
        name = f"held:{ax}"
        rows = sp.get(name)
        if rows is None:
            held[name] = _line("NOT JUDGED", 0, MIN_N["held"], note="no set")
            continue
        Ah, Dh = _load(D, name, arm), _load(D, name, "DEFAULT")
        c = [k for k, r in rows.items() if r["cause"]]
        b = [all(rows[k]["binds"].values()) for k in c]
        g = _paired(Ah, Dh, "dh", c)
        held[name] = _line(_judge(sum(b) == len(b) and g["lo"] is not None and g["lo"] > 0, g["n"], MIN_N["held"]),
                           g["n"], MIN_N["held"], binds=round(sum(b) / max(len(b), 1), 4), dh_c=g,
                           hp_trade=_paired(Ah, Dh, "hp_trade", c))
    out["held"] = held
    # bins (taught's cause scenes)
    bins, first_fire = {}, None
    for lo, hi in BINS:
        ks = [k for k in cA if lo <= tau[k]["delta_band"] < hi]
        st = _paired(A, Dd, "hp_trade", ks)
        nm = f"{lo}-{hi if hi < 9 else 'up'}"
        bins[nm] = st
        if first_fire is None and st["n"]:
            first_fire = (nm, st)
    small = [nm for nm, st in bins.items() if st["n"] < MIN_N["bins"]]
    out["bins"] = _line("NOT JUDGED" if small or first_fire is None else
                        ("fail" if first_fire[1]["mean"] < 0 else "pass"),
                        min(st["n"] for st in bins.values()), MIN_N["bins"], bins=bins,
                        lowest_firing=first_fire and first_fire[0], below_min=small)
    # strata by the TEACH line's band_slot_min (taught + held cause scenes)
    strata, veto = {}, []
    pool = []
    for name, rows in sp.items():
        if name == "taught" or name.startswith("held:"):
            Ax, Dx = _load(D, name, arm), _load(D, name, "DEFAULT")
            pool += [(rows[k], Ax[k], Dx[k]) for k in rows if rows[k]["cause"] and Ax and Dx and k in Ax and k in Dx]
    for lo, hi in STRATA:
        for gk in ("captured", "synthetic", "all"):
            sel = [(r, a_, d_) for r, a_, d_ in pool if lo <= (a_["slot_min"] if a_["slot_min"] is not None else -1) < hi
                   and (gk == "all" or r["ground"] == gk)]
            w = [_weakest(a_) for _, a_, _ in sel]
            wd = stat([d for d, t in w if t])
            trs = stat([a_["hp_trade"] - d_["hp_trade"] for _, a_, d_ in sel])
            nm = f"{lo if lo > -9 else ''}-{hi if hi < 9 else ''}:{gk}"
            strata[nm] = {"n": len(sel), "min": MIN_N["strata"], "weakest_dh": wd,
                          "weakest_no_contact": sum(not t for _, t in w),
                          "contact": round(sum(a_["contact"] for _, a_, _ in sel) / max(len(sel), 1), 4),
                          "hp_trade": trs}
            if gk == "all" and len(sel) >= MIN_N["strata"] and (
                    (wd["mean"] is not None and wd["mean"] <= 0) or (trs["mean"] is not None and trs["mean"] < 0)):
                veto.append(nm)
    out["strata"] = {"status": "reported", "strata": strata, "veto": veto,
                     "decision": "veto band_slot_min >= 0.08 for the teacher" if veto else "no veto"}
    # G
    G = sp.get("G") or {}
    Ag, Dg = _load(D, "G", arm), _load(D, "G", "DEFAULT")
    alli = both(Ag, Dg, list(G))
    nh = _paired(Ag, Dg, "hp_trade", alli)
    flat = [k for k in alli if G[k]["map"] == "flat"]
    same = lambda a_, d_: all(a_[f] == d_[f] for f in ("hp_trade", "trade", "units", "t_contact", "orders"))   # noqa
    bit = all(same(Ag[k], Dg[k]) for k in flat)
    fires = sum(1 for r in G.values() if r["teach"]) / max(len(G), 1)
    out["noharm"] = _line(_judge(nh["lo"] is not None and nh["lo"] >= LINES["noharm_lo"], nh["n"], MIN_N["noharm"]),
                          nh["n"], MIN_N["noharm"], hp_trade=nh, fires=round(fires, 4))
    out["flat_bit"] = _line(_judge(bit, len(flat), MIN_N["flat_bit"]), len(flat), MIN_N["flat_bit"],
                            rule="G's flat field: TEACH = DEFAULT bit for bit")
    gates = ["exec", "sign", "quiet", "noharm", "flat_bit", "orders", "freeze", "place", "bins"]
    failed = [g for g in gates if not str(out[g]["status"]).startswith(("pass", "reported"))]
    failed += [h for h, v in held.items() if v["status"] != "pass"]
    out["failed"] = failed
    out["passed"] = not failed
    return out


# ------------------------------------------------------------------ the new recipe (METHOD steps 4 and 6, 23.11)
# The lines: tmp/drill_s1/c2/LINES_s1.md (written before the run). An episode = one cause scene (the strategist decides
# once, before the battle). Decisions on the unrounded numbers; a line without its full support is NOT JUDGED.
N_TAUGHT, N_HELD, N_G = 256, 64, 512
HELD_PAIRS_MIN = 48
_HELD_PAIRS_WHY = ("a held axis' dh pair needs both arms in contact: DEFAULT's line never meets the enemy on a few cause "
                   "scenes (the exam: 61 / 63 / 58 of 64 on band / distance / enemy) - the 64 scenes present, >= 48 pairs")
STEP4 = {"aim": 0.6, "quiet": 0.03, "orders": 1.5, "harm": -0.02}
STEP6 = {"b1": 0.85, "axis": 0.75, "gap": 0.10, "aim": 0.10, "quiet": 0.03, "neg": 0.5, "orders": 1.5, "harm": -0.02}
HUMAN_ORDERS = 1.0
_LINES_WHY = ("METHOD step 4 (the teacher: aim >= 0.6, quiet, orders in the human norm, every held axis acts, no gross "
              "harm) and step 6 (the network: B1 >= 0.85, each held axis >= 0.75 and taught - axis <= 0.10, aim >= the "
              "teacher's - 0.10, quiet <= the teacher's + 0.03, NEG <= 0.5, orders <= 1.5 x the human's 1.0 a "
              "unit-minute - net/common/reissue.py, as LINES_o1, harm at <= -0.02 on 512 G); tmp/drill_s1/c2/LINES_s1.md")


def _share(k, n):
    return {"k": k, "n": n, "p": (k / n) if n else None,
            "two_errors": 2 * math.sqrt(max((k / n) * (1 - k / n), 1e-12) / n) if n else None}


def _mean(xs):
    xs = [x for x in xs if x is not None and x == x]
    n = len(xs)
    if not n:
        return {"n": 0, "mean": None, "two_errors": None}
    mu = sum(xs) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return {"n": n, "mean": mu, "two_errors": 2 * sd / math.sqrt(n)}


def same_rows(A, B):
    """{key: row} A and B the same battles bit for bit, the fields rows gained on 23.11 (NEW_FIELDS) set aside."""
    if A is None or B is None or sorted(A) != sorted(B):
        return False
    strip = lambda r: json.dumps({k: v for k, v in r.items() if k not in NEW_FIELDS}, sort_keys=True)   # noqa: E731
    return all(strip(A[k]) == strip(B[k]) for k in A)


def _rows_full(D, name, arm, want):
    """{key: row} of the arm's file when it holds exactly the battles `want` (a list of keys), else None."""
    R = _load(D, name, arm)
    if R is None or sorted(R) != sorted(want):
        return None
    return R


def _status(ok, support_ok):
    return "NOT JUDGED" if not support_ok else ("pass" if ok else "fail")


def _units_above(rows, keys):
    us = [x for k in keys for x in rows[k]["units"]]
    return sum(x["dh"] is not None and x["dh"] > 0 for x in us), len(us)


def _sets_present(D):
    out = {}
    for name in ["taught", "G"] + [f"held:{a}" for a in __import__("net.strategist.s1_scenes", fromlist=["HELD"]).HELD]:
        sp = spec_keys(D, name)
        if sp:
            out[name] = sp
    return out


def _arm_auth(D, names, arm, bridge, want_code=None, want_net=None, want_sim=None):
    """[problem] of an arm's files: the base arms' file sha = the key bridge's; a NET file's key - starts = the bridge's,
    code = the tree now, net = the candidate's sha."""
    bad = []
    for name in names:
        f = Path(D) / f"{_stem(name)}__{arm}.json"
        if not f.exists():
            bad.append(f"{f.name}: missing")
            continue
        b = (bridge or {}).get(name) or {}
        if arm in (b.get("arms") or {}):
            want = b["arms"][arm]["file_sha256"]
            if hashlib.sha256(f.read_bytes()).hexdigest() != want:
                bad.append(f"{f.name}: sha differs from the key bridge's")
        elif arm == "NET":
            k = json.loads(f.read_text(encoding="utf-8"))["key"]
            if k.get("starts") != b.get("starts"):
                bad.append(f"{f.name}: starts {k.get('starts')} != the bridge's {b.get('starts')}")
            if want_code is not None and k.get("code") != want_code:
                bad.append(f"{f.name}: code {k.get('code')} != the tree's {want_code}")
            if want_sim is not None and k.get("sim") != want_sim:
                bad.append(f"{f.name}: sim {k.get('sim')} != the key bridge's {want_sim}")
            if want_net is not None and k.get("net") != want_net:
                bad.append(f"{f.name}: net {k.get('net')} != the candidate's {want_net}")
        else:
            bad.append(f"{f.name}: an arm the bridge does not hold")
    return bad


def teacher_verdict(D, bridge_path):
    """METHOD step 4 for the teacher by behaviour (LINES_s1.md 'teacher'): {lines, printed, verdict}."""
    from net.strategist import s1_scenes as SS
    br = bridge_sets(bridge_path)
    D = Path(D)
    sp = _sets_present(D)
    L, P = {}, {}
    names = ["taught"] + [f"held:{a}" for a in SS.HELD] + ["G"]
    auth = []
    for arm in ("DEFAULT", "TEACH"):
        auth += _arm_auth(D, [n for n in names if n in sp], arm, br)
    auth += [f"{n}: no specs" for n in names if n not in sp]
    tau = sp.get("taught", {})
    cause = [k for k, r in tau.items() if r["cause"]]
    twins = [k for k, r in tau.items() if not r["cause"]]
    T, Dd = _rows_full(D, "taught", "TEACH", list(tau)), _rows_full(D, "taught", "DEFAULT", list(tau))
    ok_t = T is not None and Dd is not None and len(cause) == N_TAUGHT and len(twins) == N_TAUGHT
    k, n = _units_above(T, cause) if ok_t else (0, 0)
    a = _share(k, n)
    L["aim"] = {"status": _status(a["p"] is not None and a["p"] >= STEP4["aim"], ok_t), "units_above": a,
                "contact": _share(sum(T[c]["contact"] for c in cause), len(cause)) if ok_t else None,
                "dh_c_TEACH_minus_DEFAULT": _mean([T[c]["dh"] - Dd[c]["dh"] for c in cause if T[c]["dh"] is not None
                                                   and Dd[c]["dh"] is not None]) if ok_t else None,
                "rule": f"taught causes: TEACH's units meeting their first enemy higher >= {STEP4['aim']} (no contact = miss)"}
    held_tw, ch, tw_n = [], 0, 0
    axes = {}
    for ax in SS.HELD:
        nm = f"held:{ax}"
        rows = sp.get(nm, {})
        c = [k_ for k_, r in rows.items() if r["cause"]]
        t = [k_ for k_, r in rows.items() if not r["cause"]]
        Th, Dh = _rows_full(D, nm, "TEACH", list(rows)), _rows_full(D, nm, "DEFAULT", list(rows))
        ok = Th is not None and Dh is not None and len(c) == N_HELD and len(t) == N_HELD
        binds = all(all(rows[k_]["binds"].values()) for k_ in c) if c else False
        g = _mean([Th[k_]["dh"] - Dh[k_]["dh"] for k_ in c if Th[k_]["dh"] is not None and Dh[k_]["dh"] is not None]) \
            if ok else {"n": 0, "mean": None, "two_errors": None}
        lo = None if g["mean"] is None else g["mean"] - g["two_errors"]
        axes[ax] = {"status": _status(binds and lo is not None and lo > 0, ok and g["n"] >= HELD_PAIRS_MIN), "binds": binds,
                    "dh_c_TEACH_minus_DEFAULT": g, "lower": lo}
        if ok:
            ch += sum(Th[k_]["line"] != Dh[k_]["line"] for k_ in t)
            tw_n += len(t)
            held_tw += [Th[k_]["hp_trade"] - Dh[k_]["hp_trade"] for k_ in t]
    L["held_axes_act"] = {"status": "pass" if all(v["status"] == "pass" for v in axes.values()) else
                          ("NOT JUDGED" if any(v["status"] == "NOT JUDGED" for v in axes.values()) else "fail"),
                          "axes": axes, "rule": "each held axis binds 100 % and TEACH - DEFAULT dh_c lower bound > 0 "
                                                f"({N_HELD} cause scenes each)"}
    if ok_t:
        ch += sum(T[k_]["line"] != Dd[k_]["line"] for k_ in twins)
        tw_n += len(twins)
    q = _share(ch, tw_n)
    L["quiet"] = {"status": _status(q["p"] is not None and q["p"] <= STEP4["quiet"], ok_t and tw_n == N_TAUGHT + 7 * N_HELD),
                  "placement_changed": q, "twins_hp_trade_TEACH_minus_DEFAULT": _mean(
                      held_tw + ([T[k_]["hp_trade"] - Dd[k_]["hp_trade"] for k_ in twins] if ok_t else [])),
                  "rule": f"taught + held twins (no cause): TEACH's line differs from DEFAULT's <= {STEP4['quiet']}"}
    G = sp.get("G", {})
    Tg, Dg = _rows_full(D, "G", "TEACH", list(G)), _rows_full(D, "G", "DEFAULT", list(G))
    ok_g = Tg is not None and Dg is not None and len(G) == N_G
    hm = _mean([Tg[k_]["hp_trade"] - Dg[k_]["hp_trade"] for k_ in G]) if ok_g else {"n": 0, "mean": None}
    flat = [k_ for k_ in G if G[k_]["map"] == "flat"]
    same = ok_g and bool(flat) and all(all(Tg[k_][f] == Dg[k_][f] for f in ("hp_trade", "trade", "units", "t_contact",
                                                                            "orders")) for k_ in flat)
    L["no_gross_harm"] = {"status": _status(hm["mean"] is not None and hm["mean"] > STEP4["harm"] and same,
                                            ok_g and hm["n"] == N_G),
                          "G_hp_trade_TEACH_minus_DEFAULT_end": hm, "flat_bit_for_bit": same, "flat_n": len(flat),
                          "rule": f"G ({N_G}): mean TEACH - DEFAULT at the end > {STEP4['harm']} (+300 s not stored in "
                                  "the exam's arms); the flat field's battles TEACH = DEFAULT bit for bit"}
    om = {}
    for nm in names:
        Ta = _load(D, nm, "TEACH")
        om[nm] = _mean([r["orders"] for r in Ta.values()]) if Ta else {"n": 0, "mean": None}
    ok_o = all(v["n"] for v in om.values())
    L["orders"] = {"status": _status(all(v["mean"] <= STEP4["orders"] * HUMAN_ORDERS for v in om.values() if v["mean"]
                                         is not None), ok_o), "TEACH_by_set": om,
                   "rule": f"TEACH's orders to the game a unit-minute, each set's mean <= {STEP4['orders']} x "
                           f"{HUMAN_ORDERS} (the human norm, accepted)"}
    old = table(D, "TEACH")
    P["trade_cause_TEACH_minus_DEFAULT"] = old["value"]["hp_trade"]
    P["trade_cause_TEACH_minus_DUMB"] = old["sign"]["hp_trade"]
    P["flat_PLACE_minus_DEFAULT"] = old["place"]["hp_trade"]
    P["band_bins"] = old["bins"]["bins"]
    P["strata"] = old["strata"]["decision"]
    if auth:
        st = "NOT JUDGED"
    else:
        sts = [v["status"] for v in L.values()]
        st = "FAILED" if "fail" in sts else ("NOT JUDGED" if "NOT JUDGED" in sts else "PASSED")
    return {"verdict": st, "arm": "TEACH", "lines": L, "printed": P, "auth": auth, "lines_why": _LINES_WHY,
            "eyes": "2 battles by eye apart (METHOD step 4): tmp/drill_s1/c2/vision", "bridge": str(bridge_path)}


def _b1_rows(sp, N, names):
    """[(set, key, hit)] of the cause scenes: the network's candidate (row 'k') = the teacher's (spec 'teach')."""
    out = []
    for nm in names:
        for k_, r in sp[nm].items():
            if r["cause"]:
                out.append((nm, k_, N[nm][k_].get("k") == r["teach"] if N.get(nm) and k_ in N[nm] else None))
    return out


def accept(D, run_dir, bridge_path, neg_choices=None):
    """METHOD step 6 for the network (LINES_s1.md 'network'): {verdict PASSED | FAILED | NOT JUDGED, lines, printed}.
    D holds the base arms (copied, sha = the bridge's) + NET of the candidate run_dir/final.pt (sha = run.json's); NEG -
    neg_choices {set: {key: k}} of the untrained network on the same scenes (accept_neg)."""
    from net.strategist import s1_scenes as SS
    D = Path(D)
    br = bridge_sets(bridge_path)
    rj = json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8"))
    fsha = hashlib.sha256((Path(run_dir) / "final.pt").read_bytes()).hexdigest()
    auth = []
    if rj.get("sha256") != fsha or not rj.get("passed") or rj.get("card") != "s0_1":
        auth.append(f"run.json: passed {rj.get('passed')} / card {rj.get('card')} / sha {str(rj.get('sha256'))[:16]} "
                    f"!= final.pt {fsha[:16]}")
    held = [f"held:{a}" for a in SS.HELD]
    names = ["taught"] + held + ["G"]
    sp = _sets_present(D)
    auth += [f"{n}: no specs" for n in names if n not in sp]
    code = _tree_sha(KEY_TREES)
    for arm in ("DEFAULT", "TEACH"):
        auth += _arm_auth(D, [n for n in names if n in sp], arm, br)
    sim_b = (json.loads(Path(bridge_path).read_text(encoding="utf-8")).get("sim") or {}).get("now")
    auth += [] if sim_b else ["the key bridge holds no sim key"]
    auth += _arm_auth(D, [n for n in names if n in sp], "NET", br, code, fsha[:16], sim_b)
    N = {n: _rows_full(D, n, "NET", list(sp.get(n, {}))) for n in names}
    T = {n: _rows_full(D, n, "TEACH", list(sp.get(n, {}))) for n in names}
    Df = {n: _rows_full(D, n, "DEFAULT", list(sp.get(n, {}))) for n in names}
    full = all(N[n] is not None and T[n] is not None and Df[n] is not None for n in names) and not auth
    L, P = {}, {}
    if full:
        sizes = {n: sum(r["cause"] for r in sp[n].values()) for n in names if n != "G"}
        full = sizes.get("taught") == N_TAUGHT and all(sizes.get(h) == N_HELD for h in held) and len(sp["G"]) == N_G
    if not full:
        return {"verdict": "NOT JUDGED", "auth": auth, "lines": {}, "why": "an arm, a set or its full support missing"}
    ep = _b1_rows(sp, N, ["taught"] + held)
    hits = sum(bool(h) for _, _, h in ep)
    b1 = _share(hits, len(ep))
    L["B1"] = {"status": _status(b1["p"] >= STEP6["b1"], len(ep) == N_TAUGHT + 7 * N_HELD), "share": b1,
               "rule": f"cause scenes (taught {N_TAUGHT} + held 7 x {N_HELD}): NET's candidate = the teacher's (the "
                       f"pointer's index 1-3, exact) >= {STEP6['b1']}"}
    tt = _share(sum(bool(h) for s, _, h in ep if s == "taught"), sum(1 for s, _, _ in ep if s == "taught"))
    ax = {}
    for h in held:
        a_ = _share(sum(bool(x) for s, _, x in ep if s == h), sum(1 for s, _, _ in ep if s == h))
        ax[h] = {"share": a_, "gap": tt["p"] - a_["p"],
                 "status": _status(a_["p"] >= STEP6["axis"] and tt["p"] - a_["p"] <= STEP6["gap"], a_["n"] == N_HELD)}
    L["held_axes"] = {"status": "pass" if all(v["status"] == "pass" for v in ax.values()) else "fail", "taught": tt,
                      "axes": ax, "rule": f"each held axis >= {STEP6['axis']} and taught - axis <= {STEP6['gap']}"}
    ck = {n: [k_ for k_, r in sp[n].items() if r["cause"]] for n in ["taught"] + held}
    kn = sum(_units_above(N[n], ck[n])[0] for n in ck)
    kt = sum(_units_above(T[n], ck[n])[0] for n in ck)
    nu = sum(_units_above(N[n], ck[n])[1] for n in ck)
    nt = sum(_units_above(T[n], ck[n])[1] for n in ck)
    an, at = _share(kn, nu), _share(kt, nt)
    L["aim"] = {"status": _status(an["p"] >= at["p"] - STEP6["aim"], nu == nt and nu > 0), "NET": an, "TEACH": at,
                "rule": f"units meeting their first enemy higher on the cause scenes: NET >= TEACH - {STEP6['aim']}"}
    tw = {n: [k_ for k_, r in sp[n].items() if not r["cause"]] for n in ["taught"] + held}
    qn = _share(sum(N[n][k_]["k"] != 0 for n in tw for k_ in tw[n]), sum(len(v) for v in tw.values()))
    qt = _share(sum(T[n][k_]["line"] != Df[n][k_]["line"] for n in tw for k_ in tw[n]), sum(len(v) for v in tw.values()))
    L["quiet"] = {"status": _status(qn["p"] <= qt["p"] + STEP6["quiet"], qn["n"] == N_TAUGHT + 7 * N_HELD),
                  "NET_moved": qn, "TEACH_moved": qt,
                  "rule": f"twins (no cause): NET leaves the centre line <= TEACH's share + {STEP6['quiet']}"}
    if neg_choices is None:
        L["NEG"] = {"status": "NOT JUDGED", "why": "no NEG choices"}
    else:
        nh = [(s, k_) for s, k_, _ in ep]
        got = [neg_choices.get(s, {}).get(k_) for s, k_ in nh]
        ng = _share(sum(g == sp[s][k_]["teach"] for (s, k_), g in zip(nh, got) if g is not None), sum(g is not None
                                                                                                    for g in got))
        L["NEG"] = {"status": _status(ng["p"] is not None and ng["p"] <= STEP6["neg"], ng["n"] == len(ep)), "B1": ng,
                    "rule": f"the untrained StratNet1's B1 on the same cause scenes <= {STEP6['neg']}"}
    om = {n: _mean([r["orders"] for r in N[n].values()]) for n in names}
    L["orders"] = {"status": _status(all(v["mean"] <= STEP6["orders"] * HUMAN_ORDERS for v in om.values()),
                                     all(v["n"] == len(sp[n]) for n, v in om.items())), "NET_by_set": om,
                   "rule": f"NET's orders a unit-minute, each set's mean <= {STEP6['orders']} x {HUMAN_ORDERS}"}
    hz = {}
    Dr = _load(D, "G", "DEFAULT_r")                     # (G's DEFAULT replayed today: + 300 s and the end)
    if Dr is not None and not same_rows(Dr, Df["G"]):
        auth.append("G__DEFAULT_r.json: the replay differs from the stored DEFAULT (beyond the new fields)")
        Dr = None
    for w, f in (("end", "hp_trade"), ("300", "hp_trade_300")):
        base = Df["G"] if w == "end" else Dr
        if base is None or sorted(base) != sorted(sp["G"]):
            hz[w] = {"n": 0, "mean": None, "status": "NOT JUDGED"}
            continue
        m = _mean([N["G"][k_][f] - base[k_][f] for k_ in sp["G"] if f in N["G"][k_] and f in base[k_]])
        hz[w] = {**m, "status": _status(m["mean"] > STEP6["harm"], m["n"] == N_G),
                 "known": m["mean"] is not None and STEP6["harm"] < m["mean"] < 0}
    L["harm"] = {"status": "fail" if any(v["status"] == "fail" for v in hz.values()) else
                 ("NOT JUDGED" if any(v["status"] == "NOT JUDGED" for v in hz.values()) else "pass"), "G": hz,
                 "rule": f"G ({N_G}): mean NET - DEFAULT <= {STEP6['harm']} at + 300 s or the end -> fail; (-0.02, 0) "
                         "-> a known harm for the journal"}
    cs = [(n, k_) for n in ck for k_ in ck[n]]
    P["trade_cause_NET_minus_DEFAULT"] = _mean([N[n][k_]["hp_trade"] - Df[n][k_]["hp_trade"] for n, k_ in cs])
    P["trade_cause_NET_minus_TEACH"] = _mean([N[n][k_]["hp_trade"] - T[n][k_]["hp_trade"] for n, k_ in cs])
    P["G_TEACH_minus_DEFAULT_end"] = _mean([T["G"][k_]["hp_trade"] - Df["G"][k_]["hp_trade"] for k_ in sp["G"]])
    bins = {}
    for lo, hi in BINS:
        sel = [h for s, k_, h in ep if s == "taught" and lo <= sp[s][k_]["delta_band"] < hi]
        bins[f"{lo}-{hi if hi < 9 else 'up'}"] = _share(sum(bool(h) for h in sel), len(sel))
    P["B1_by_band_bin"] = bins
    P["G_crest_moves"] = _share(sum(N["G"][k_]["k"] != 0 for k_ in sp["G"]), len(sp["G"]))
    sts = [v["status"] for v in L.values()]
    st = "FAILED" if "fail" in sts else ("NOT JUDGED" if "NOT JUDGED" in sts else "PASSED")
    if auth:
        st = "NOT JUDGED"
    return {"verdict": st, "arm": "NET", "run_sha256": fsha, "lines": L, "printed": P, "auth": auth,
            "eyes": "3 battles by eye (METHOD step 6) - pending, not in this verdict; the manual's traps: none built",
            "lines_why": _LINES_WHY}


def neg_choices(D, device="cpu", cache=None, n_taught=N_TAUGHT, n_held=N_HELD):
    """{set: {key: k}} of a fresh StratNet1 (the untrained network; no parent: METHOD step 6 'the parent cannot') on the
    cause scenes of taught and held - the choice is made before the battle, so no play is needed."""
    from net.strategist import s1_net as SN
    from net.strategist import s1_scenes as SS
    import torch
    torch.manual_seed(0)
    net = SN.StratNet1().to(device)
    out = {}
    for nm in ["taught"] + [f"held:{a}" for a in SS.HELD]:
        specs = specs_of(nm, n_taught if nm == "taught" else n_held, None, cache)
        cs = [s for s in specs if s["s1"].get("cause")]
        ks = SN.choose(net, SN.observe(cs, device), device)
        out[nm] = {bkey(s): int(k) for s, k in zip(cs, ks)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("run", "table", "preflight", "teacher", "accept"))
    ap.add_argument("--bridge", default="", help="run / teacher / accept: the key bridge (tmp/drill_s1/c2/key_bridge.json)")
    ap.add_argument("--specs-cache", default="", help="run: the sets' specs kept here (built once, before the GPU)")
    ap.add_argument("--build-only", action="store_true", help="run: build (and cache) the specs, play nothing")
    ap.add_argument("--run", default="", help="accept: the candidate's training folder (run.json, final.pt)")
    ap.add_argument("--out", default="tmp/drill_s1/c/exam")
    ap.add_argument("--sets", default="taught,flat,held,G")
    ap.add_argument("--arms", default="", help="run: only these arms (default: each set's ARMS_OF)")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--n-g", type=int, default=512)
    ap.add_argument("--net", default="")
    ap.add_argument("--arm", default="TEACH", help="table: the arm judged (TEACH before, NET after)")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--verdict", default="", help="teacher / accept: the verdict file written (atomic; never into --out "
                                                    "for teacher - the stored exam is read only)")
    a = ap.parse_args(argv)
    if a.what == "run":
        return run(a)
    if a.what in ("teacher", "accept"):
        from net.drill import store as ST
        try:
            if a.what == "teacher":
                v = teacher_verdict(a.out, a.bridge)
            else:
                v = accept(a.out, a.run, a.bridge, neg_choices(a.out, a.device, a.specs_cache or None))
        except SystemExit:
            raise
        except Exception as e:                                  # (a crash is an ERROR, never a measured FAILED)
            print(f"{a.what.upper()} ERROR {type(e).__name__}: {e}", flush=True)
            sys.exit(3)
        f = Path(a.verdict or (f"tmp/drill_s1/c2/teacher_s1.json" if a.what == "teacher" else Path(a.out) / "accept_s1.json"))
        ST.atomic_text(json.dumps(v, indent=1, default=str), f)
        brief = {k: v_["status"] for k, v_ in v["lines"].items()}
        print(a.what.upper(), v["verdict"], json.dumps(brief), "auth", v.get("auth"), "->", f, flush=True)
        sys.exit({"PASSED": 0, "FAILED": 1, "NOT JUDGED": 2}[v["verdict"]])
    if a.what == "table":
        t = table(a.out, a.arm)
        print(json.dumps(t, indent=1, default=str))
        Path(a.out, f"table_{a.arm}.json").write_text(json.dumps(t, indent=1, default=str), encoding="utf-8")
        print("TABLE", "passed" if t["passed"] else f"FAILED / NOT JUDGED: {t['failed']}")
        sys.exit(0 if t["passed"] else 1)
    from collections import Counter
    for name in set_names(a.sets.split(",")):
        t0 = time.time()
        sp = set_specs(name, a.n, a.n_g)
        rows = spec_rows(sp)
        c = Counter((r["twin"], r["twin_made"]) for r in rows)
        rj = Counter()
        for r in rows:
            rj.update(r["rejects"] or {})
        print(name, len(sp), "battles", round(time.time() - t0, 1), "s | teacher fires", sum(bool(r["teach"]) for r in rows),
              "| twins", dict(c), "| binds", sum(all(r["binds"].values()) for r in rows if r["held"] and r["cause"]),
              "| maps", dict(Counter(r["map"].split(":")[0] for r in rows)), "| rejects", dict(rj), flush=True)


if __name__ == "__main__":
    main()

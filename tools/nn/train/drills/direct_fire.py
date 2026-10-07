"""Drill "direct_fire": a direct-fire unit (a flat shot, the passport's missile.direct: the militia's pistols, the
handgunners, the Night Runners' throwing stars) finds a place with a clear line of fire - beside its own men, at
an angle into the enemy's flank - and fires at the enemies that go round our infantry's flank (the pincer); it stays
where it is when moving out would put it in the way of a free enemy (docs/en/training/training.md "Drills").
The arcing shooters (archers, slings, crossbows) are never given orders, labelled or measured here.

The skill by the simulator's own rules, not by eye (`evaluate`): for every standing direct-fire unit with
ammunition, out of melee, not a lord, its fire's worth in gold a second here and at OFFSETS beside it (across the
line to its nearest enemy in reach), at its N_CAND nearest enemies within range + REACH_EXTRA_M:
    worth = men / reload x the share of its men in range (missile.rank_share for the units ranged per man, else
            centre to centre) x the share of its lines clear past friends (missile._lines: the bullet's arc,
            missile.arc_los) x the hit chance (missile.hit_chance) x HP a hit (melee.per_hit: the target's armour)
            x (1 - shield, from the front only) x (1 - 2 friendly_fire when the target is in melee) x the target's
            gold / health, less the shots that hit our own men on the way (the arc's catch) x their gold / health;
            x FLANK_W for an enemy at the flank or rear of one of our melee units (the pincer);
    a place's score = the best worth there x (HORIZON_S - the time to get there, turn and aim; a unit that fires on
            the move loses none); a place is UNSAFE when a free enemy melee unit (standing, out of melee, no
            missiles) runs to it within SAFE_S.
plan(st): the skilled decision - go to the best SAFE place (or any, if staying is unsafe too) when its score beats
staying's by GO_GAIN; else attack the best target from here when it is not the one aimed at (its worth below
1 / GO_GAIN of the best) or beyond the fire arc. The reckless decision: the best place, safe or not.
The counterfactual in normal battles (build/drill_fire/cf.py, s45_units/m30, 256 battles a script, paired):
the network's direct-fire units overridden by `position` - see docs/en/training/training.md "Drills".

Frame: only the embedded one (the situation inside a normal battle; frame() is the same - the clean and broad
frames were told apart by the actor at AUC 1.00, and kiting's skill stayed a habit of its drill). A normal generated
battle (tools/nn/armies: factions at random, both lords, attack or defence) where our side swaps K = 1-2 units of
nearest cost for direct-fire units of its faction (DIRECT) and is deployed again WITH them by the generator's own
rule (tools/nn/armies/place.py: missile units in the second line, behind the melee line - so they stand behind
their own men, as in every normal battle). All our direct-fire units are tagged TAG_OURS (the generator's own too).
A variant per battle (VARIANTS): "plain" - nothing more; "pincer" - 1-2 enemy melee units (TAG_ENEMY) beyond the
end of the enemy's front line on one flank go round the end of our line once near (envelop) and attack our end unit
in its flank; "hunter" - one enemy melee unit (TAG_HUNTER) behind the end of the enemy's line on one flank waits
and attacks any of our direct-fire units that comes within HUNT_M (the unsafe place). Each inserted enemy unit is
paid by the enemy dropping the unit of nearest cost; the gold evened (drills.balance).
Scripts: naive = ai_like for our whole army (its shooters keep their post behind the line); skilled = ai_like with
our direct-fire units on `plan`; reckless = the same with the reckless decision (verify.py plays it beside the two:
skilled must beat it on the hunter battles, else the unsafe case is not real). enemy = ai_like, the tagged units as
their variant says. The teacher labels with `teacher` (plan on a hold base: "stay" is HOLD) at `moments` (the
transfer detector's situation).
"""
import torch

from tools.nn.sim import geometry, melee, missile
from tools.nn.sim import orders as O
from tools.nn.train import drills as D
from tools.nn.train import opponents

EMPIRE, SKAVEN = "wh_main_emp_empire", "wh2_main_skv_skaven"
DIRECT = {EMPIRE: ("wh_dlc04_emp_inf_free_company_militia_0", "wh_main_emp_inf_handgunners"),
          SKAVEN: ("wh2_main_skv_inf_night_runners_0",)}
MELEE = {EMPIRE: ("wh_main_emp_inf_swordsmen", "wh_main_emp_inf_spearmen_0", "wh_main_emp_inf_spearmen_1"),
         SKAVEN: ("wh2_main_skv_inf_clanrats_0", "wh2_main_skv_inf_clanrat_spearmen_0", "wh2_main_skv_inf_clanrats_1")}
TAG_HUNTER = 3           # a hunter of the "hunter" variant (state STATIC tag; drills.TAG_OURS 1, TAG_ENEMY 2)
VARIANTS = (("plain", 0.4), ("pincer", 0.3), ("hunter", 0.3))
K_DIRECT = (1, 2)        # our direct-fire units swapped in, inclusive
K_FLANK = (1, 2)         # pincer: enemy flankers, inclusive
WIDTH_M = 30.0
GAP_M = (6.0, 20.0)      # an inserted enemy unit beyond the end of its line
HUNTER_BACK_M = (30.0, 70.0)   # the hunter this far behind the enemy's front
ENV_START_M = 160.0      # a flanker goes round once one of our melee units is this near
ENV_OUT_M = 15.0         # the flank point beyond our end unit's flank ...
ENV_BACK_M = 5.0         # ... and this far behind its centre
ENV_WAY_M = 25.0         # the waypoint beside its front (clear of it) while the flanker is still in front
ENV_READY_M = 12.0       # a flanker this near its point attacks
HUNT_M = 90.0            # a hunter attacks our direct-fire unit this near (centres)
GUARD_M = 50.0           # ... and any enemy this near

OFFSETS = (0.0, -45.0, -25.0, 25.0, 45.0)   # m across the line to the nearest enemy in reach (0: stay)
N_CAND = 3               # the nearest enemies in reach considered as targets
REACH_EXTRA_M = 40.0     # an enemy within range + this (centre to centre) is in reach
HORIZON_S = 30.0         # the fire's horizon of a place's score
GO_GAIN = 1.25           # go when the best place's score is this many times staying's
SAFE_S = 20.0            # a free enemy melee unit that runs to a place within this makes it unsafe
SAFE_EDGE_M = 20.0       # ... counted from this far (the two formations' half depths)
FLANK_W = 1.5            # worth x this for an enemy at the flank / rear of one of our melee units within FLANK_NEAR_M
FLANK_NEAR_M = 50.0
FLANK_DEG = 45.0         # beyond this off the front of our unit: its flank (contact.front_deg)
AIM_S = 5.0              # turn + aim + retarget after a move (aim_s 2-3.8, retarget_s 3)
MAP_M = 760.0            # places kept within the map
_PARAMS = []


def params():
    if not _PARAMS:
        from tools.nn.sim.params import load
        _PARAMS.append(load())
    return _PARAMS[0]


# --- the skill: worth of places, the decision -----------------------------------------------------------------

def shooters(st):
    """[B, N] the units the skill is about: standing, direct fire with ammunition, out of melee, not a lord."""
    u = st.u
    alive = (u["side"] > 0) & (u["men"] > 0) & ~u["gone"]
    return alive & ~u["r"] & u["direct"] & (u["a"] > 0) & (u["range"] > 0) & ~u["m"] & ~u["lord"]


def _flankers(u, pw, standing):
    """[B, j] enemies at the flank or rear of one of our (j's enemy's) melee units within FLANK_NEAR_M."""
    ours = standing & ~((u["range"] > 0) & (u["a"] > 0)) & ~u["lord"]
    off = pw["rel_i"].abs() > FLANK_DEG * geometry.DEG                          # [B, k, j]: j off k's front
    near = pw["enemy"] & (pw["dist"] < FLANK_NEAR_M) & off & ours[:, :, None] & standing[:, None, :]
    return near.any(1)


@torch.compiler.disable
def evaluate(st, prm=None):
    """The skill's numbers for every unit (meaningful where shooters(st)): dict of
    who [B, N]; score, px, pz, t_go [B, N, P] (P = len(OFFSETS), place 0 = here); best [B, N, P] the best target
    slot at each place (-1 none); unsafe [B, N, P]; worth0 [B, N] the best worth here; cur [B, N] the worth here of
    the target it aims at now (none yet: of the nearest in reach, the one fire at will takes); pw (the pairwise geometry
    now)."""
    prm = prm or params()
    u = st.u
    B, N = u["x"].shape
    dev = u["x"].device
    spacing = prm.sim["formation"]["spacing_m"]
    arc_on = prm.sim["missile"].get("arc_los") is not None
    who = shooters(st)
    alive = (u["side"] > 0) & (u["men"] > 0) & ~u["gone"]
    standing = alive & ~u["r"]
    pw = geometry.pairwise(u, spacing)
    foe = pw["enemy"] & standing[:, None, :]
    reach = foe & (pw["dist"] <= (u["range"] + REACH_EXTRA_M)[:, :, None])
    d_reach = torch.where(reach, pw["dist"], torch.full_like(pw["dist"], 1e9))
    k = min(N_CAND, N)
    cd, cand = torch.topk(-d_reach, k, dim=2)                       # [B, N, k] the nearest in reach
    has = -cd < 1e9
    th = pw["theta"].gather(2, cand[:, :, :1]).squeeze(2)          # across the line to the nearest in reach
    lx, lz = torch.cos(th), -torch.sin(th)
    single = u["men0"] <= 1
    hit_hp = melee.per_hit(u["m_damage"][:, :, None], u["m_ap"][:, :, None], u["armour"][:, None, :],
                           u["hp_man"][:, None, :], u["resist_missile"][:, None, :], single=single[:, None, :])
    gold_hp = (u["cost"] / u["hp0"].clamp(min=1.0))[:, None, :]          # [B, 1, j]
    rate = u["men"] / u["reload"].clamp(min=1e-3)                          # [B, i] shots a second
    flank = _flankers(u, pw, standing)
    weight = torch.where(flank, torch.full_like(u["x"], FLANK_W), torch.ones_like(u["x"]))
    ff = u["friendly_fire"][:, :, None] * u["m"][:, None, :].float()
    per_man = missile.per_man_mask(u, prm)
    free_melee = standing & ~u["m"] & ~((u["range"] > 0) & (u["a"] > 0))
    P = len(OFFSETS)
    out = {k_: torch.zeros(B, N, P, device=dev) for k_ in ("score", "px", "pz", "t_go")}
    out["best"] = torch.full((B, N, P), -1, dtype=torch.long, device=dev)
    out["unsafe"] = torch.zeros(B, N, P, dtype=torch.bool, device=dev)
    worth0 = torch.zeros(B, N, device=dev)
    cur = torch.zeros(B, N, device=dev)
    for p, off in enumerate(OFFSETS):
        x = (u["x"] + off * lx).clamp(-MAP_M, MAP_M)
        z = (u["z"] + off * lz).clamp(-MAP_M, MAP_M)
        if off:
            v = dict(u)
            v["x"], v["z"] = torch.where(who, x, u["x"]), torch.where(who, z, u["z"])
            pwp = geometry.pairwise(v, spacing)
        else:
            v, pwp = u, pw
        hit = missile.hit_chance(v, pwp, prm)
        share = (pwp["dist"] <= u["range"][:, :, None]).float()
        if per_man is not None:
            share = torch.where(per_man[:, :, None], missile.rank_share(v, pwp), share)
        front = (pwp["rel_j"].abs() <= prm.battle["shield_defence_angle_missile"] * geometry.DEG) & u["small_arms"][:, :, None]
        shield = torch.where(front, u["shield"][:, None, :], torch.zeros_like(pwp["dist"]))
        base = rate[:, :, None] * share * hit * hit_hp * (1 - shield) * (1 - 2 * ff) * gold_hp * weight[:, None, :]
        wbest = torch.zeros(B, N, device=dev)
        bbest = torch.full((B, N), -1, dtype=torch.long, device=dev)
        for c in range(k):
            tgt = torch.where(has[:, :, c], cand[:, :, c], torch.full_like(cand[:, :, c], -1))
            blocked, catch = missile._lines(v, pwp, tgt, prm, arc_on)
            w = base.gather(2, tgt.clamp(min=0)[:, :, None]).squeeze(2) * (1 - blocked)
            if catch is not None:
                w = w - (catch * hit_hp * gold_hp).sum(2) * rate * (1 - blocked)
            w = torch.where(tgt >= 0, w, torch.full_like(w, -1e9))
            better = w > wbest
            wbest = torch.where(better, w, wbest)
            bbest = torch.where(better, tgt, bbest)
            if p == 0:
                # the target it shoots now; none yet: the one fire at will takes (the nearest in reach)
                aimed = torch.where(u["aim_tgt"] >= 0, u["aim_tgt"] == tgt, torch.full_like(has[:, :, c], c == 0))
                cur = torch.where(aimed & (tgt >= 0), w.clamp(min=0), cur)
        tg = torch.where(u["fire_move"], torch.zeros_like(wbest),
                         torch.full_like(wbest, abs(off)) / u["run"].clamp(min=0.5) + (AIM_S if off else 0.0))
        out["score"][:, :, p] = wbest.clamp(min=0) * (HORIZON_S - tg).clamp(min=0)
        out["best"][:, :, p] = bbest
        out["px"][:, :, p], out["pz"][:, :, p], out["t_go"][:, :, p] = x, z, tg
        dx = x[:, :, None] - u["x"][:, None, :]
        dz = z[:, :, None] - u["z"][:, None, :]
        te = (torch.sqrt(dx * dx + dz * dz) - SAFE_EDGE_M).clamp(min=0) / u["run"][:, None, :].clamp(min=0.5)
        out["unsafe"][:, :, p] = (pw["enemy"] & free_melee[:, None, :] & (te < SAFE_S)).any(2)
        if p == 0:
            worth0 = wbest.clamp(min=0)
    out.update(who=who, worth0=worth0, cur=cur, pw=pw)
    return out


def _decide(ev, reckless):
    """(go [B, N], place [B, N]) of a decision on the evaluation ev."""
    score = ev["score"]
    if reckless:
        ok = torch.ones_like(ev["unsafe"])
    else:
        ok = ~ev["unsafe"] | ev["unsafe"][:, :, :1]      # staying is unsafe too: safety does not choose
    alt = torch.where(ok, score, torch.full_like(score, -1.0))
    alt = torch.cat([torch.full_like(alt[:, :, :1], -1.0), alt[:, :, 1:]], 2)
    v, place = alt.max(2)
    go = ev["who"] & (v > GO_GAIN * score[:, :, 0]) & (v > 0)
    return go, torch.where(go, place, torch.zeros_like(place))


PLAN_EVERY_S = 1.0       # plan() is computed again after this much battle time on the same state (the network
#                          decides once a second; the scripts and the detector are called every 0.5 s step)
_CACHE = {}


@torch.compiler.disable
def plan(st):
    """The decisions of every unit [B, N] (meaningful where shooters(st)): dict of go, place (skilled), go_r,
    place_r (reckless), retarget (stay and attack `tgt`), tgt, unsafe_go (reckless goes where skilled does not:
    the unsafe case), x, z / x_r, z_r (the places), ev. Kept for PLAN_EVERY_S of battle time on the same state."""
    t = float(st.t.max()) if st.t.numel() else 0.0
    c = _CACHE.get("plan")
    if c is not None and c[0] is st and c[2] == tuple(st.u["x"].shape) and c[1] <= t < c[1] + PLAN_EVERY_S - 1e-6:
        return c[3]
    out = _plan(st)
    _CACHE["plan"] = (st, t, tuple(st.u["x"].shape), out)
    return out


def _plan(st):
    ev = evaluate(st)
    go, place = _decide(ev, False)
    go_r, place_r = _decide(ev, True)
    u = st.u
    tgt = ev["best"][:, :, 0]
    rel = ev["pw"]["rel_i"].gather(2, tgt.clamp(min=0)[:, :, None]).squeeze(2).abs()
    off_arc = rel > u["arc"] * geometry.DEG
    worse = ev["cur"] * GO_GAIN < ev["worth0"]
    retarget = ev["who"] & ~go & (tgt >= 0) & (ev["worth0"] > 0) & (worse | off_arc)
    at = (lambda key, pl: ev[key].gather(2, pl[:, :, None]).squeeze(2))
    return dict(go=go, place=place, go_r=go_r, place_r=place_r, retarget=retarget, tgt=tgt,
                unsafe_go=go_r & ~go, x=at("px", place), z=at("pz", place), x_r=at("px", place_r),
                z_r=at("pz", place_r), ev=ev)


def position(st, reckless=False, base=None):
    """(Orders, given [B, N]): `base`'s orders (default hold) with the direct-fire units' that go: MOVE (run) to
    the place; that stay but should shoot another target (or one beyond their fire arc): ATTACK it (unless already
    ordered so); that stay with a target from here while the base would walk them: HOLD. given: where this script
    gave its own order."""
    o = base.clone() if base is not None else O.hold(st.B, st.N, st.device)
    p = plan(st)
    go = p["go_r"] if reckless else p["go"]
    x, z = (p["x_r"], p["z_r"]) if reckless else (p["x"], p["z"])
    D.move(o, go, x, z, run=True)
    retarget = p["retarget"] & ~go & (st.u["order_target"] != p["tgt"])
    D.attack(o, retarget, p["tgt"], run=False)
    # a shooter with a target from where it stands is not walked away by the base (ai_like walks its shooters back to
    # a post behind the melee line: after a step aside it undid it every second) - it stands and shoots
    stay = p["ev"]["who"] & ~go & ~retarget & (p["ev"]["worth0"] > 0) & (o.kind == O.MOVE)
    o.kind = torch.where(stay, torch.full_like(o.kind, O.HOLD), o.kind)
    return o, go | retarget | stay


def teacher(st):
    """The teacher's labels (tag-blind): `position` on a hold base - a unit that should stay (the unsafe case)
    is labelled HOLD."""
    return position(st)[0]


# --- the embedded frame ----------------------------------------------------------------------------------------

def _variant(rng):
    r, acc = float(rng.random()), 0.0
    for name, p in VARIANTS:
        acc += p
        if r < acc:
            return name
    return VARIANTS[-1][0]


def _generated(rng, ours, enemy, sub, drop):
    """A normal generated battle (drills.generated's) where our side swaps the units of nearest cost (not the lord)
    for the keys `sub` and is deployed again WITH them (tools/nn/armies/place.py), and the enemy drops a unit of
    nearest cost for each key of `drop` (inserted after, by the caller) and is deployed again without them."""
    from tools.nn.armies import generate
    from tools.nn.armies import place as placement
    from tools.nn.armies import pools as P
    from tools.nn.sim import scenario
    arena = generate.default().generate(rng, sides=(ours, enemy), name="embedded")
    pools = P.load()
    for tag, keys, add in (("own", sub, True), ("enemy", drop, False)):
        if not keys:
            continue
        side = arena["sides"][tag]
        pool = pools[side["faction"]]
        by_key = {u.key: u for u in pool.units}
        units = [by_key[u["key"]] for u in side["units"] if not u.get("general")]
        for key in keys:
            if units:
                c = D.cost(key)
                units.remove(min(units, key=lambda u: abs(u.cost - c)))
            if add:
                units.append(by_key[key])
        side["units"] = placement.place(pool.lord, units, arena["deployment_m"])
        side["cost"] = pool.lord.cost + sum(u.cost for u in units)
    desc = scenario.from_arena(arena, "attack" if rng.random() < 0.5 else "defend")
    for s in (1, 2):
        for u in desc["sides"][s]["units"]:
            u.pop("name", None)
    desc["sides"][1]["ai"], desc["sides"][2]["ai"] = False, True
    return desc


DIRECT_KEYS = frozenset(k for ks in DIRECT.values() for k in ks)


def embedded(rng):
    best = None
    for _ in range(D.EMBED_TRIES):
        fo, fe = (EMPIRE, SKAVEN)[int(rng.integers(2))], (EMPIRE, SKAVEN)[int(rng.integers(2))]
        k = int(rng.integers(K_DIRECT[0], K_DIRECT[1] + 1))
        sub = [DIRECT[fo][int(rng.integers(len(DIRECT[fo])))] for _ in range(k)]
        variant = _variant(rng)
        pick = (lambda: MELEE[fe][int(rng.integers(len(MELEE[fe])))])
        extra = ([pick() for _ in range(int(rng.integers(K_FLANK[0], K_FLANK[1] + 1)))] if variant == "pincer"
                 else [pick()] if variant == "hunter" else [])
        desc = _generated(rng, fo, fe, sub, extra)
        ours, enemy = desc["sides"][1]["units"], desc["sides"][2]["units"]
        for u in ours:
            if u["key"] in DIRECT_KEYS:
                u["tag"] = D.TAG_OURS
        if extra:
            ax = D.axes(desc, 1)
            b_enemy = enemy[0]["b"]
            s = float(rng.choice([-1.0, 1.0]))
            start = D.extent(enemy, ax, s) + float(rng.uniform(*GAP_M))
            f_e = D.front(enemy, ax, enemy=True)
            tag = D.TAG_ENEMY if variant == "pincer" else TAG_HUNTER
            back = 0.0 if variant == "pincer" else float(rng.uniform(*HUNTER_BACK_M))
            enemy += [D.unit(key, *ax.world(f_e + back, s * (start + i * (WIDTH_M + 6.0) + WIDTH_M / 2)), b_enemy,
                             WIDTH_M, tag=tag) for i, key in enumerate(extra)]
            D.shift(enemy, ax, -s * (len(extra) * WIDTH_M + (len(extra) - 1) * 6.0) / 2)
        desc["meta"] = {"variant": variant, "k": k, "direct": sub, "extra": extra}
        if D.balance(desc):
            return desc
        if best is None or D.imbalance(desc) < D.imbalance(best):
            best = desc
    return best


def frame(rng):
    """No clean frame: the embedded one (module docstring)."""
    return embedded(rng)


# --- the enemy's tagged units -----------------------------------------------------------------------------------

def envelop(st, o, mask):
    """Orders o (in place) for the units in mask [B, N] out of melee: once one of their enemy's melee units is within
    ENV_START_M, go round the nearer flank of the nearest such unit (by a waypoint beside its front while in front
    of it) to a point ENV_OUT_M beyond that flank and ENV_BACK_M behind its centre, then attack it."""
    u = st.u
    v = D.View(st)
    target_ok = v.standing & ~((u["range"] > 0) & (u["a"] > 0)) & ~u["lord"]
    i, d = v.nearest(target_ok)
    act = mask & v.standing & ~u["m"] & (d < ENV_START_M)
    kx, kz = u["x"].gather(1, i), u["z"].gather(1, i)
    kb = u["b"].gather(1, i) * geometry.DEG
    fx, fz = torch.sin(kb), torch.cos(kb)                 # the target's forward
    rx, rz = torch.cos(kb), -torch.sin(kb)                # ... and right
    front, _ = geometry.dims(u, params().sim["formation"]["spacing_m"])
    half = front.gather(1, i) / 2
    ox, oz = u["x"] - kx, u["z"] - kz
    ahead, lat = ox * fx + oz * fz, ox * rx + oz * rz
    s = torch.where(lat >= 0, torch.ones_like(lat), -torch.ones_like(lat))
    px = kx + rx * s * (half + ENV_OUT_M) - fx * ENV_BACK_M
    pz = kz + rz * s * (half + ENV_OUT_M) - fz * ENV_BACK_M
    in_front = (ahead > 0) & (lat.abs() < half + ENV_WAY_M - 5.0)
    wx = kx + rx * s * (half + ENV_WAY_M) + fx * 20.0
    wz = kz + rz * s * (half + ENV_WAY_M) + fz * 20.0
    gx, gz = torch.where(in_front, wx, px), torch.where(in_front, wz, pz)
    near = torch.sqrt((u["x"] - px) ** 2 + (u["z"] - pz) ** 2) < ENV_READY_M
    D.move(o, act & ~near, gx, gz, run=True)
    D.attack(o, act & near, i, run=True)
    return o


def hunt(st, o, mask):
    """Orders o (in place) for the units in mask out of melee: attack the nearest of their enemy's direct-fire units
    within HUNT_M, else any enemy within GUARD_M, else hold (a reserve)."""
    u = st.u
    v = D.View(st)
    i_d, d_d = v.nearest(u["direct"])
    i_a, d_a = v.nearest()
    free = mask & v.standing & ~u["m"]
    o.kind = torch.where(free, torch.full_like(o.kind, O.HOLD), o.kind)
    D.attack(o, free & (d_a < GUARD_M), i_a, run=True)
    D.attack(o, free & (d_d < HUNT_M), i_d, run=True)
    return o


def enemy(st):
    """ai_like; the pincer's flankers envelop, the hunter hunts."""
    o = opponents.ai_like(st)
    envelop(st, o, D.tagged(st, D.TAG_ENEMY))
    hunt(st, o, D.tagged(st, TAG_HUNTER))
    return o


def naive(st):
    """ai_like for our whole army (the direct-fire units keep their post behind the line)."""
    return opponents.ai_like(st)


def skilled(st):
    """ai_like with the direct-fire units on `plan` (where it gives an order)."""
    return position(st, base=opponents.ai_like(st))[0]


def reckless(st):
    """ai_like with the direct-fire units on the reckless decision (the best place, safe or not)."""
    return position(st, reckless=True, base=opponents.ai_like(st))[0]


# --- transfer: the situation in any battle (drills/transfer.py) -------------------------------------------------
T_TOWARD_MS = 0.5        # moving to a place: velocity towards it at least this


def _toward(u, x, z):
    dx, dz = x - u["x"], z - u["z"]
    n = torch.sqrt(dx * dx + dz * dz).clamp(min=1e-3)
    return (u["vx"] * dx + u["vz"] * dz) / n


T_STILL_MS = 0.5         # "moving": its speed at least this (a needless move: below)


def _cases(st):
    """(plan, go, retarget, unsafe, still) [B, N]: the decision's four cases - go to a better place, shoot a better
    target from here, the unsafe case (the reckless decision would go, the skilled one stays), and a needless move
    (it moves although the skilled decision is to stay and shoot - a target in range from here, no better place,
    no free enemy melee unit reaching it here: the kiting teacher of step s45 taught "run" wider than its situation)."""
    p = plan(st)
    u = st.u
    ev = p["ev"]
    moving = torch.sqrt(u["vx"] * u["vx"] + u["vz"] * u["vz"]) >= T_STILL_MS
    still = ev["who"] & ~p["go"] & ~p["unsafe_go"] & (ev["worth0"] > 0) & ~ev["unsafe"][:, :, 0] & moving
    return p, p["go"], p["retarget"] & ~p["unsafe_go"] & ~still, p["unsafe_go"], still


@torch.compiler.disable
def transfer(st):
    """(situation, applied, mistake) [B, N] for a direct-fire unit (standing, ammunition, out of melee, not a lord):
    situation = one of _cases: the skilled decision goes to a better place, or shoots a better target from here (a
    clear line, the flank, the pincer), or the reckless one would go to an unsafe place and the skilled one stays, or
    the unit moves where it should stand and shoot; applied = it does what the skilled decision says (moves towards its
    place at T_TOWARD_MS; aims at / is ordered at its target; in the unsafe case does not move towards the unsafe
    place; a needless move never is); mistake = it moves to the unsafe place, or moves needlessly."""
    p, go, retarget, unsafe, still = _cases(st)
    u = st.u
    to_go = _toward(u, p["x"], p["z"]) >= T_TOWARD_MS
    to_bad = _toward(u, p["x_r"], p["z_r"]) >= T_TOWARD_MS
    aimed = (u["aim_tgt"] == p["tgt"]) | (u["order_target"] == p["tgt"])
    sit = go | retarget | unsafe | still
    applied = (go & to_go) | (retarget & aimed) | (unsafe & ~to_bad)
    return sit, sit & applied, (unsafe & to_bad) | still


def moments(st, o):
    """[B, N] where the teacher's labels count outside the clean frames: the transfer situation - so the labels are
    "go" (a move) and "stay and shoot" (hold, or attack the best target from here) both: the unsafe case and a
    needless move are labelled to stand."""
    _, go, retarget, unsafe, still = _cases(st)
    return go | retarget | unsafe | still


DRILL = D.Drill("direct_fire", frame, enemy, naive, skilled,
                "direct-fire units: a clear line beside our own men, the pincer's flankers; stay out of a free enemy's way",
                transfer=transfer, embedded=embedded, teacher=teacher, moments=moments, reckless=reckless,
                transfer_ref=0.6, teach_cap=0.05, teach_stop=0.15, teach_stop_mistake=0.1, heavy=True)

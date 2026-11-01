"""The cavalry's blows by the game's memory (build/diag_mem/cav2; config/nn/sim.json impact_comb_why, charge_defence_why,
cycle_cavalry_why, wound_face_cavalry in wound_hp_why): the ranks behind's first blows alone in the comb
(comb_blows_rear), the front rank's men in reach after the lances (front_reach), the brace the ridden-in men keep
(brace_kept), the charge defence per blow by the struck man's facing (charge_defence crowd), a horseman's cycle (one
swing an interval, melee.cycle_cavalry) and the wounded pool of a unit a horseman strikes (kills.wound_face_cavalry).
Every expected number is worked out here from the rule. Needs torch (the container)."""
import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import battle, geometry, impact, melee, scenario  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

P = load()
B = P.battle
KNIGHTS = "wh_main_emp_cav_empire_knights"
CSPEAR, CLANRATS, SWORD = "wh2_main_skv_inf_clanrat_spearmen_0", "wh2_main_skv_inf_clanrats_1", "wh_main_emp_inf_swordsmen"
SP = P.sim["formation"]["spacing_m"]


def army(side1, side2):
    def units(rows):
        return [{"key": k, "x": x, "z": z, "b": b, "width": 30} for k, x, z, b in rows]
    return {"attacker": 1, "sides": {1: {"faction": "wh_main_emp_empire", "units": units(side1)},
                                     2: {"faction": "wh2_main_skv_skaven", "units": units(side2)}}}


def touching(striker, target, params=P):
    """The striker (side 1, slot 0, facing east) and the target (side 2, its first slot, facing west) in contact."""
    st = scenario.build([army([(striker, 0, 0, 90)], [(target, 60, 0, 270)])], params)
    H = st.N // 2
    front, depth = geometry.dims(st.u, SP)
    st.u["x"][0, H] = (float(depth[0, 0]) + float(depth[0, H])) / 2
    st.u["order_kind"][0, 0], st.u["order_target"][0, 0] = O.ATTACK, H
    return st, H


def strikes(st, params, charge=0.0, ran=None, first=False, parts=None):
    pw = geometry.pairwise(st.u, SP)
    ch = torch.zeros_like(st.u["men"])
    ch[0, 0] = charge
    kw = {"parts": parts} if parts is not None else {}
    return melee.strikes(st.u, pw, pw["enemy"], params, ch, first=first, ran=ran, **kw), pw


def wave(**kw):
    imp = dict(P.sim["contact"]["impact"])
    imp["wave"] = dict(imp.get("wave") or {}, **kw)
    return P.with_cal("contact", impact=imp)


def test_the_ranks_behind_strike_their_first_blow_alone_the_front_rank_its_cohorts():
    p = wave(comb=1, comb_blows=3, comb_blows_rear=1)
    st, H = touching(KNIGHTS, CLANRATS, p)
    u = st.u
    u["ride"][0, H] = 1.0
    I = float(u["interval"][0, 0])
    contact = torch.zeros(st.B, st.N, st.N, dtype=torch.bool)
    contact[0, 0, H] = True
    dt = 0.1
    rear, front = [], []
    for k in range(int(16 / dt)):
        u["kd_t"][0, H] = (k + 1) * dt
        _, r, f = impact.deferred(u, p, contact, dt)
        rear.append(float(r[0, 0, H]))
        front.append(float(f[0, 0, H]))
    # the ranks behind: one whole blow over their window, no second; the front rank: its blows at I and 2 I
    assert sum(rear) == pytest.approx(1.0, abs=1e-4)
    hits = [round((k + 1) * dt, 1) for k, x in enumerate(front) if x > 0]
    assert sum(front) == 2.0 and abs(hits[0] - I) <= dt and abs(hits[1] - 2 * I) <= dt
    # absent: comb_blows for both
    imp = dict(P.sim["contact"]["impact"], wave={"on": 1, "rise_s": 3.2, "comb": 1, "comb_blows": 3})
    assert impact.comb_blows_rear(P.with_cal("contact", impact=imp)) == 3


def test_the_ridden_in_men_keep_the_brace():
    st, H = touching(KNIGHTS, CSPEAR)
    u = st.u
    u["ride"][0, H] = 1.0
    u["kd_t"][0, H] = 20.0
    off = P.with_cal("contact", impact=dict(P.sim["contact"]["impact"], brace_kept=0))
    on = P.with_cal("contact", impact=dict(P.sim["contact"]["impact"], brace_kept=1))
    assert bool(impact.ridden(u, off)[0, H]) and not bool(impact.ridden(u, on)[0, H])
    assert impact.brace_kept(on) and not impact.brace_kept(off)


def _p_hit(st, H, charge):
    """The rule's chance and HP a hit of the knights' blow on the target man for a charge left (no reflection, no splash)."""
    u = st.u
    extra = float(u["charge_bonus"][0, 0]) * charge + float(u["bonus_v_inf"][0, 0])
    p = melee.hit_chance(torch.tensor(float(u["attack"][0, 0]) + extra), torch.tensor(float(u["defence"][0, H])),
                         B["melee_hit_chance_normalisation_coefficient"], B["melee_hit_chance_base"],
                         B["melee_hit_chance_min"], B["melee_hit_chance_max"], float(P.sim["melee"]["hit_roll_offset"]))
    dmg, ap = float(u["damage"][0, 0]), float(u["ap_damage"][0, 0])
    share = dmg / (dmg + ap)
    res = melee.resist_of(u, P.sim["melee"])
    res = res[0, 0, H] if res.shape[1] > 1 else res[0, 0, H]
    hit = melee.per_hit(torch.tensor(dmg + extra * share), torch.tensor(ap + extra * (1 - share)), u["armour"][0, H],
                        u["hp_man"][0, H], res)
    return float(p), float(hit)


def test_the_charge_defence_keeps_the_charge_of_the_crowd_s_blows_from_beside_the_man():
    q = 0.27
    st, H = touching(KNIGHTS, CSPEAR)
    base = P.with_cal("melee", cycle_cavalry={"on": 0})
    cd = dict(P.sim["contact"]["charge_defence"])
    cd.pop("crowd", None)
    p_off = base.with_cal("contact", charge_defence=cd)
    p_on = base.with_cal("contact", charge_defence=dict(cd, crowd=q))
    (r_off, hit_off, _, F), _ = strikes(st, p_off, charge=0.8)
    (r_on, hit_on, _, F2), _ = strikes(st, p_on, charge=0.8)
    p0, h0 = _p_hit(st, H, 0.0)
    p1, h1 = _p_hit(st, H, 0.8)
    # off: the braced spearmen in front cancel the whole charge; on: q of the blows keep it, the HP a hit by the hits
    assert float(hit_off[0, 0, H]) == pytest.approx(h0, rel=1e-4)
    w0, w1 = (1 - q) * p0, q * p1
    assert float(hit_on[0, 0, H]) == pytest.approx((w0 * h0 + w1 * h1) / (w0 + w1), rel=1e-4)
    # the rate: the mixed chance (the miss rule's per_s, cycle_cavalry off)
    miss = float(P.sim["melee"]["miss_s"])
    I = float(st.u["interval"][0, 0])
    pm = w0 + w1
    ratio = float(r_on[0, 0, H] / r_off[0, 0, H])
    assert ratio == pytest.approx((pm / (pm * I + miss) * float(hit_on[0, 0, H])) / (p0 / (p0 * I + miss) * h0), rel=1e-3)
    # the blow at the landing (ran) comes from the front: the charge cancelled
    ran = torch.zeros_like(st.u["r"])
    ran[0, 0] = True
    (_, hit_l, _, _), _ = strikes(st, p_on, charge=0.8, ran=ran)
    assert float(hit_l[0, 0, H]) == pytest.approx(h0, rel=1e-4)


def test_a_horseman_strikes_once_an_interval_the_face_share_of_his_knights():
    st, H = touching(KNIGHTS, CLANRATS)
    off = P.with_cal("melee", cycle_cavalry={"on": 0})
    ff = float(P.sim["melee"]["fighting_files"])
    for share in (ff, 0.72):
        on = P.with_cal("melee", cycle_cavalry={"on": 1, "face_share": share})
        (r_on, hit, _, F), _ = strikes(st, on)
        (r_off, hit2, _, F2), _ = strikes(st, off)
        p, _ = _p_hit(st, H, 0.0)
        I = float(st.u["interval"][0, 0])
        swings = float(r_on[0, 0, H]) / (float(F[0, 0, H]) * p * float(hit[0, 0, H]))
        # a swing an interval (less the search after a kill: busy <= 1) x the face share over fighting_files
        assert swings <= share / ff / I + 1e-6 and swings >= 0.9 * share / ff / I
        miss = float(P.sim["melee"]["miss_s"])
        assert float(r_off[0, 0, H]) / (float(F2[0, 0, H]) * p * float(hit2[0, 0, H])) == pytest.approx(1 / (p * I + miss), rel=1e-4)
    # the infantry's blows on the horse keep the miss rule
    (r1, _, _, _), _ = strikes(st, P.with_cal("melee", cycle_cavalry={"on": 1, "face_share": 0.72}))
    (r2, _, _, _), _ = strikes(st, off)
    assert float(r1[0, H, 0]) == pytest.approx(float(r2[0, H, 0]), rel=1e-6)


def test_the_front_rank_in_reach_after_the_lances_is_its_files_x_fighting_files():
    out = {}
    for fr in (0, 1):
        p = wave(on=1, rise_s=3.2, fall_s=2.0, comb=1, comb_blows=3, comb_blows_rear=1, front_reach=fr)
        st, H = touching(KNIGHTS, CLANRATS, p)
        u = st.u
        u["ride"][0, H] = 1.0
        u["kd_t"][0, H] = 3.0
        u["ran_in"][0, 0] = True
        u["contact_s"][0, 0] = 3.0
        if "melee_s" in u:
            u["melee_s"][0, 0] = 3.0
        parts = {}
        late = torch.zeros_like(u["r"])
        late[0, 0] = True
        pw = geometry.pairwise(u, SP)
        melee.strikes(u, pw, pw["enemy"], p, torch.zeros_like(u["men"]), first=True, late=late, parts=parts)
        out[fr] = (float(parts["s_front"][0, 0, H]), float(parts["s_rear"][0, 0, H]))
    ff = float(P.sim["melee"]["fighting_files"])
    assert out[0][0] > 0 and out[1][0] == pytest.approx(out[0][0] * ff, rel=1e-5)
    # the ranks behind's first blows: the same men past the front files
    assert out[1][1] == pytest.approx(out[0][1], rel=1e-6)


def test_the_wounded_pool_of_a_unit_struck_by_a_horseman_holds_its_men_at_the_face():
    st, H = touching(KNIGHTS, CLANRATS)
    u = st.u
    N = st.N
    hp = torch.zeros(1, N, N)
    hp[0, 0, H] = 100.0
    hit = torch.full((1, N, N), 30.0)
    fresh = torch.full((1, N, N), 47.0)
    F = torch.zeros(1, N, N)
    F[0, H, 0] = 10.0                                  # the target's own men striking the knights
    u["hp_abs"][0, H] = u["men"][0, H] * u["hp_man"][0, H] - 290.0      # W = 290 wounded HP
    h1, _ = melee.wound_hp(u, hp, hit, fresh, F)
    k = torch.ones_like(u["men"])
    k[0, H] = 2.0
    h2, _ = melee.wound_hp(u, hp, hit, fresh, F, own_k=k)
    # W* = own x (60 - 30): 300 by its 10 men (nearly full: little fresh HP), 600 by 20 (room: the whole fresh hits)
    D, R = 100.0, 100.0 * 47 / 30
    for own, h in ((10.0, h1), (20.0, h2)):
        gap = max(own * (60 - 30) - 290.0, 0.0)
        Dn = min(max(D + gap * (1 - D / R), D), R)
        assert float(h[0, 0, H]) == pytest.approx(Dn, rel=1e-5)
    assert float(h2[0, 0, H]) > float(h1[0, 0, H])


def test_a_battle_without_a_horseman_is_the_same_with_the_rules_on_or_off():
    imp = dict(P.sim["contact"]["impact"], brace_kept=0, wave={"on": 1, "rise_s": 1.1})
    cd = {k: v for k, v in P.sim["contact"]["charge_defence"].items() if k != "crowd"}
    off = P.with_cal("contact", impact=imp, charge_defence=cd).with_cal("melee", cycle_cavalry={"on": 0}) \
        .with_cal("kills", wound_face_cavalry=0)
    hp = {}
    for name, params in (("on", P), ("off", off)):
        st = scenario.build([army([(SWORD, 0, 0, 90)], [(CSPEAR, 40, 0, 270)])], params)
        H = st.N // 2
        o = O.hold(st.B, st.N)
        o.kind[0, 0], o.target[0, 0] = O.ATTACK, H
        for _ in range(60):
            battle.step(st, o, params, params.dt)
        hp[name] = st.u["hp_abs"].clone()
    assert torch.equal(hp["on"], hp["off"])

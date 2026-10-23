"""The line of fire past friends by the game's rules (missile.friend_los: tools/nn/sim/missile.py friend_lines,
friend_rays,
clear_shot; docs/en/game/mechanics/missiles.md "Fire on friends and the line of fire"). Needs torch (the container)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import battle, geometry, missile, replay, scenario  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

P = load()
OFF = P.with_cal("missile", friend_los=None)
HG, ARCHER = "wh_main_emp_inf_handgunners", "wh2_dlc13_emp_inf_archers_0"
XBOW, SLINGER = "wh_main_emp_inf_crossbowmen", "wh2_main_skv_inf_skavenslave_slingers_0"
SPEAR, SLAVES = "wh_main_emp_inf_spearmen_0", "wh2_main_skv_inf_skavenslave_spearmen_0"


def army(side1, side2):
    rows = (lambda rs: [{"key": r[0], "x": r[1], "z": r[2], "b": r[3], "general": False} for r in rs])
    return {"attacker": 1, "sides": {1: {"faction": "wh_main_emp_empire", "units": rows(side1)},
                                     2: {"faction": "wh2_main_skv_skaven", "units": rows(side2)}}}


def setup(own, d=100.0, params=P):
    st = scenario.build([army(own, [(SLAVES, -100 + d, 0, 270)])], params)
    pw = geometry.pairwise(st.u, params.sim["formation"]["spacing_m"])
    target = torch.full_like(st.u["side"], -1)
    target[0, :len(own)] = st.N // 2
    return st, pw, target


def lines(st, pw, target, contact=None):
    """(low, high, closed) [B, N]: the men's share on the low arc and on the 45 deg arc, the unit's line closed."""
    low = missile.friend_lines(st.u, pw, target, P, contact)
    high = missile.friend_lines(st.u, pw, target, P, contact, lob=torch.ones_like(target, dtype=torch.bool))
    return low, high, missile.friend_rays(st.u, pw, target, P)


def first_volley(own, d=100.0, params=P, steps=16):
    """Projectiles each own unit loosed in the first 8 s (its first volley), over the lone first unit's."""
    def spent(rows):
        st = scenario.build([army(rows, [(SLAVES, -100 + d, 0, 270)])], params)
        for _ in range(steps):
            battle.step(st, replay.hold(st), params)
        return [float(st.u["ammo0"][0, i] - st.u["a"][0, i]) for i in range(len(rows))], st
    lone = spent(own[:1])[0][0]
    shots, st = spent(own)
    return [s / lone for s in shots], st


def test_the_mod_values_friends_at_their_own_size():
    assert P.battle["projectile_friendly_fire_man_radius_coefficient"] == 1.0
    assert P.battle["projectile_friendly_fire_man_height_coefficient"] == 1.0
    assert P.sim["missile"]["friend_los"]["rays_clear"] == 2


def _clear(st, pw, target, params=P):
    info = {}
    tg, clear = missile.clear_shot(st.u, pw, target, torch.ones_like(st.u["side"], dtype=torch.bool),
                                   torch.full_like(target, -1), torch.zeros_like(target, dtype=torch.bool), params,
                                   info=info)
    return tg, clear, info


SCREEN = (lambda x: [(SPEAR, x, -15, 90), (SPEAR, x, 15, 90)])      # two of our spear blocks side by side, 59 m wide


def test_rule2_a_close_screen_closes_the_line_bullets_hold():
    """Our spearmen just ahead of the handgunners' front: the bullets' low arc passes below their heads on all three
    rays - the line is closed and the unit does not shoot at that target (no other in range: it holds)."""
    own = [(HG, -100, 0, 90)] + SCREEN(-92)
    st, pw, target = setup(own)
    low, high, closed = lines(st, pw, target)
    assert bool(closed[0, 0]) and float(low[0, 0]) > 0.5
    tg, clear, info = _clear(st, pw, target)
    assert int(tg[0, 0]) == -1 and bool(info["held"][0, 0]) and not bool(info["lob"][0, 0])
    assert first_volley(own)[0][0] == 0


def test_rule2_arrows_clear_a_screen_ahead_and_lob_when_their_low_arc_cannot():
    """Arrows rise over the screen's heads within ~1 m (a far target: a steep low arc): the line is open. A near target
    (16 m) gives a flat low arc that the screen closes: beyond 10 m they shoot at 45 deg, the whole volley."""
    far = [(ARCHER, -100, 0, 90)] + SCREEN(-89)
    st, pw, target = setup(far)
    low, _, closed = lines(st, pw, target)
    assert not bool(closed[0, 0]) and float(low[0, 0]) < 0.02
    st, pw, target = setup(far, d=16.0)
    low, high, closed = lines(st, pw, target)
    assert bool(closed[0, 0]) and float(high[0, 0]) < 0.02
    tg, clear, info = _clear(st, pw, target)
    assert int(tg[0, 0]) == st.N // 2 and bool(info["lob"][0, 0]) and float(clear[0, 0]) > 0.98
    # the same with bullets: no lob, they hold
    st, pw, target = setup([(HG, -100, 0, 90)] + SCREEN(-89), d=16.0)
    tg, _, info = _clear(st, pw, target)
    assert int(tg[0, 0]) == -1 and not bool(info["lob"][0, 0])


def test_rule2_arcing_fire_lobs_only_beyond_10_m(monkeypatch):
    """A closed line (forced here): arcing fire beyond lob_min_m (10 m) lobs, within it holds like direct fire."""
    monkeypatch.setattr(missile, "friend_rays", lambda u, pw, target, params: target >= 0)
    monkeypatch.setattr(missile, "friend_lines", lambda u, pw, target, params, contact=None, lob=None:
                        torch.where(lob, torch.zeros_like(u["men"]), torch.full_like(u["men"], 0.4)))
    for d, lobs in ((9.0, False), (11.0, True)):
        st, pw, target = setup([(ARCHER, -100, 0, 90)], d=d)
        tg, clear, info = _clear(st, pw, target)
        assert bool(info["lob"][0, 0]) == lobs and (int(tg[0, 0]) >= 0) == lobs
        if lobs:
            assert float(clear[0, 0]) == pytest.approx(1.0)        # the 45 deg arc's share (0 here), not the low's


def test_rule2_a_far_screen_the_arc_clears_and_enemies_never_block():
    """Friends 40 m ahead at a target 90 m off: the bullets' arc is over their heads (the game: full volleys)."""
    st, pw, target = setup([(HG, -100, 0, 90), (SPEAR, -60, 0, 90)], d=90.0)
    low, _, closed = lines(st, pw, target)
    assert not bool(closed[0, 0]) and float(low[0, 0]) < 0.05
    # an enemy unit in the way is no friend: nothing blocks
    st = scenario.build([army([(HG, -100, 0, 90)], [(SLAVES, -10, 0, 270), (SPEAR, -86, 0, 270)])], P)
    pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])
    target = torch.full_like(st.u["side"], -1)
    target[0, 0] = st.N // 2
    low, _, closed = lines(st, pw, target)
    assert float(low[0, 0]) == 0 and not bool(closed[0, 0])


def test_rule3_the_own_ranks_never_block_but_in_melee_with_the_target():
    st, pw, target = setup([(HG, -100, 0, 90)], d=60.0)
    low, _, closed = lines(st, pw, target)
    assert float(low[0, 0]) == 0 and not bool(closed[0, 0])
    contact = torch.zeros(st.B, st.N, st.N, dtype=torch.bool)
    contact[0, 0, st.N // 2] = True
    low_m, _, closed_m = lines(st, pw, target, contact=contact)
    assert float(low_m[0, 0]) > 0.3 and not bool(closed_m[0, 0])      # its ranks behind the first are in the way


def test_rule3_and_5_units_in_each_other_bullets_lose_the_men_behind_the_others_men():
    """Two handgunner units on one place: neither line is closed (the rays start at the front), but the men whose
    bullet passes through the other unit's men skip - the front rank fires, the ranks behind lose most."""
    pair = [(HG, -100, 0, 90), (HG, -100, 0, 90)]
    st, pw, target = setup(pair, d=90.0)
    low, _, closed = lines(st, pw, target)
    assert not bool(closed[0, 0]) and not bool(closed[0, 1])
    assert 0.2 < float(low[0, 0]) < 0.9
    share, _ = first_volley(pair, d=90.0)
    old, _ = first_volley(pair, d=90.0, params=OFF)
    assert share[0] < 0.75 and share[1] < 0.75 and old[0] > 0.95
    # arrows rise over the other unit's heads within ~1 m: few of them are lost
    arch, _ = first_volley([(ARCHER, -100, 0, 90), (ARCHER, -100, 0, 90)], d=100.0)
    assert arch[0] > 0.75 and arch[0] > share[0] + 0.15


def test_rule4_a_friend_hit_takes_the_full_damage():
    """The bullets that pass below a screen's heads hit it with the rule's damage, no cut (can_damage_allies)."""
    st, pw, target = setup([(HG, -100, 0, 90), (SPEAR, -60, 0, 90)], d=90.0)
    u = st.u
    tg, clear, catch = missile.clear_shot(u, pw, target, torch.ones_like(u["side"], dtype=torch.bool),
                                          torch.full_like(target, -1), torch.zeros_like(target, dtype=torch.bool), P,
                                          with_catch=True)
    loaded = torch.ones_like(u["men"])
    shots, hp, hit = missile.volley(u, pw, tg, P.dt, P, clear=clear, loaded=loaded, catch=catch)
    caught = float(catch[0, 0, 1] * shots[0, 0])
    assert caught > 0
    # the screen faces away from the shooters: no shield; HP = the friends' hits x the rule's HP a hit
    assert float(hp[0, 0, 1]) == pytest.approx(caught * float(hit[0, 0, 1]), rel=1e-4)


def test_the_bookkeeping_and_the_behaviour_shares():
    from tools.nn.train import behaviour
    pair = [(HG, -100, 0, 90), (HG, -100, 0, 90)]
    st = scenario.build([army(pair, [(SLAVES, -10, 0, 270)])], P)
    watch = behaviour.Tracker(st, P, st.u["side"] == 1, lively=False)
    live = torch.ones(1, dtype=torch.bool)
    for _ in range(16):
        battle.step(st, replay.hold(st), P)
        watch.update(st, live)
    m = watch.summary(np.array([True]))
    assert 0.2 < m["friend_skip_share_direct"] < 0.9
    assert m["friend_skip_share_arc"] is None and m["forced_lob_share_arc"] is None
    s = {k: float(v[0]) for k, v in watch.sums.items()}
    assert s["fc_cap_d"] > 0 and s["fc_lob_d"] == 0


def test_rule2_two_of_three_rays_one_closed_ray_leaves_the_line_open():
    """A screen just ahead of the handgunners covering only the end of their front closes one ray: the line stays open
    (the covered men skip their shots); a screen covering the end and the middle closes two: the line is closed."""
    def rays_shut(z):
        st, pw, target = setup([(HG, -100, 0, 90), (SPEAR, -92, z, 90)])
        u = st.u
        closed = missile.friend_rays(u, pw, target, P)
        low = missile.friend_lines(u, pw, target, P)
        return bool(closed[0, 0]), float(low[0, 0]), st, pw, target
    one, low1, st, pw, target = rays_shut(-25.0)
    assert not one and low1 > 0.1
    tg, clear, info = _clear(st, pw, target)
    assert int(tg[0, 0]) == st.N // 2 and not bool(info["held"][0, 0]) and float(clear[0, 0]) < 0.95
    two, _, st, pw, target = rays_shut(-10.0)
    assert two
    tg, _, info = _clear(st, pw, target)
    assert int(tg[0, 0]) == -1 and bool(info["held"][0, 0])


def test_rule2_a_closed_line_moves_the_fire_to_the_next_target_in_range():
    """Handgunners whose line to the enemy ahead our screen closes take another enemy in range with a clear line."""
    st = scenario.build([army([(HG, -100, 0, 90)] + SCREEN(-92), [(SLAVES, 0, 0, 270), (SLAVES, -100, 80, 180)])], P)
    pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])
    H = st.N // 2
    target = torch.full_like(st.u["side"], -1)
    target[0, 0] = H
    assert bool(missile.friend_rays(st.u, pw, target, P)[0, 0])
    tg, clear, info = _clear(st, pw, target)
    assert int(tg[0, 0]) == H + 1 and not bool(info["held"][0, 0]) and float(clear[0, 0]) > 0.95


@pytest.mark.parametrize("key", [SLINGER, XBOW])
def test_rule2_slings_and_crossbows_lob_like_bows(key):
    """Slings and crossbows (trajectory dual_low_fixed) are arcing fire: a closed line at a target beyond 10 m - the
    whole volley at 45 deg; their passports are not direct fire."""
    own = [(key, -100, 0, 90)] + SCREEN(-89)
    st, pw, target = setup(own, d=16.0)
    assert not bool(st.u["direct"][0, 0])
    assert bool(missile.friend_rays(st.u, pw, target, P)[0, 0])
    tg, clear, info = _clear(st, pw, target)
    assert int(tg[0, 0]) == st.N // 2 and bool(info["lob"][0, 0]) and float(clear[0, 0]) > 0.98

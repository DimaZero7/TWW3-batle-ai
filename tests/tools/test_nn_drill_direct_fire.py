"""tools/nn/train/drills/direct_fire.py: the embedded frame (a normal battle with our direct-fire units deployed by
the generator's rule, the variants' tagged enemy units, even gold), the skill's evaluation on hand-made scenes (a
militia unit behind its own spearmen is blocked and goes beside them; a place a free enemy reaches first is unsafe:
skilled stays, reckless goes; arcing shooters are never given orders; the pincer's flankers weigh more), the enemy's
tagged units (envelop, hunt), the transfer detector, and the scripts in the simulator."""
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import battle, scenario
from tools.nn.sim import orders as O
from tools.nn.sim.params import load
from tools.nn.train import drills as D
from tools.nn.train.drills import direct_fire as F

UNITS = json.loads((Path(__file__).resolve().parents[2] / "config/nn/units.json").read_text(encoding="utf-8"))["units"]
MILITIA, ARCHERS = "wh_dlc04_emp_inf_free_company_militia_0", "wh2_dlc13_emp_inf_archers_0"
SLINGERS = "wh2_main_skv_inf_skavenslave_slingers_0"
SPEAR, SWORD, CLANRAT = "wh_main_emp_inf_spearmen_0", "wh_main_emp_inf_swordsmen", "wh2_main_skv_inf_clanrats_0"


def _scene(ours, enemy, attacker=1):
    st = scenario.build([D.army(ours, enemy, attacker, F.EMPIRE, F.SKAVEN)])
    return st


def _slot(st, x, z):
    d = (st.u["x"][0] - x) ** 2 + (st.u["z"][0] - z) ** 2
    d = torch.where(st.u["side"][0] > 0, d, torch.full_like(d, 1e9))
    return int(d.argmin())


def _blocked_scene(shooter=MILITIA, hunters=False):
    """Our spearmen at the origin facing +z, our shooter 20 m behind them, the enemy's spearmen 40 m ahead of ours
    (60 m from the shooter, in range); with hunters: Skaven clanrats (run 4.2) 135 m to both sides of the shooter
    (out of reach), nearer to its side places than to it, and the target is the enemy's slave slingers."""
    ours = [D.unit(SPEAR, 0, 0, 0, 30), D.unit(shooter, 0, -20, 0, 30)]
    # with hunters the target is a missile unit (slave slingers): not a free melee unit, so staying is safe
    enemy = [D.unit(SLINGERS if hunters else SPEAR, 0, 40, 180, 30)]
    if hunters:
        enemy += [D.unit(CLANRAT, 135, -20, 270, 30), D.unit(CLANRAT, -135, -20, 90, 30)]
    return _scene(ours, enemy)


class TestFrame:
    def test_a_normal_battle_with_our_direct_fire_units_tagged(self):
        seen_variants, factions = set(), set()
        direct = {k for k, v in UNITS.items() if (v.get("missile") or {}).get("direct")}
        for desc, ours in D.battles(F.DRILL, range(24), embed=1.0):
            assert desc["frame"] == "embedded"
            mine, theirs = desc["sides"][ours]["units"], desc["sides"][3 - ours]["units"]
            assert any(u.get("general") for u in mine) and any(u.get("general") for u in theirs)
            assert any(u.get("tag") == D.TAG_OURS for u in mine)
            for u in mine:                      # every direct-fire unit of ours tagged, nothing else
                assert (u.get("tag") == D.TAG_OURS) == (u["key"] in direct)
            meta = desc["meta"]
            seen_variants.add(meta["variant"])
            factions.add(desc["sides"][ours]["faction"])
            tags = [u.get("tag", 0) for u in theirs]
            want = {"plain": 0, "pincer": D.TAG_ENEMY, "hunter": F.TAG_HUNTER}[meta["variant"]]
            if want:
                assert tags.count(want) == len(meta["extra"]) >= 1
            assert all(t in (0, want) for t in tags)
            assert len(mine) <= D.MAX_UNITS + 1 and len(theirs) <= D.MAX_UNITS + 1
            assert D.imbalance(desc) <= D.EMBED_TOLERANCE + 0.05
        assert seen_variants == {"plain", "pincer", "hunter"}
        assert factions == {F.EMPIRE, F.SKAVEN}

    def test_our_direct_fire_units_stand_behind_the_melee_line(self):
        for desc, ours in D.battles(F.DRILL, range(8), embed=1.0):
            ax = D.axes(desc, ours)
            mine = desc["sides"][ours]["units"]
            line = [u for u in mine if not u.get("general") and not UNITS[u["key"]].get("missile")]
            if not line:                        # an army of shooters only: no melee line to stand behind
                continue
            front = D.front(line, ax)
            for u in mine:
                if u.get("tag") == D.TAG_OURS:
                    assert ax.local(u["x"], u["z"])[0] < front - 5


class TestSkill:
    def test_a_shooter_behind_its_own_men_is_blocked_and_goes_beside_them(self):
        st = _blocked_scene()
        i = _slot(st, 0, -20)
        p = F.plan(st)
        ev = p["ev"]
        assert bool(ev["who"][0, i])
        assert float(ev["worth0"][0, i]) < 0.2 * float(ev["score"][0, i].max() / F.HORIZON_S)
        assert bool(p["go"][0, i]) and int(p["place"][0, i]) != 0
        assert abs(float(p["x"][0, i])) >= 20                  # across the line, beside the spearmen
        o, given = F.position(st)
        assert bool(given[0, i]) and int(o.kind[0, i]) == O.MOVE and bool(o.run[0, i])

    def test_unsafe_places_skilled_stays_reckless_goes(self, monkeypatch):
        monkeypatch.setattr(F, "SAFE_S", 25.0)              # side places 90-110 m from a hunter, here 135 m
        F._CACHE.clear()
        st = _blocked_scene(hunters=True)
        i = _slot(st, 0, -20)
        p = F.plan(st)
        assert bool(p["ev"]["unsafe"][0, i, 1:].all()) and not bool(p["ev"]["unsafe"][0, i, 0])
        assert not bool(p["go"][0, i]) and bool(p["go_r"][0, i]) and bool(p["unsafe_go"][0, i])
        o, _ = F.position(st)
        assert int(o.kind[0, i]) != O.MOVE
        o, _ = F.position(st, reckless=True)
        assert int(o.kind[0, i]) == O.MOVE
        F._CACHE.clear()

    def test_arcing_shooters_are_never_given_orders(self):
        st = _blocked_scene(ARCHERS)
        i = _slot(st, 0, -20)
        p = F.plan(st)
        assert not bool(p["ev"]["who"][0, i]) and not bool(p["go"][0, i] | p["retarget"][0, i] | p["unsafe_go"][0, i])
        _, given = F.position(st)
        assert not bool(given.any())
        sit, _, _ = F.transfer(st)
        assert not bool(sit.any())

    def test_an_enemy_at_our_units_flank_is_a_flanker(self):
        ours = [D.unit(SPEAR, 0, 0, 0, 30), D.unit(MILITIA, -60, -30, 0, 30)]
        enemy = [D.unit(SWORD, 30, 5, 270, 30), D.unit(SWORD, 0, 120, 180, 30)]
        st = _scene(ours, enemy)
        prm = load()
        from tools.nn.sim import geometry
        u = st.u
        pw = geometry.pairwise(u, prm.sim["formation"]["spacing_m"])
        standing = (u["side"] > 0) & (u["men"] > 0)
        fl = F._flankers(u, pw, standing)[0]
        assert bool(fl[_slot(st, 30, 5)]) and not bool(fl[_slot(st, 0, 120)])


class TestEnemy:
    def test_a_flanker_in_front_goes_round_then_attacks_at_the_flank(self):
        ours = [D.unit(SPEAR, 0, 0, 0, 30)]
        enemy = [D.unit(SWORD, 5, 100, 180, 30, tag=D.TAG_ENEMY)]
        st = _scene(ours, enemy)
        f, k = _slot(st, 5, 100), _slot(st, 0, 0)
        o = F.envelop(st, O.hold(st.B, st.N), D.tagged(st, D.TAG_ENEMY))
        assert int(o.kind[0, f]) == O.MOVE and float(o.x[0, f]) > 15 + F.ENV_WAY_M - 1
        enemy = [D.unit(SWORD, 15 + F.ENV_OUT_M, -F.ENV_BACK_M, 270, 30, tag=D.TAG_ENEMY)]
        st = _scene(ours, enemy)
        f = _slot(st, 15 + F.ENV_OUT_M, -F.ENV_BACK_M)
        o = F.envelop(st, O.hold(st.B, st.N), D.tagged(st, D.TAG_ENEMY))
        assert int(o.kind[0, f]) == O.ATTACK and int(o.target[0, f]) == k

    def test_a_hunter_attacks_our_direct_fire_unit_near_else_holds(self):
        ours = [D.unit(MILITIA, 0, 30, 0, 30), D.unit(SPEAR, 300, 0, 0, 30)]
        enemy = [D.unit(CLANRAT, 0, 100, 180, 30, tag=F.TAG_HUNTER)]
        st = _scene(ours, enemy)
        h, m = _slot(st, 0, 100), _slot(st, 0, 30)
        o = F.hunt(st, O.hold(st.B, st.N), D.tagged(st, F.TAG_HUNTER))
        assert int(o.kind[0, h]) == O.ATTACK and int(o.target[0, h]) == m
        ours = [D.unit(MILITIA, 0, -100, 0, 30), D.unit(SPEAR, 300, 0, 0, 30)]
        st = _scene(ours, enemy)
        o = F.hunt(st, opponents_ai(st), D.tagged(st, F.TAG_HUNTER))
        assert int(o.kind[0, _slot(st, 0, 100)]) == O.HOLD


def opponents_ai(st):
    from tools.nn.train import opponents
    return opponents.ai_like(st)


class TestTransfer:
    def test_moving_to_the_place_is_applied_standing_is_not(self):
        F._CACHE.clear()
        st = _blocked_scene()
        i = _slot(st, 0, -20)
        sit, applied, mistake = F.transfer(st)
        assert bool(sit[0, i]) and not bool(applied[0, i]) and not bool(mistake[0, i])
        p = F.plan(st)
        dx, dz = float(p["x"][0, i]) - 0.0, float(p["z"][0, i]) + 20.0
        n = (dx * dx + dz * dz) ** 0.5
        st.u["vx"][0, i], st.u["vz"][0, i] = 3.0 * dx / n, 3.0 * dz / n
        F._CACHE.clear()
        sit, applied, mistake = F.transfer(st)
        assert bool(sit[0, i]) and bool(applied[0, i])
        F._CACHE.clear()

    def test_moving_to_an_unsafe_place_is_the_mistake(self, monkeypatch):
        monkeypatch.setattr(F, "SAFE_S", 25.0)
        F._CACHE.clear()
        st = _blocked_scene(hunters=True)
        i = _slot(st, 0, -20)
        p = F.plan(st)
        dx, dz = float(p["x_r"][0, i]), float(p["z_r"][0, i]) + 20.0
        n = (dx * dx + dz * dz) ** 0.5
        st.u["vx"][0, i], st.u["vz"][0, i] = 3.0 * dx / n, 3.0 * dz / n
        F._CACHE.clear()
        sit, applied, mistake = F.transfer(st)
        assert bool(sit[0, i]) and bool(mistake[0, i]) and not bool(applied[0, i])
        F._CACHE.clear()


    def test_a_needless_move_is_a_mistake_and_labelled_to_stand(self):
        F._CACHE.clear()
        st = _scene([D.unit(MILITIA, 0, 0, 0, 30)], [D.unit(SLINGERS, 0, 60, 180, 35)])
        i = _slot(st, 0, 0)
        sit, applied, mistake = F.transfer(st)
        assert not bool(sit[0, i])                         # a clear line, nothing better: no situation
        st.u["vx"][0, i] = 3.0
        F._CACHE.clear()
        sit, applied, mistake = F.transfer(st)
        assert bool(sit[0, i]) and bool(mistake[0, i]) and not bool(applied[0, i])
        assert bool(F.moments(st, F.teacher(st))[0, i]) and int(F.teacher(st).kind[0, i]) in (O.HOLD, O.ATTACK)
        F._CACHE.clear()


class TestRun:
    def test_the_scripts_play_embedded_battles_in_the_simulator(self):
        pairs = D.battles(F.DRILL, range(4), embed=1.0)
        st = scenario.build([p[0] for p in pairs])
        ours = torch.tensor([p[1] for p in pairs])
        params = load()
        for script in (F.naive, F.skilled, F.reckless, F.teacher):
            for _ in range(6):
                o = D.merged(st, ours, script(st), F.enemy(st))
                O.check(o, st.N)
                battle.step(st, o, params, params.dt)
        m = F.moments(st, F.teacher(st))
        assert m.shape == st.u["x"].shape


def test_the_drill_is_loaded_and_evaluated_only_when_a_step_asks():
    assert "direct_fire" in D.NAMES and "direct_fire" not in D.READY and "direct_fire" not in D.TRAIN
    assert D.load(["direct_fire"])["direct_fire"].reckless is F.reckless
    try:
        assert "direct_fire" not in D.evaluated()
        assert D.set_extra(["direct_fire", "kiting", "nope"]) == ["direct_fire"]
        assert D.evaluated()[-1] == "direct_fire" and set(D.READY) <= set(D.evaluated())
    finally:
        D.set_extra([])


class TestHeavy:
    def test_the_transfer_detector_only_while_heavy_is_on(self):
        from tools.nn.train.drills import transfer as XF
        try:
            D.set_extra(["direct_fire"])
            assert "direct_fire" not in XF.detectors()
            D.HEAVY_ON[0] = True
            assert "direct_fire" in XF.detectors()
            D.HEAVY_ON[0] = False
            assert "direct_fire" in XF.detectors(["direct_fire"])      # asked by name
        finally:
            D.HEAVY_ON[0] = False
            D.set_extra([])

    def test_the_teacher_runs_only_on_the_battles_it_may_label(self):
        import numpy as np
        from tools.nn.train import league, randomise, rollout, scenes
        B = 6
        lay = league.Layout(np.zeros(B, dtype=int), 1 + np.arange(B) % 2, np.array([league.CODE["nearest"]] * B))
        src = scenes.Generated(np.arange(B), 6, None, "cpu", seed=1)
        env = rollout.Battles(lay, device="cpu", source=src, compile=False, spread=randomise.NONE,
                              teach_normal={"direct_fire": F.DRILL})
        assert env.teach_heavy == (True,)
        i = env.teach_normal[0][0]
        seen = []

        def spy(st):
            seen.append(st.B)
            return O.hold(st.B, st.N, st.device)

        b = env.rows_learn % env.B
        live = ~env.st.done[b]
        env.teach_draw = torch.tensor([0.01, 0.9, 0.01, 0.9, 0.9, 0.9])
        env.set_teach_shares({D.normal_name("direct_fire"): 0.05})
        o, mom = env._heavy_teacher(i, spy, lambda st, o: torch.ones_like(st.u["tag"], dtype=torch.bool), b, live)
        assert seen == [2] and bool(mom[0].any()) and bool(mom[2].any()) and not bool(mom[1].any())
        env.set_teach_shares({D.normal_name("direct_fire"): 0.0})
        assert env._heavy_teacher(i, spy, None, b, live) == (None, None) and seen == [2]

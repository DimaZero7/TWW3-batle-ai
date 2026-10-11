"""The new units of 11.10.2026 in the simulator (docs/en/training/simulator.md "Contact effects", "Bursts"): the Ratling
Gun's burst of 18 bullets a man, its ammunition pool and its fire in melee; contact effects (poison, the censer's
contagion, suppressing fire) laid by hits, 10 s from the last hit, stacking, not by friendly fire. Needs torch."""
import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import battle, effects, geometry, replay, scenario  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

P = load()
RATLING, MONKS, CENSER = ("wh2_dlc12_skv_inf_ratling_gun_0", "wh2_main_skv_inf_plague_monks",
                          "wh2_main_skv_inf_plague_monk_censer_bearer")
WOLF, WOLFP, HALB = "wh2_dlc16_skv_mon_wolf_rats_0", "wh2_dlc16_skv_mon_wolf_rats_1", "wh_main_emp_inf_halberdiers"
SWORD, GENERAL = "wh_main_emp_inf_swordsmen", "wh_main_emp_cha_general_0"
POISON, CONTAGION, SUPPRESS = ("wh_main_unit_contact_poison", "wh2_main_unit_contact_plague_monk_censer_morale",
                               "wh2_dlc12_unit_contact_suppressive_fire")


def battle_of(side1, side2):
    """side1 / side2: [(key, x, z, bearing[, men])]; the Empire against the Skaven."""
    def units(rows):
        return [dict({"key": r[0], "x": r[1], "z": r[2], "b": r[3], "general": r[0] == GENERAL},
                     **({"men": r[4]} if len(r) > 4 else {})) for r in rows]
    # (the skirmish mode off: the gun stands, as the probes' shooters)
    return scenario.build([{"attacker": 1, "sides": {
        1: {"faction": "wh_main_emp_empire", "units": units(side1), "skirmish": False},
        2: {"faction": "wh2_main_skv_skaven", "units": units(side2), "skirmish": False}}}], P)


def slot_of(key):
    return {k: j for j, i, k in effects.contact_keys(P)}[key]


def gun_on(target_key=GENERAL, d=100.0, men=None):
    """The Ratling Gun (side 2) at (0, 0) facing south, a target of side 1 d m south of it."""
    gun = (RATLING, 0.0, 0.0, 180.0) + ((men,) if men else ())
    return battle_of([(target_key, 0.0, -d, 0.0)], [gun])


def fire_log(st, seconds):
    """[(t, bullets fired by the gun this step)] under HOLD (the gun fires at will)."""
    H = st.N // 2
    out = []
    for k in range(int(round(seconds / P.dt))):
        a0 = float(st.u["a"][0, H])
        battle.step(st, replay.hold(st), P)
        out.append(((k + 1) * P.dt, a0 - float(st.u["a"][0, H])))
    return out


class TestPassports:
    def test_the_new_units_carry_their_database_numbers(self):
        u = P.units
        m = u[RATLING]["missile"]
        assert (u[RATLING]["men"], m["shots_per_volley"], m["ammo"], m["reload_s"], m["damage"], m["ap_damage"]) == \
            (32, 18, 270, 5.0, 2, 6)
        assert m["contact_phase"] == SUPPRESS and u[CENSER]["melee"]["contact_phase"] == CONTAGION
        assert u[WOLFP]["melee"]["contact_phase"] == POISON and u[WOLF]["melee"]["contact_phase"] is None
        assert u[WOLF]["caste"] == "war_beast" and u[WOLF]["speed"]["run"] == 9.5 and u[WOLF]["speed"]["charge"] == 11.5
        assert u[CENSER]["melee"]["magic"] and u[WOLF]["melee"]["magic"] and not u[WOLFP]["melee"]["magic"]

    def test_the_gun_bursts_and_the_others_keep_their_volley(self):
        g = P.static(RATLING)
        assert (g["spv"], g["bst_rate"], g["reload"], g["ammo0"], g["melee_fire"]) == (18.0, 4.0, 12.0, 8640, True)
        assert P.static(SWORD)["spv"] == 1.0 and P.static("wh_main_emp_inf_handgunners")["bst_rate"] == 0.0
        assert not P.static("wh_main_emp_inf_handgunners")["melee_fire"]

    def test_the_weapons_lay_their_contact_effects(self):
        idx = effects.index(P)
        assert P.static(CENSER)["mfx"] == idx[CONTAGION] and P.static(WOLFP)["mfx"] == idx[POISON]
        assert P.static(RATLING)["sfx"] == idx[SUPPRESS] and P.static(RATLING)["mfx"] == -1
        assert P.static(SWORD)["mfx"] == P.static(SWORD)["sfx"] == -1


class TestBurst:
    def test_a_burst_is_18_bullets_a_man_at_4_a_second_every_12_s(self):
        log = fire_log(gun_on(GENERAL, 100.0), 40.0)
        shots = [(t, n) for t, n in log if n > 1e-6]
        assert shots, "the gun never fired"
        assert max(n for _, n in shots) <= 32 * 4 * P.dt + 1e-3          # 64 bullets a step at most
        starts = [t for (t, n), (_, prev) in zip(log, [(0, 0.0)] + log) if n > 1e-6 and prev <= 1e-6]
        first = sum(n for t, n in log if starts[0] <= t < starts[0] + 6.0)
        assert first == pytest.approx(32 * 18, rel=0.01)                  # 576: the lone General's lane in the game
        assert starts[1] - starts[0] == pytest.approx(12.0, abs=P.dt)      # the measured cycle

    def test_the_ammunition_is_one_pool_of_the_unit(self):
        st = gun_on(GENERAL, 100.0, men=16)
        H = st.N // 2
        assert float(st.u["a"][0, H]) == 8640                             # half the crew killed: the same pool
        log = fire_log(st, 12.0)
        assert sum(n for _, n in log) == pytest.approx(16 * 18, rel=0.01)  # a burst of the men left

    def test_the_gun_fires_in_melee(self):
        st = battle_of([(SWORD, 0.0, -40.0, 0.0)], [(RATLING, 0.0, 0.0, 180.0)])
        H = st.N // 2
        orders = O.hold(st.B, st.N, st.device)
        orders.kind[0, 0], orders.target[0, 0], orders.run[0, 0] = O.ATTACK, H, True
        k = 0
        while not bool(st.u["m"][0, H]) and k < 60:
            battle.step(st, orders, P)
            k += 1
        assert bool(st.u["m"][0, H])
        a0 = float(st.u["a"][0, H])
        for _ in range(30):
            battle.step(st, orders, P)
        assert bool(st.u["m"][0, H]) and float(st.u["a"][0, H]) < a0


class TestContactEffects:
    def test_suppressing_fire_lays_x07_speed_on_the_target(self):
        st = gun_on(SWORD, 80.0)
        j = slot_of(SUPPRESS)
        for _ in range(40):
            battle.step(st, replay.hold(st), P)
        assert float(st.u[f"cfx{j}_s"][0, 0]) > 0
        assert int(st.u["fx_on"][0, 0]) >> effects.index(P)[SUPPRESS] & 1
        c = effects.card(st.u, P)
        assert float(c["speed"][0, 0]) == pytest.approx(0.7)

    def test_a_contact_effect_lasts_10_s_from_the_last_hit(self):
        st = battle_of([(SWORD, 0.0, -80.0, 0.0)], [(CENSER, 0.0, 0.0, 180.0)])
        j = slot_of(CONTAGION)
        st.u[f"cfx{j}_s"][0, 0] = 10.0
        n = 0
        while float(st.u[f"cfx{j}_s"][0, 0]) > 0:
            battle.step(st, replay.hold(st), P)
            n += 1
        assert n * P.dt == pytest.approx(10.0)

    def test_the_censer_lays_its_contagion_in_melee_and_the_card_shows_minus_10(self):
        st = battle_of([(SWORD, 0.0, -14.0, 0.0)], [(CENSER, 0.0, 0.0, 180.0)])
        H = st.N // 2
        j = slot_of(CONTAGION)
        orders = O.hold(st.B, st.N, st.device)
        orders.kind[0, H], orders.target[0, H], orders.run[0, H] = O.ATTACK, 0, True
        for _ in range(20):
            battle.step(st, orders, P)
        assert float(st.u[f"cfx{j}_s"][0, 0]) > 9.0 and float(st.u[f"cfx{j}_s"][0, H]) == 0
        assert float(effects.card(st.u, P)["leadership"][0, 0]) == -10.0

    def test_poison_weakens_speed_and_both_damages_and_effects_stack(self):
        st = battle_of([("wh_main_emp_inf_handgunners", 0.0, -80.0, 0.0)], [(WOLFP, 0.0, 0.0, 180.0)])
        u = st.u
        u[f"cfx{slot_of(POISON)}_s"][0, 0] = 5.0
        pw = geometry.pairwise(u, P.sim["formation"]["spacing_m"])
        same = u["side"][:, :, None] == u["side"][:, None, :]
        standing = (u["side"] > 0) & ~u["r"]
        base = {k: float(u[k][0, 0]) for k in ("run", "damage", "ap_damage", "m_damage", "m_ap")}
        old = effects.apply(u, P, 0.5, standing, torch.zeros_like(standing), pw["dist"], same)
        for k, v in base.items():
            assert float(u[k][0, 0]) == pytest.approx(0.85 * v), k
        effects.restore(u, old)
        u[f"cfx{slot_of(SUPPRESS)}_s"][0, 0] = 5.0                         # another one stacks
        old = effects.apply(u, P, 0.5, standing, torch.zeros_like(standing), pw["dist"], same)
        assert float(u["run"][0, 0]) == pytest.approx(0.85 * 0.7 * base["run"])
        effects.restore(u, old)
        assert float(u["run"][0, 0]) == pytest.approx(base["run"])

    def test_friendly_fire_and_misses_lay_nothing(self):
        st = gun_on(SWORD, 80.0)
        u = st.u
        H = st.N // 2
        N = st.N
        hp = torch.zeros(1, N, N)
        hp[0, H, 0] = 50.0                                                 # the gun hits the swordsmen
        hit = torch.full((1, N, N), 5.0)
        j = slot_of(SUPPRESS)
        effects.lay(u, P, 0.5, torch.zeros_like(hp), hit, hp, hit, torch.zeros_like(hp), torch.zeros(1))
        assert float(u[f"cfx{j}_s"][0, 0]) == 0                            # not an enemy: nothing
        effects.lay(u, P, 0.5, torch.zeros_like(hp), hit, torch.zeros_like(hp), hit, torch.ones_like(hp), torch.zeros(1))
        assert float(u[f"cfx{j}_s"][0, 0]) == 0                            # no hit: nothing
        effects.lay(u, P, 0.5, torch.zeros_like(hp), hit, hp, hit, torch.ones_like(hp), torch.zeros(1))
        assert float(u[f"cfx{j}_s"][0, 0]) == 10.0                         # 10 hits: on for 10 s

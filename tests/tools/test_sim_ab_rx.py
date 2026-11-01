"""The 'effects from others' input (ab_rx, tools/nn/sim/abilities.py; config/nn/sim.json abilities.enemies_by_targets):
an ability reaches enemies only when it targets them - the enemies within 35 m of a Warlord's Rally do not see it on
themselves (the game's card shows a phase only on a unit it lies on: tools/nn/companion/exchange.py received), his
friends do. Needs torch (the container)."""
import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import abilities, geometry, scenario  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

P = load()
LORD, CLAN, SWORD = "wh2_main_skv_cha_warlord_0", "wh2_main_skv_inf_clanrats_0", "wh_main_emp_inf_swordsmen"
RALLY = "wh_main_character_abilities_rally"


def test_rally_shows_on_friends_not_on_the_enemies_near():
    a = {"attacker": 2, "sides": {
        1: {"faction": "wh_main_emp_empire", "units": [{"key": SWORD, "x": 0, "z": 0, "b": 0}]},
        2: {"faction": "wh2_main_skv_skaven", "units": [{"key": LORD, "x": 0, "z": 20, "b": 180},
                                                       {"key": CLAN, "x": 15, "z": 25, "b": 180}]}}}
    bit = 1 << abilities.keys(P).index(RALLY)
    k = abilities.slot_keys(LORD, P.units, P.abilities).index(RALLY)
    for on, enemy_sees in ((1, False), (0, True)):
        p = P.with_cal("abilities", enemies_by_targets=on)
        st = scenario.build([a], p)
        u, H = st.u, st.N // 2
        u[f"ab{k}_on"][0, H] = 10.0                              # Rally active
        pw = geometry.pairwise(u, p.sim["formation"]["spacing_m"])
        same = u["side"][:, :, None] == u["side"][:, None, :]
        alive = (u["side"] > 0) & (u["men"] > 0)
        old = abilities.apply(u, p, 0.5, alive, torch.zeros_like(alive), pw["dist"], same)
        assert bool(int(u["ab_rx"][0, H + 1]) & bit)             # the friend within 35 m: on him
        assert bool(int(u["ab_rx"][0, 0]) & bit) == enemy_sees   # the enemy 20 m off: on him only with the old bug
        abilities.restore(u, old)

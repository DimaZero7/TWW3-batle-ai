"""tools.nn.sim without torch: the state and order layouts (the contract with the network) and the
calibration file. The mechanics tests need torch: tests/tools/test_sim.py (run them in the container)."""
import json

from tools.nn import gamedata
from tools.nn.sim import orders, params, state


class TestLayout:
    def test_observed_fields_are_the_recorded_ones(self):
        recorded = set(gamedata.FLOAT_FIELDS) | set(gamedata.BOOL_FIELDS) | {"fat"}
        assert recorded <= set(state.OBSERVED)
        # The recording's `t` (target) is `target` here; `t` is the battle time, as gamedata.Battle.t.
        assert "target" in state.OBSERVED and "t" not in state.OBSERVED
        assert state.FATIGUE_LEVELS == gamedata.FATIGUE_LEVELS

    def test_the_network_reads_these_names(self):
        # tools/nn/model/observation.py reads a dict of these [B, N] arrays plus side and t.
        needed = {"x", "z", "b", "men", "hp", "mp", "ms", "m", "mv", "f", "fire", "a", "k", "ox", "oz", "lf", "rf",
                  "bf", "target", "fat", "vis"}
        assert needed <= set(state.OBSERVED)
        assert "side" in state.STATIC

    def test_groups_do_not_share_names(self):
        names = [n for g in state.GROUPS.values() for n in g]
        assert len(names) == len(set(names))
        for group in state.GROUPS.values():
            assert all(code in ("f", "b", "i") and text for code, text in group.values())

    def test_orders_have_four_kinds_and_five_fields(self):
        assert orders.KINDS == ("hold", "move", "attack", "withdraw")
        assert (orders.HOLD, orders.MOVE, orders.ATTACK, orders.WITHDRAW) == (0, 1, 2, 3)
        assert set(orders.FIELDS) == {"kind", "x", "z", "target", "run"}


class TestCalibration:
    def test_every_number_says_where_it_comes_from(self):
        sim = json.loads((params.CONFIG / "sim.json").read_text(encoding="utf-8"))
        for name, section in sim.items():
            if name.startswith("_"):
                continue
            assert isinstance(section, dict) and section.get("why"), name

    def test_passports_give_every_static_field(self):
        p = params.load()
        for key in p.units:
            row = p.static(key, "wh_main_emp_empire")
            assert set(row) == set(state.STATIC) - {"side", "lord"}, key
        archers = p.static("wh2_dlc13_emp_inf_archers_0")
        assert archers["reload"] == 11.0 and archers["hit_rate"] == 0.42 and archers["ammo0"] == 90 * 20
        slingers = p.static("wh2_main_skv_inf_skavenslave_slingers_0", "wh2_main_skv_skaven")
        assert slingers["reload"] == 11.5 and slingers["morale_bonus"] == 6
        spear = p.static("wh_main_emp_inf_spearmen_0")
        assert spear["range"] == 0 and spear["width"] == 30.0

    def test_changing_a_calibrated_number_keeps_the_rest(self):
        p = params.load()
        before = p.sim["melee"]["hit_slope"]
        q = p.with_cal("melee", hit_slope=before + 1)
        assert q.sim["melee"]["hit_slope"] == before + 1 and p.sim["melee"]["hit_slope"] == before
        assert q.sim["missile"] == p.sim["missile"] and q.sim["melee"]["impact"] == p.sim["melee"]["impact"]

"""apps.ai: duel observation, decision contract and bundled policies."""
import pytest

from tests.lua_runtime import load, new_runtime


@pytest.fixture
def lua():
    runtime = new_runtime()
    g = runtime.globals()
    g.ai = load(runtime, "apps.ai.services")
    g.contract = load(runtime, "apps.ai.contract")
    g.forced = load(runtime, "apps.ai.policies.forced_melee")
    g.delayed = load(runtime, "apps.ai.policies.delayed_melee")
    runtime.execute("""
        function observe(overrides, order, now)
            local unit = {routing=false, shattered=false, men=100, in_melee=false, idle=true}
            local target = {valid=true, visible=true}
            for k, v in pairs(overrides or {}) do
                if k == 'valid' or k == 'visible' then target[k] = v else unit[k] = v end
            end
            return ai.duel_observation(unit, target, order or {}, now or 0)
        end
    """)
    return runtime


class TestForcedMelee:
    @pytest.mark.parametrize("overrides, reason", [
        ("{routing = true}", "self_unavailable"),
        ("{men = 0}", "self_unavailable"),
        ("{visible = false}", "target_unavailable"),
        ("{in_melee = true}", "already_in_melee"),
    ])
    def test_waits(self, lua, overrides, reason):
        action, why = lua.eval(f"ai.decide_duel(forced, observe({overrides}))")
        assert (action, why) == ("wait", reason)

    def test_charge_then_retry_after_idle(self, lua):
        assert lua.eval("ai.decide_duel(forced, observe())") == ("attack", "initial_charge")
        order = "{ordered = true, last_order_ms = 1000}"
        assert lua.eval(f"ai.decide_duel(forced, observe({{}}, {order}, 3000))") == ("wait", "keep_current_order")
        assert lua.eval(f"ai.decide_duel(forced, observe({{}}, {order}, 6000))") == ("attack", "retry_after_idle")


class TestDelayedMelee:
    def test_warmup_before_first_charge(self, lua):
        assert lua.eval("ai.decide_duel(delayed, observe({}, {}, 5000))") == ("wait", "warmup_10s")
        assert lua.eval("ai.decide_duel(delayed, observe({}, {}, 10000))") == ("attack", "initial_charge")


class TestContract:
    def test_profiles(self, lua):
        lua.execute("""
            local p = contract.profile('empire-15-v1', 7)
            assert(p.units_per_side == 15 and p.max_command_proposals_per_step == 30)
            assert(not pcall(contract.profile, 'empire-7-v1', 7))
            assert(not pcall(contract.profile, 'unknown'))
        """)

    def test_bad_decision_rejected(self, lua):
        lua.execute("""
            local bad = {version = 'x', decide = function() return 'fly', 'why' end}
            assert(not pcall(ai.decide_duel, bad, observe()))
        """)

"""apps.deployment: placement-v2 contract and verification report."""
import pytest

from tests.lua_runtime import load, new_runtime


@pytest.fixture
def lua():
    runtime = new_runtime()
    g = runtime.globals()
    g.contract = load(runtime, "apps.deployment.contract")
    g.services = load(runtime, "apps.deployment.services")
    g.sandbox = load(runtime, "apps.sandbox.services")
    runtime.execute("""
        zone = {source='controlled_scenario_xml', center={x=0, z=0}, axis_u={x=1, z=0},
            axis_v={x=0, z=1}, half_u=100, half_v=50}
        roster = {{id='lord', kind='lord'}, {id='spears', kind='spearmen'}}
        ctx = contract.context(roster, zone, sandbox.copy, 'deployment-placement-v2')
        plan = {{unit_id='lord', x=0, z=0, facing_deg=90, width_m=5},
                {unit_id='spears', x=40, z=10, facing_deg=90, width_m=30}}
        function phase() return 'Deployment' end
    """)
    return runtime


class TestContract:
    def test_valid_plan(self, lua):
        lua.execute("contract.validate(plan, ctx)")

    def test_rejects_outside_zone_width_and_foreign_unit(self, lua):
        lua.execute("""
            local function rejects(p, message)
                local ok, err = pcall(contract.validate, p, ctx)
                assert(not ok and err:find(message, 1, true), err)
            end
            rejects({plan[1], {unit_id='spears', x=150, z=0, facing_deg=0, width_m=30}},
                'requested reference outside own deployment zone')
            rejects({plan[1], {unit_id='spears', x=0, z=0, facing_deg=0, width_m=50}},
                'width outside engineering input bounds')
            rejects({plan[1], {unit_id='enemy', x=0, z=0, facing_deg=0, width_m=30}},
                'foreign or duplicate deployment unit')
            rejects({plan[1]}, 'complete deployment required')
        """)


class TestVerify:
    def test_stable_sample_is_valid(self, lua):
        lua.execute("""
            local function measure(id)
                for _, p in ipairs(plan) do
                    if p.unit_id == id then
                        return {position={x=p.x, z=p.z}, ordered_position={x=p.x, z=p.z},
                            bearing_deg=p.facing_deg, ordered_bearing_deg=p.facing_deg, ordered_width_m=p.width_m}
                    end
                end
            end
            local first = services.verify(plan, ctx, measure, function() return 30 end, phase)
            assert(first.valid and not first.stable)
            local second = services.verify(plan, ctx, measure, function() return 30 end, phase,
                services.index_actual(first))
            assert(second.valid and second.stable and second.min_pair_distance == 30)
        """)

    def test_contact_and_missing_measurement_are_errors(self, lua):
        lua.execute("""
            local report = services.verify(plan, ctx, function() error('no unit') end,
                function() return 0.5 end, phase)
            assert(not report.valid)
            local codes = {}
            for _, e in ipairs(report.errors) do codes[e.code] = true end
            assert(codes.native_measurement_unavailable and codes.native_contact_unresolved)
        """)

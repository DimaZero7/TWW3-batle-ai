-- Deployment contract 'deployment-placement-v2'. Pure.
-- A plan is an array of {unit_id, x, z, facing_deg, width_m}, one per own unit.
-- No footprint is inferred from width. See docs/en/apps/deployment.md.
local value = require('apps.core.value')
local units = require('apps.units.contract')

local M = {version = 2, contract = 'deployment-placement-v2'}
local finite = value.finite

-- Zone from the controlled scenario XML: centre, orthonormal axes, half sizes.
function M.zone(z)
    assert(type(z) == 'table' and z.source == 'controlled_scenario_xml',
        'trusted scenario zone required')
    assert(z.center and z.axis_u and z.axis_v, 'missing zone vector')
    for _, p in ipairs({z.center, z.axis_u, z.axis_v}) do
        assert(finite(p.x) and finite(p.z), 'invalid zone vector')
    end
    assert(finite(z.half_u) and finite(z.half_v) and z.half_u > 0 and z.half_v > 0,
        'invalid zone dimensions')
    local u, v = z.axis_u, z.axis_v
    assert(math.abs(u.x * u.x + u.z * u.z - 1) < .00001 and math.abs(v.x * v.x + v.z * v.z - 1) < .00001
        and math.abs(u.x * v.x + u.z * v.z) < .00001, 'zone axes not orthonormal')
    return z
end

function M.inside(p, z, tolerance)
    local dx, dz = p.x - z.center.x, p.z - z.center.z
    tolerance = tolerance or 0
    return math.abs(dx * z.axis_u.x + dz * z.axis_u.z) <= z.half_u + tolerance
        and math.abs(dx * z.axis_v.x + dz * z.axis_v.z) <= z.half_v + tolerance
end

-- copy(value) deep-copies plain data (sandbox.services.copy).
function M.context(roster, zone, copy, selected)
    assert(selected == M.contract, 'explicit deployment-placement-v2 selection required')
    M.zone(zone)
    local lord_min, lord_max = units.width_bounds('lord')
    local formation_min, formation_max = units.width_bounds('formation')
    return {version = 2, contract = M.contract, own_roster = copy(roster), zone = copy(zone), limits = {
        verification_profile = 'ordered-anchor-stable-contact-1m-v1',
        infantry = {min_width_m = formation_min, max_width_m = formation_max},
        lord = {min_width_m = lord_min, max_width_m = lord_max,
            width_semantics = units.WIDTH.lord.semantics},
        width_bounds_kind = 'engineering_input_bounds_not_measured_game_limits',
        anchor_native_position_semantics = 'ordered_anchor_exact_actual_reference_stable',
        reference_boundary_tolerance_m = .01,
        ordered_anchor_tolerance_m = .1,
        native_position_stability_tolerance_m = .25,
        native_contact_tolerance_m = 1,
        required_stable_samples = 2,
        settle_sample_interval_ms = 1000,
        settle_timeout_ms = 10000,
        geometry_scope = 'ordered_anchor_actual_reference_stability_and_native_pair_distance',
        full_entity_bounds_verified = false}}
end

function M.check_context(c)
    assert(c.version == 2 and c.contract == M.contract, 'deployment-placement-v2 context required')
end

function M.validate(plan, c)
    M.check_context(c)
    local z = M.zone(c.zone)
    local own = {}
    for _, u in ipairs(c.own_roster) do
        assert(u.id and not own[u.id], 'duplicate roster identity')
        own[u.id] = u
    end
    assert(type(plan) == 'table' and #plan == #c.own_roster, 'complete deployment required')
    for k in pairs(plan) do
        assert(type(k) == 'number' and k % 1 == 0 and k >= 1 and k <= #plan, 'deployment must be array')
    end
    local seen = {}
    local allowed = {unit_id = true, x = true, z = true, facing_deg = true, width_m = true}
    for _, p in ipairs(plan) do
        assert(type(p) == 'table', 'placement object required')
        for k in pairs(p) do assert(allowed[k], 'unknown placement field') end
        local u = own[p.unit_id]
        assert(u and not seen[p.unit_id], 'foreign or duplicate deployment unit')
        seen[p.unit_id] = true
        for _, k in ipairs({'x', 'z', 'facing_deg', 'width_m'}) do
            assert(finite(p[k]), 'nonfinite deployment field')
        end
        assert(p.facing_deg >= 0 and p.facing_deg < 360, 'invalid facing')
        assert(units.valid_width(u.kind, p.width_m), 'width outside engineering input bounds')
        assert(M.inside(p, z, 0), 'requested reference outside own deployment zone')
    end
    return plan
end

return M

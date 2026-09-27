-- Deployment transaction: collect both plans, validate, apply, verify.
-- Pure: engine access comes in through callbacks (deployment.adapter).
local value = require('apps.core.value')
local contract = require('apps.deployment.contract')

local M = {version = 2}
local finite, distance = value.finite, value.distance_xz

local function phase(get_phase)
    assert(get_phase() == 'Deployment', 'deployment phase closed')
end

local function add(list, code, detail)
    local row = {code = code}
    if detail then for k, v in pairs(detail) do row[k] = v end end
    list[#list + 1] = row
end

-- policies[side].deploy(context) -> plan. Both plans are collected and
-- validated before anything is applied.
function M.prepare(policies, contexts, get_phase, copy)
    phase(get_phase)
    local plans = {}
    for side = 1, 2 do
        phase(get_phase)
        plans[side] = policies[side].deploy(copy(contexts[side]))
    end
    for side = 1, 2 do contract.validate(plans[side], contexts[side]) end
    return plans
end

function M.apply(plans, get_phase, place)
    phase(get_phase)
    for side = 1, 2 do
        for _, p in ipairs(plans[side]) do
            phase(get_phase)
            place(side, p)
        end
    end
end

-- Inspects one native sample without changing the plan. Invalid or unsettled
-- samples are retried by the caller until the bounded deadline
-- (limits.settle_timeout_ms). previous = units of the prior report by id.
-- measure(unit_id) -> {position, ordered_position, bearing_deg,
--   ordered_bearing_deg, ordered_width_m}; pair_distance(a, b) -> metres.
function M.verify(plan, c, measure, pair_distance, get_phase, previous)
    contract.validate(plan, c)
    phase(get_phase)
    local limits = c.limits
    local report = {units = {}, pairs = {}, errors = {}, valid = true, stable = previous ~= nil,
        geometry_scope = limits.geometry_scope, full_entity_bounds_verified = false}
    local function invalid(target, code, detail)
        add(target, code, detail)
        add(report.errors, code, detail)
        report.valid = false
    end
    for i, p in ipairs(plan) do
        phase(get_phase)
        local unit = {unit_id = p.unit_id, requested = p, errors = {}}
        local ok, m = pcall(measure, p.unit_id)
        if not ok or type(m) ~= 'table' then
            invalid(unit.errors, 'native_measurement_unavailable', {unit_id = p.unit_id})
            report.stable = false
        else
            unit.actual = m
            local pos_ok = m.position and finite(m.position.x) and finite(m.position.z)
            local ordered_ok = m.ordered_position and finite(m.ordered_position.x)
                and finite(m.ordered_position.z)
            if not pos_ok then invalid(unit.errors, 'native_position_unavailable', {unit_id = p.unit_id}) end
            if not ordered_ok then
                invalid(unit.errors, 'native_ordered_position_unavailable', {unit_id = p.unit_id})
            end
            if pos_ok then
                unit.requested_native_reference_delta_m = distance(p, m.position)
                if not contract.inside(m.position, c.zone, limits.reference_boundary_tolerance_m) then
                    invalid(unit.errors, 'native_reference_outside_zone', {unit_id = p.unit_id})
                end
            end
            if ordered_ok then
                unit.requested_ordered_anchor_delta_m = distance(p, m.ordered_position)
                if unit.requested_ordered_anchor_delta_m > limits.ordered_anchor_tolerance_m then
                    invalid(unit.errors, 'ordered_anchor_mismatch', {unit_id = p.unit_id,
                        delta_m = unit.requested_ordered_anchor_delta_m,
                        limit_m = limits.ordered_anchor_tolerance_m})
                end
            end
            if not (finite(m.bearing_deg) and finite(m.ordered_bearing_deg) and finite(m.ordered_width_m)) then
                invalid(unit.errors, 'native_orientation_or_width_unavailable', {unit_id = p.unit_id})
            else
                unit.requested_native_bearing_delta_deg = math.abs((m.bearing_deg - p.facing_deg + 180) % 360 - 180)
                unit.requested_ordered_width_delta_m = math.abs(m.ordered_width_m - p.width_m)
            end
            if previous then
                local old = previous[p.unit_id]
                if pos_ok and old and old.position and finite(old.position.x) and finite(old.position.z) then
                    unit.native_position_delta_since_previous_m = distance(m.position, old.position)
                    unit.stable = unit.native_position_delta_since_previous_m
                        <= limits.native_position_stability_tolerance_m
                else
                    unit.stable = false
                end
                if not unit.stable then
                    report.stable = false
                    add(unit.errors, 'native_position_unstable', {unit_id = p.unit_id,
                        delta_m = unit.native_position_delta_since_previous_m,
                        limit_m = limits.native_position_stability_tolerance_m})
                end
            else
                unit.stable = false
                add(unit.errors, 'previous_sample_required', {unit_id = p.unit_id})
            end
        end
        for j = 1, i - 1 do
            phase(get_phase)
            local other = plan[j].unit_id
            local pair = {unit_a = p.unit_id, unit_b = other, errors = {}}
            local good, d = pcall(pair_distance, p.unit_id, other)
            if good and finite(d) then
                pair.distance_m = d
                report.min_pair_distance = report.min_pair_distance and math.min(report.min_pair_distance, d) or d
                if d <= limits.native_contact_tolerance_m then
                    invalid(pair.errors, 'native_contact_unresolved', {unit_a = p.unit_id, unit_b = other,
                        distance_m = d, limit_m = limits.native_contact_tolerance_m})
                end
            else
                invalid(pair.errors, 'native_pair_distance_unavailable', {unit_a = p.unit_id, unit_b = other})
            end
            report.pairs[#report.pairs + 1] = pair
        end
        report.units[#report.units + 1] = unit
    end
    return report
end

-- Previous-sample lookup for the next verify call.
function M.index_actual(report)
    local by_id = {}
    for _, unit in ipairs(report.units) do by_id[unit.unit_id] = unit.actual end
    return by_id
end

return M

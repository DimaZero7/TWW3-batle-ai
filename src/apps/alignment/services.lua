-- Aligning our army with the enemy's main group (pure). Only alignment, no
-- approach: the distance to them stays as it is (approach is the next step).
-- Realign only when clearly off (user rule "no rushing").
--
-- The line to stand on (params.line):
--   'front' (default, 28.09.2026, task 26): only a turn where we stand, never a
--     shift aside. Farther than near_m between our anchor and their centre: face
--     their centre, but only when they are more than far_angle_deg off our way
--     (a narrow lane is walked along it, not across: the game, window_game);
--     nearer: square to their front (the window needs our line parallel to theirs). The game's AI in defence turns to face our army by
--     itself when we move aside (6-15 deg per burst, 40 deg in a battle, only
--     within 180-270 m) while its centre stays (0.1-3.4 m): aligning with where
--     it LOOKS, shifting aside, chased it round and round and took the army
--     245-344 m aside onto rocks. A turn in place does not set it turning.
--   'centres': always face their centre (a turn only).
--   'enemy_facing' (before): stand on the line their army looks along, through
--     their centre, facing them (a turn and a shift).
-- Without their main group: the battlefield axis (our centre -> theirs).
local value = require('apps.core.value')
local battlefield = require('apps.battlefield.services')

local M = {}
local finite = value.finite

M.DEFAULTS = {max_angle_deg = 10, max_offset_m = 15, line = 'front', near_m = 270, far_angle_deg = 20}

local function params_of(params)
    local p = {}
    for k, v in pairs(M.DEFAULTS) do
        if params and params[k] ~= nil then p[k] = params[k] else p[k] = v end
    end
    return p
end
M.params = params_of

-- Smallest signed difference a - b in degrees, in (-180, 180].
function M.angle_diff(a, b)
    local d = (a - b) % 360
    if d > 180 then d = d - 360 end
    return d
end

-- The line to stand on: {origin, bearing = the way we must face}.
-- enemy = {centre = {x, z}, facing = bearing | nil} (their main group, apps.vision);
-- current = {anchor, bearing} (where we stand).
function M.line(field, enemy, current, params)
    local p = params_of(params)
    local has_centre = enemy and enemy.centre and finite(enemy.centre.x) and finite(enemy.centre.z)
    if p.line == 'enemy_facing' and has_centre and finite(enemy.facing) then
        -- Their front looks along `facing`; we stand on that line and look back at them.
        return {origin = enemy.centre, bearing = (enemy.facing + 180) % 360, source = 'enemy_facing'}
    end
    -- 'front' needs an army that looks somewhere; a march to a goal point (no facing) keeps
    -- the battlefield axis below: onto the line to the point, turning and shifting.
    if (p.line == 'centres' or (p.line == 'front' and finite(enemy and enemy.facing)))
            and has_centre and current and current.anchor then
        local dx, dz = enemy.centre.x - current.anchor.x, enemy.centre.z - current.anchor.z
        if p.line == 'front' and finite(enemy.facing) and math.sqrt(dx * dx + dz * dz) <= p.near_m then
            -- Square to their front, turning where we stand (origin: our anchor, no offset).
            return {origin = current.anchor, bearing = (enemy.facing + 180) % 360, source = 'front'}
        end
        if dx ~= 0 or dz ~= 0 then
            return {origin = enemy.centre, bearing = math.deg(math.atan2(dx, dz)) % 360, source = 'centres'}
        end
    end
    return {origin = field.origin, bearing = field.bearing, source = 'field'}
end

-- current = {anchor = {x, z} (centre of our front line), bearing}.
-- Returns {needed, angle_off_deg, offset_m, reasons, source}.
function M.check(field, current, enemy, params)
    local p = params_of(params)
    assert(current and current.anchor and finite(current.anchor.x) and finite(current.bearing), 'Current position required')
    local line = M.line(field, enemy, current, p)
    local angle = M.angle_diff(current.bearing, line.bearing)
    local offset = battlefield.to_frame(line, current.anchor).across
    local reasons = {}
    -- Far off ('front' facing their centre): only a coarse turn.
    local max_angle = (p.line == 'front' and line.source == 'centres') and p.far_angle_deg or p.max_angle_deg
    if math.abs(angle) > max_angle then reasons[#reasons + 1] = 'angle' end
    if math.abs(offset) > p.max_offset_m then reasons[#reasons + 1] = 'offset' end
    return {needed = #reasons > 0, angle_off_deg = angle, offset_m = offset, reasons = reasons,
        source = line.source, max_angle_deg = max_angle, max_offset_m = p.max_offset_m}
end

-- Where to put our formation's anchor: on the line, as far along it as now
-- (with 'centres' that is where we stand: the army only turns).
function M.target(field, current, enemy, params)
    local line = M.line(field, enemy, current, params)
    local along = battlefield.to_frame(line, current.anchor).along
    return {anchor = battlefield.to_world(line, along, 0), bearing = line.bearing, along_m = along,
        source = line.source}
end

-- The governor: when an alignment may actually be ordered, so the army never
-- aligns forever (user, 27.09.2026). Three rules on top of check's tolerance:
--   * not while our units are still moving from the last order;
--   * not more often than every cooldown_ms (20 s);
--   * give up after give_up_after alignments in a row that did not make the
--     error smaller (something else is in the way: ground, a moving enemy),
--     and leave the decision to the level above.
M.GOVERNOR = {cooldown_ms = 20000, give_up_after = 3}

local function error_of(check)
    return math.abs(check.offset_m) + math.abs(check.angle_off_deg)
end

function M.new_governor(params)
    local p = {}
    for k, v in pairs(M.GOVERNOR) do
        if params and params[k] ~= nil then p[k] = params[k] else p[k] = v end
    end
    local g = {last_ms = nil, last_error = nil, useless = 0, given_up = false}
    -- decide(now_ms, check, moving) -> 'align' | 'aligned' | 'moving' | 'cooldown' | 'given_up'.
    -- Call record_order(now_ms, check) after the alignment orders were given.
    function g.decide(now_ms, check, moving)
        if g.given_up then return 'given_up' end
        if not check.needed then
            g.useless = 0
            return 'aligned'
        end
        if moving then return 'moving' end
        if g.last_ms and now_ms - g.last_ms < p.cooldown_ms then return 'cooldown' end
        if g.last_error and error_of(check) >= g.last_error then
            g.useless = g.useless + 1
            if g.useless >= p.give_up_after then
                g.given_up = true
                return 'given_up'
            end
        end
        return 'align'
    end
    function g.record_order(now_ms, check)
        g.last_ms, g.last_error = now_ms, error_of(check)
    end
    return g
end

-- How far their front reaches past ours on each side (positive: their flank
-- hangs over ours), in the battlefield frame.
function M.overhang(field)
    return {left_m = field.own.left_m - field.enemy.left_m, right_m = field.enemy.right_m - field.own.right_m}
end

return M

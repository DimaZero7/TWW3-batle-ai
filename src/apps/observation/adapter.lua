-- One side's view of the battle: the only input an AI side may receive.
-- Built exclusively from gated sources:
--   own units   -> units.state_adapter (owned = true, full readout);
--   enemy units -> intel.adapter (visibility; position only while visible;
--                  last seen position from this side's memory);
--   own shooter -> enemy pairs -> units.range_adapter (withheld unless visible).
-- Contrast: telemetry.sampler_adapter is the full (omniscient) summary and
-- must never feed a side.
local unit_state = require('apps.units.state_adapter')
local unit_range = require('apps.units.range_adapter')
local intel = require('apps.intel.adapter')
local services = require('apps.observation.services')

local M = {version = 1}

-- ctx: side, alliance, own = {{name, unit}}, enemies = {{name, unit}},
-- memory (intel.services.new_memory for this side), cco(unit, key),
-- now_ms, shooters = set of own names to build range readings for.
function M.observe_side(ctx)
    local view = services.new_view(ctx.side, ctx.now_ms)
    local own_names = {}
    for _, it in ipairs(ctx.own) do own_names[it.name] = true end
    local function public_id(u)
        -- Only visible units reach here (state_adapter gates on visibility).
        return u:name()
    end
    for _, it in ipairs(ctx.own) do
        view.own[it.name] = unit_state.observe(it.unit, {owned = true,
            observer_alliance = ctx.alliance, cco = ctx.cco, target_id = public_id})
    end
    for _, it in ipairs(ctx.enemies) do
        view.enemies[it.name] = intel.observe(it.unit, ctx.alliance, it.name, ctx.now_ms, ctx.memory)
    end
    for _, source in ipairs(ctx.own) do
        if ctx.shooters and ctx.shooters[source.name] then
            for _, target in ipairs(ctx.enemies) do
                view.range[source.name .. '>' .. target.name] = unit_range.observe(source.unit, target.unit, {
                    observer_alliance = ctx.alliance, observed_ms = ctx.now_ms, cco = ctx.cco,
                    is_friendly = function(u) return own_names[u:name()] == true end})
            end
        end
    end
    return services.check(view)
end

return M

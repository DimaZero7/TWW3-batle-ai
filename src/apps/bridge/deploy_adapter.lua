-- The deployment by the companion's strategist (nn_arena config.deploy 'strategist'; docs/ru/apps/bridge.md
-- "Расстановка стратегом", docs/ru/training/strategist.md "В игре"). In the Deployment phase, once:
--   1. writes the deployment document (services.deploy_document): our units' full rows, the enemy's by its
--      composition only, both deployment zones as the battle file declares them;
--   2. waits for the companion's plan (services.parse_plan) every POLL_MS real ms, up to opts.wait_s; none by then:
--      done('no_plan') - the caller ends the phase as before (the battle file's places);
--   3. takes every planned unit under script control and waits until the engine says so (the flag came ~200 ms
--      after take_control in the 26.09.2026 check: research/evidence/units/deployment-20260926), at most
--      CONTROL_TRIES polls - then teleports anyway (the flag is diagnostic, the controller's association is what moves
--      the unit);
--   4. teleports each to its place (apps.orders.adapter teleport: point, bearing - the engine's sector middle, width);
--   5. after opts.settle_ms real ms reads every unit back (position, bearing(), ordered_width(), unit_distance to the
--      others placed before it: the engine's distance between their boxes) and logs each against the plan
--      (services.deploy_check: nn_deploy_check); a deviation is logged, never stops the battle;
--   6. releases the controllers and calls done('placed') - the caller ends the phase; whoever commands our side
--      next (the planner, the network's bridge) takes the units as usual.
-- Real-time callbacks throughout: the deployment does not wait on the battle's clock.
local json = require('apps.core.json')
local orders = require('apps.orders.adapter')
local facing = require('apps.orders.facing')
local exchange = require('apps.bridge.exchange_adapter')
local services = require('apps.bridge.services')

local M = {}

M.DOC_FILE = 'tww3_bai_nn_deploy.json'
M.PLAN_FILE = 'tww3_bai_nn_plan.txt'
M.POLL_MS = 100
M.CONTROL_TRIES = 30
M.PREFIX = 'tww3_bai_nn_deploy_'
M.CALLBACKS = {'poll', 'control', 'settle'}

local function read(fn)
    local ok, v = pcall(fn)
    if ok then return v end
    return nil
end

local function round(v, k)
    if type(v) ~= 'number' or v ~= v then return nil end
    local m = 10 ^ (k or 1)
    return math.floor(v * m + 0.5) / m
end

-- opts: bm, army, own / enemies (nn_arena items: name, unit), own_rows(), enemy_rows() (the rows of nn_arena's
-- net_rows), vec(x, z), emit(event, fields), meta (batch, factions, attacker, map), zones {own, enemy}, wait_s,
-- settle_ms, guard(fn) (the entry's error guard), done(status).
function M.start(opts)
    local bm = opts.bm
    local h = {status = 'asked', polls = 0, items = {}}
    local function later(fn, ms, name)
        bm:real_callback(opts.guard(fn), ms, M.PREFIX .. name)
    end
    local function finish(status, fields)
        h.status = status
        exchange.remove(M.DOC_FILE)
        fields.stage = status
        opts.emit('nn_deploy', fields)
        opts.done(status)
    end

    local function measure()
        local n, far, turned, outside, overlaps, missing = 0, 0, 0, 0, 0, 0
        local max_d, max_db, min_pair
        for i, x in ipairs(h.items) do
            local u, p = x.it.unit, x.p
            local pos = read(function() return u:position() end)
            local m = {x = pos and read(function() return pos:get_x() end),
                z = pos and read(function() return pos:get_z() end),
                b = read(function() return u:bearing() end), w = read(function() return u:ordered_width() end)}
            local c = services.deploy_check(p, m, opts.zones.own, facing.engine(p.b))
            local pair
            for j = 1, i - 1 do
                local d = read(function() return u:unit_distance(h.items[j].it.unit) end)
                if type(d) == 'number' and (pair == nil or d < pair) then pair = d end
            end
            n = n + 1
            if c.missing then
                missing = missing + 1
            else
                if c.far then far = far + 1 end
                if c.turned then turned = turned + 1 end
                if not c.inside then outside = outside + 1 end
                max_d = math.max(max_d or 0, c.d_m)
                if c.db then max_db = math.max(max_db or 0, c.db) end
            end
            if pair ~= nil then
                if pair <= 1 then overlaps = overlaps + 1 end    -- the 26.09.2026 check: > 1 m apart, no touch
                min_pair = math.min(min_pair or pair, pair)
            end
            opts.emit('nn_deploy_check', {u = x.it.name,
                plan = {x = round(p.x), z = round(p.z), b = round(p.b, 2), w = round(p.w), cx = round(p.cx), cz = round(p.cz)},
                at = {x = round(m.x), z = round(m.z), b = round(m.b, 2), w = round(m.w)},
                d_m = round(c.d_m, 2), db = round(c.db, 2), edge_m = round(c.edge_m), inside = c.inside,
                far = c.far, turned = c.turned, pair_m = round(pair), missing = c.missing})
        end
        for _, x in ipairs(h.items) do orders.release(x.uc) end
        finish('placed', {units = n, far = far, turned = turned, outside = outside, overlaps = overlaps,
            missing = missing, max_d_m = round(max_d, 2), max_db = round(max_db, 2), min_pair_m = round(min_pair)})
    end

    local function place()
        for _, x in ipairs(h.items) do
            orders.teleport(x.uc, opts.vec(x.p.x, x.p.z), x.p.b, x.p.w)
        end
        opts.emit('nn_deploy', {stage = 'teleported', units = #h.items, settle_ms = opts.settle_ms})
        later(measure, opts.settle_ms, 'settle')
    end

    local function wait_control(tries)
        local all = true
        for _, x in ipairs(h.items) do
            if read(function() return x.it.unit:is_script_controlled() end) ~= true then all = false end
        end
        if all or tries >= M.CONTROL_TRIES then
            opts.emit('nn_deploy', {stage = 'control', confirmed = all, polls = tries})
            return place()
        end
        later(function() wait_control(tries + 1) end, M.POLL_MS, 'control')
    end

    local function apply(plan)
        local by_name, planned, unknown = {}, {}, {}
        for _, it in ipairs(opts.own) do by_name[it.name] = it end
        for _, p in ipairs(plan.places) do
            local it = by_name[p.unit]
            if it and not planned[p.unit] then
                planned[p.unit] = true
                h.items[#h.items + 1] = {it = it, p = p, uc = orders.take_control(opts.army, it.unit)}
            else
                unknown[#unknown + 1] = p.unit
            end
        end
        local unplanned = {}
        for _, it in ipairs(opts.own) do
            if not planned[it.name] then unplanned[#unplanned + 1] = it.name end
        end
        opts.emit('nn_deploy', {stage = 'plan', places = #h.items, unknown = unknown, unplanned = unplanned,
            think_ms = plan.think_ms, polls = h.polls})
        if #h.items == 0 then return finish('empty_plan', {}) end
        wait_control(0)
    end

    local tries = math.max(1, math.ceil((opts.wait_s or 0) * 1000 / M.POLL_MS))
    local function poll()
        h.polls = h.polls + 1
        local plan, why = services.parse_plan(exchange.read(M.PLAN_FILE))
        if plan and plan.batch == opts.meta.batch then
            exchange.remove(M.PLAN_FILE)
            return apply(plan)
        end
        h.why = plan and 'other batch' or why
        if h.polls >= tries then return finish('no_plan', {polls = h.polls, why = h.why}) end
        later(poll, M.POLL_MS, 'poll')
    end

    exchange.remove(M.PLAN_FILE)
    local doc = services.deploy_document(opts.meta, opts.zones, opts.own_rows(), opts.enemy_rows())
    h.write_mode = exchange.write(M.DOC_FILE, json.encode(doc))
    opts.emit('nn_deploy', {stage = 'asked', own = #opts.own, enemy = #opts.enemies, wait_s = opts.wait_s,
        write_mode = h.write_mode})
    poll()
    return h
end

-- The deployment's pending callbacks (the entry's cleanup on an error or the battle's end).
function M.cancel(bm)
    for _, name in ipairs(M.CALLBACKS) do
        pcall(function() bm:remove_real_callback(M.PREFIX .. name) end)
    end
end

return M

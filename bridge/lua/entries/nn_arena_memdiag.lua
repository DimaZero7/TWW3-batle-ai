-- Entry: nn_arena with a DIAGNOSTIC memory reading (bridge.build nn-arena --mem-diag only; never a series, an exam or
-- the owner's battle: bridge/config/memreader.json). The battle is entries.nn_arena's, unchanged; beside it, read only
-- (memreader Plus, apps.mem.entities - as the probes read), every DIAG_MS of model time from the battle's start:
--   the enemy lord's battle unit: config.mem.layout.u_fields brk / oi / oq (the break-off window, the order queue's
--   index and count: names and offsets from the sites file) and the current order's slot (layout.form d_oslot +
--   oi x d_ostride): its first pointer (the order's class, vt) and the pointer at d_otgt (the order's target: one of the
--   bound units' names, '?' another, '' none)
--   the men of the lord and of our units whose centre is within NEAR_UNIT_M of him: the rows of the men within NEAR_MAN_M
--   of him (apps.mem.entities ROW), and the ids that left a unit's list since the last poll (gone)
-- Event 'mem_diag' {t, tick, lord = {brk, oi, oq, vt, tg, row}, near = {unit = rows}, gone = {unit = ids}, new};
-- 'mem_diag_attach' once {mem_diag = true, status, bound}; these rows carry mem_diag = true, the run's manifest
-- mem_diag and config.mem_diag: such a run is never counted in a series.
local base = require('entries.nn_arena')
local memmen = require('apps.mem.entities')
local telemetry = require('apps.telemetry.adapter')

local M = {}

local LOG = 'tww3_bai_events.jsonl'
local TIMER = 'tww3_bai_nn_arena_memdiag'
M.DIAG_MS = 100
M.NEAR_UNIT_M = 60
M.NEAR_MAN_M = 8
M.LORD = 'enemy_lord'

function M.main(bm, config, globals)
    local state = base.main(bm, config, globals)
    if not (config.mem_diag and config.mem and state and state.sides) then return state end
    local mr = _G.memreader_plus
    local emit, flush = telemetry.buffered_sink(LOG, function(row)
        row.schema, row.build, row.scenario, row.mem_diag = 1, config.build, 'nn_arena', true
        row.batch, row.run_id, row.run = state.batch, state.run_id, 1
        row.model_ms = bm:time_elapsed_ms()
    end)
    local handle, status = nil, {ok = false, why = 'no memreader_plus'}
    if mr then handle, status = memmen.attach(mr, config.mem) end
    local bound, names, ptr_name, units = {}, {}, {}, {}
    if handle then
        for side = 1, 2 do
            for _, it in ipairs(state.sides[side]) do
                if it.name == M.LORD or side == 1 then
                    local ok, info = handle.bind(it.name, it.unit)
                    bound[it.name] = ok and info or {why = info}
                    if ok then
                        names[#names + 1] = it.name
                        units[it.name] = it.unit
                        ptr_name[info.ptr] = it.name
                    end
                end
            end
        end
        handle.unhook()
    end
    emit('mem_diag_attach', {mem_diag = true, status = status, bound = bound})
    flush()
    if not (handle and units[M.LORD]) then return state end
    local L = config.mem.layout
    local F, U = L.form or {}, L.u_fields or {}
    local spec = {}
    for _, k in ipairs({'brk', 'oi', 'oq'}) do
        if U[k] then spec[k] = {U[k][1], U[k][2]} end
    end
    local last_ids, c = {}, 0
    local function lord_fields()
        local p = handle.reader.units[M.LORD].ptr
        local out = {}
        local ok, s = pcall(mr.read_struct, p, 0, spec)
        if ok then for k in pairs(spec) do out[k] = s[k] end end
        if out.oi and F.d_oslot and F.d_ostride and F.d_otgt and out.oi >= 0 and out.oi < 64 then
            local at = F.d_oslot + out.oi * F.d_ostride
            local okv, vt = pcall(mr.read_pointer, p, at)
            out.vt = okv and vt and mr.tostring(vt) or nil
            local okt, tg = pcall(mr.read_pointer, p, at + F.d_otgt)
            if okt and tg and not mr.is_null(tg) then out.tg = ptr_name[mr.tostring(tg)] or '?' else out.tg = '' end
        end
        return out
    end
    local function dist(a, b)
        local ok, d = pcall(function()
            local p, q = a:position(), b:position()
            return math.sqrt((p:get_x() - q:get_x()) ^ 2 + (p:get_z() - q:get_z()) ^ 2)
        end)
        return ok and d or nil
    end
    local failed = false
    local function poll()
        if state.finished then
            pcall(function() bm:remove_process(TIMER) end)
            flush()
            return
        end
        if not state.active then return end
        c = c + 1
        local lord = units[M.LORD]
        local which = {M.LORD}
        for _, n in ipairs(names) do
            if n ~= M.LORD then
                local d = dist(units[n], lord)
                if d and d <= M.NEAR_UNIT_M then which[#which + 1] = n end
            end
        end
        local d = handle.poll(which)
        local row = {t = bm:time_elapsed_ms(), tick = d.tick, c = c, cost = d.cost, new = d.new, near = {}, gone = {}}
        row.lord = lord_fields()
        local lr = d.units[M.LORD]
        local W = #memmen.ROW
        local lx, lz
        if lr and lr.rows and #lr.rows >= W then
            lx, lz = lr.rows[3], lr.rows[4]
            row.lord.row = lr.rows
        end
        for name, u in pairs(d.units) do
            if name ~= M.LORD then
                local keep, now = {}, {}
                local rows = u.rows or {}
                for i = 1, #rows, W do
                    now[rows[i]] = true
                    if lx and (rows[i + 2] - lx) ^ 2 + (rows[i + 3] - lz) ^ 2 <= M.NEAR_MAN_M ^ 2 then
                        for j = 0, W - 1 do keep[#keep + 1] = rows[i + j] end
                    end
                end
                if #keep > 0 then row.near[name] = keep end
                local gone = {}
                for id in pairs(last_ids[name] or {}) do
                    if not now[id] then gone[#gone + 1] = id end
                end
                if #gone > 0 then row.gone[name] = gone end
                last_ids[name] = now
            end
        end
        for name in pairs(last_ids) do
            if not d.units[name] then last_ids[name] = nil end       -- (out of NEAR_UNIT_M: no gone from a stale list)
        end
        local e = handle.errors()
        if #e > 0 then row.errors = e end
        emit('mem_diag', row)
        if c % 10 == 0 then flush() end
    end
    bm:repeat_callback(function()
        local ok, err = pcall(poll)
        if not ok and not failed then
            failed = true                       -- (once: the battle goes on, the diagnosis says why it stopped)
            pcall(emit, 'mem_diag_error', {message = tostring(err)})
            pcall(flush)
        end
    end, M.DIAG_MS, TIMER)
    return state
end

return M

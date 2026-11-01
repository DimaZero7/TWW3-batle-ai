-- The morale of battle units read from the game's memory (memreader Plus, workshop 3811098873; docs/ru/game/memreader.md):
-- the points, the state, the list of effects (each a number and its points) the game sums into the morale's target, the
-- timed effects, and the unit's fields the effects are made from (casualty shares, fire flags, the fight's balance).
-- PROBES ONLY: never in the network's battle bridge or in exams. Read only - nothing is written to the game's memory;
-- the binding of a script unit to its battle unit is apps.mem.entities' (a prefix on unit:ammo_left(), removed after).
--
-- The structure offsets are NOT in this repository: the build (tools/build.py --mem) takes them from build/mem/sites.json
-- (outside Git) into the probe's config (cfg.layout); this file knows only their names:
--   u_morale   the battle unit's morale object (an offset in the unit; the object's own fields below are offsets in it)
--   o_unit     the object -> its battle unit (the check: the same unit)
--   o_state    the state (int32: 0 impetuous, 1 eager, 2 confident, 3 steady, 4 shaken, 5 wavering, 6 routing, 7 shattered)
--   o_points   the morale points now (int32);  o_target  base + the effects' sum (int32: the step's goal)
--   o_fx       the effects' list: a ring of nodes with a sentinel at o_fx (its +8: the first node); o_fx_n the count
--              (uint32); a node: n_next (pointer), n_id (int32, the effect's number: the game's name table, R-work
--              notes/mem/morale.md), n_val (int32, points)
--   o_timed    a vector (capacity, size +4, data +8) of timed effects, t_stride bytes each: id (int32), value (int32),
--              until (float)
--   o_contacts the enemies in contact: a ring of nodes like o_fx (n_next; c_id the enemy's number, c_side 0 front /
--              1 flank / 2 rear: the side first struck from, kept while the enemy is in contact)
--   o_surprise, o_waver, o_charge, o_rally (int32 timers in ticks; -1 off), o_bytes (the flags at +0 .. +15: first,
--              terror flight, shock, may rout, may shatter, routs...: read as 16 bytes, told apart offline)
--   u_fields   {name = {offset, type}} the battle unit's own fields (hp left, recent / extended casualties, fire and
--              artillery flags, the fight's balance, men): read as they are, named offline (tools/nn/mem_probe.py)
-- Only the unit's own memory and the nodes of its own two lists are read (bounded by FX_MAX: a list being rebuilt by the
-- simulation thread while we read it gives ok = 0, never a walk off into other memory).
--
-- attach(mr, cfg) -> handle (or nil, status): apps.mem.entities.attach (the checks, the hook); handle.bind(name, unit),
-- handle.unhook() as there; handle.poll(names) -> {c, tick, cost, units = {name = ROW fields + fx, tm, ct, uf}}:
--   st, pts, tgt   state, points, target;  n, ok  the list's count and whether the walk met it and the sentinel;
--   fx  {id, val, id, val, ...};  tm  {id, val, until, ...};  ct  {id, side, ...};  tim  {surprise, waver, charge, rally};
--   by  the 16 flag bytes;  uf  {name = value} of cfg.layout.u_fields
local E = require('apps.mem.entities')

local M = {}

M.FX_MAX = 64
M.TIMED_MAX = 16
M.CONTACTS_MAX = 16

local function int(v)
    if type(v) ~= 'number' or v ~= v then return 0 end
    return v
end
local function r3(v)
    if type(v) ~= 'number' or v ~= v then return 0 end
    return math.floor(v * 1000 + 0.5) / 1000
end

-- The reader (pure over mr's reads; the probe's timer calls poll, the tests directly).
function M.reader(mr, L)
    local R = {units = {}, order = {}, calls = 0, errors = {}}
    local O = L.u_morale
    local spec = {
        un = {L.o_unit, 'pointer'},
        st = {L.o_state, 'int32'}, pts = {L.o_points, 'int32'}, tgt = {L.o_target, 'int32'},
        n = {L.o_fx_n, 'uint32'}, first = {L.o_fx + 8, 'pointer'},
        tn = {L.o_timed + 4, 'uint32'}, td = {L.o_timed + 8, 'pointer'},
        cfirst = {L.o_contacts + 8, 'pointer'},
        sur = {L.o_surprise, 'int32'}, wav = {L.o_waver, 'int32'}, chg = {L.o_charge, 'int32'},
        ral = {L.o_rally, 'int32'},
    }
    for k = 0, 15 do spec['b' .. k] = {L.o_bytes + k, 'uint8'} end
    local node = {nx = {L.n_next, 'pointer'}, id = {L.n_id, 'int32'}, v = {L.n_val, 'int32'}}
    local cnode = {nx = {L.n_next, 'pointer'}, id = {L.c_id, 'int32'}, v = {L.c_side, 'int32'}}
    local timed = {id = {0, 'int32'}, v = {4, 'int32'}, t = {8, 'float'}}
    local ufields = {}
    for name, f in pairs(L.u_fields or {}) do ufields[name] = {f[1], f[2]} end

    function R.add(name, unit)
        if not R.units[name] then R.order[#R.order + 1] = name end
        R.units[name] = {ptr = unit}
    end

    -- a ring list from `first` to the sentinel at unit + off: {id, val, ...}, met (the walk ended at the sentinel)
    local function ring(unit, off, first, layout, max)
        local sentinel = mr.add(unit, O + off)
        local out, p, k = {}, first, 0
        while p and not mr.is_null(p) and not mr.eq(p, sentinel) and k < max do
            local q = mr.read_struct(p, 0, layout)
            out[#out + 1], out[#out + 2] = int(q.id), int(q.v)
            p, k = q.nx, k + 1
        end
        return out, (p ~= nil and not mr.is_null(p) and mr.eq(p, sentinel)), k
    end

    function R.read(unit)
        local s = mr.read_struct(unit, O, spec)
        local row = {st = int(s.st), pts = int(s.pts), tgt = int(s.tgt), n = int(s.n),
            tim = {int(s.sur), int(s.wav), int(s.chg), int(s.ral)}, by = {}}
        for k = 0, 15 do row.by[k + 1] = int(s['b' .. k]) end
        row.same = (s.un ~= nil and mr.eq(s.un, unit)) and 1 or 0
        local met, k = false, 0
        if row.n <= M.FX_MAX then
            row.fx, met, k = ring(unit, L.o_fx, s.first, node, M.FX_MAX)
        else
            row.fx = {}
        end
        row.ok = (met and k == row.n) and 1 or 0
        row.tm = {}
        local tn = int(s.tn)
        if tn > 0 and tn <= M.TIMED_MAX and s.td and not mr.is_null(s.td) then
            local stride = L.t_stride or 12
            for i = 0, tn - 1 do
                local q = mr.read_struct(s.td, i * stride, timed)
                local n = #row.tm
                row.tm[n + 1], row.tm[n + 2], row.tm[n + 3] = int(q.id), int(q.v), r3(q.t)
            end
        end
        row.tn = tn
        if L.o_contacts then
            local cmet
            row.ct, cmet = ring(unit, L.o_contacts, s.cfirst, cnode, M.CONTACTS_MAX)
            row.cok = cmet and 1 or 0
        end
        if next(ufields) then
            local q = mr.read_struct(unit, 0, ufields)
            row.uf = {}
            for name, f in pairs(ufields) do
                row.uf[name] = (f[2] == 'float') and r3(q[name]) or int(q[name])
            end
        end
        return row
    end

    function R.poll(names)
        R.calls = R.calls + 1
        local out = {c = R.calls, units = {}}
        local os_ = rawget(_G, 'os')
        local clock = type(os_) == 'table' and os_.clock or nil
        local t0 = clock and clock() or nil
        for _, name in ipairs(names or R.order) do
            local U = R.units[name]
            if U then
                local ok, row = pcall(R.read, U.ptr)
                if ok then
                    out.units[name] = row
                elseif #R.errors < 20 then
                    R.errors[#R.errors + 1] = name .. ': ' .. tostring(row)
                end
            end
        end
        if t0 then out.cost = math.floor((clock() - t0) * 1e5 + 0.5) / 100 end    -- ms of processor time
        return out
    end

    return R
end

-- The checks and the binding (apps.mem.entities), then the morale reader over the bound units. -> handle, status
function M.attach(mr, cfg)
    cfg = cfg or {}
    local L = cfg.layout or {}
    if not (L.u_morale and L.o_points and L.o_fx) then
        return nil, {ok = false, why = 'no morale layout in the config (u_morale, o_points, o_fx)'}
    end
    local men, status = E.attach(mr, cfg)
    if not men then return nil, status end
    local R = M.reader(mr, L)
    local handle = {reader = R, men = men}
    function handle.bind(name, unit)
        local ok, info = men.bind(name, unit)
        if not ok then return false, info end
        local p = men.reader.units[name].ptr
        R.add(name, p)
        local okr, row = pcall(R.read, p)
        if not okr then return false, 'morale read failed: ' .. tostring(row) end
        if row.same ~= 1 then return false, 'the morale object does not point back to its unit' end
        info.st, info.pts, info.n, info.ok = row.st, row.pts, row.n, row.ok
        return true, info
    end
    function handle.unhook() men.unhook() end
    function handle.tick(name)
        local U = R.units[name]
        return U and men.reader.tick(U.ptr) or nil
    end
    function handle.poll(names) return R.poll(names) end
    function handle.errors()
        local e = R.errors
        R.errors = {}
        return e
    end
    return handle, status
end

return M

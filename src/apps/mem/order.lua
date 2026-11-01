-- What a battle unit is doing with its formation, read from the game's memory (memreader Plus, workshop 3811098873;
-- docs/ru/game/memreader.md): the top behaviour of the unit and the state of its machine, the group's 'in place' mark and
-- 'reform' flag, the group's order and the suborder on top, how many men hold a move task; the stat 'scalar_speed' (base,
-- bonus, floor); the way (the job of the search and its nodes) of a unit under a move order.
-- PROBES ONLY: never in the network's battle bridge or in exams. Read only - nothing is written to the game's memory;
-- the binding of a script unit to its battle unit is apps.mem.entities'.
--
-- The structure offsets are NOT in this repository: the build (tools/build.py --mem) takes them from build/mem/sites.json
-- (outside Git) into the probe's config (cfg.layout.form); this file knows only their names:
--   u_beh_n, u_beh, u_group, u_abil_n, u_stat, u_rad    the unit: count and array of its behaviours, its current group,
--                 its active abilities, the 'scalar_speed' floats (base, bonus, floor), the two radii of its way's request
--   b_state, b_react, b_group, b_fl (a list of byte offsets)   the behaviour: the current state's pointer, its melee
--                 reaction's pointer, its group, flag bytes
--   g_n, g_u32 (offsets of u32 fields, named offline), g_u8 (of bytes), g_order, g_sub_n, g_sub, g_sub_step   the group:
--                 its men, its own fields, the embedded order, the count, array and stride of its suborders
--   o_job, j_u32, j_u8, j_head, j_next, j_prev, j_f0, j_f1, n_f0, n_f1    the order's job of the search: fields, first node,
--                 the node's links, the ranges of floats dumped from the job and from a node
--   s_fl           byte offsets read from the top suborder;  m_u8  byte offsets read from every man
-- Pointers are told apart offline by small numbers (`new` gives each its address once): functions' tables and states
-- are named by R-work notes/p4. A job and its nodes are read once, when the job is done; nothing else is followed.
--
-- reader(mr, L, ids) -> R: R.add(name, unit), R.poll(names) -> {units = {name = row}, new = {{k, address}}, jobs}
-- row: n, ab, vt, st, re, be (flag bytes list), gs (0/1 the unit's group is the behaviour's), gn, gu (list), gb (list),
--      ov, sn, sv, sf (list), mc (for each of m_u8: men with it non-zero), spd {base, bonus, floor}, rad {a, b};
--      every M.ROWS_EVERY polls also mr: flat rows {id, u8 ...} of the men.
local M = {}

M.ROWS_EVERY = 10
M.JOB_EVERY = 5
M.NODES_MAX = 32
M.JOBS_MAX = 40

local function num(v)
    if type(v) ~= 'number' or v ~= v then return 0 end
    return v
end
local function r3(v)
    if type(v) ~= 'number' or v ~= v then return 0 end
    return math.floor(v * 1000 + 0.5) / 1000
end

local function bytes_spec(list)
    local s = {}
    for i, off in ipairs(list or {}) do s['b' .. i] = {off, 'uint8'} end
    return s
end

function M.reader(mr, L, ids)
    local F = L.form
    local R = {units = {}, order = {}, calls = 0, errors = {}, keys = {}, nkeys = 0, jobs = {}, njobs = 0}
    local gspec = bytes_spec(F.g_u8)
    for i, off in ipairs(F.g_u32) do gspec['u' .. i] = {off, 'uint32'} end
    gspec.gn = {F.g_n, 'uint32'}
    gspec.sn = {F.g_sub_n, 'uint32'}
    local bspec = bytes_spec(F.b_fl)
    bspec.st = {F.b_state, 'pointer'}
    bspec.re = {F.b_react, 'pointer'}
    bspec.gr = {F.b_group, 'pointer'}
    bspec.vt = {0, 'pointer'}
    local uspec = {n = {F.u_beh_n, 'uint32'}, ab = {F.u_abil_n, 'uint32'}, gr = {F.u_group, 'pointer'},
        s0 = {F.u_stat, 'float'}, s1 = {F.u_stat + 4, 'float'}, s2 = {F.u_stat + 8, 'float'},
        r0 = {F.u_rad, 'float'}, r1 = {F.u_rad + 4, 'float'}}
    local sspec = bytes_spec(F.s_fl)
    sspec.vt = {0, 'pointer'}
    local mspec = bytes_spec(F.m_u8)

    -- a pointer as a small number (0: null), its address told once
    local function key(p, new)
        if not p or mr.is_null(p) then return 0 end
        local s = mr.tostring(p)
        local k = R.keys[s]
        if not k then
            R.nkeys = R.nkeys + 1
            k = R.nkeys
            R.keys[s] = k
            new[#new + 1] = {k, s}
        end
        return k
    end

    function R.add(name, unit)
        if not R.units[name] then R.order[#R.order + 1] = name end
        R.units[name] = {ptr = unit}
    end

    -- the job of the search, once when done: its fields, the floats of its nodes along the links
    local function read_job(group, jobs)
        if not F.o_job then return end
        -- (only an order of the search holds a job there: the upper half of a heap address, else no read at all)
        local hi = mr.read_struct(group, 0, {h = {F.g_order + F.o_job + 4, 'uint32'}}).h
        if num(hi) < 1 or num(hi) > 0x7fff then return end
        local j = mr.read_pointer(group, F.g_order + F.o_job)
        if not j or mr.is_null(j) then return end
        local s = mr.tostring(j)
        if R.jobs[s] or R.njobs >= M.JOBS_MAX then return end
        local spec = {head = {F.j_head, 'pointer'}}
        for i, off in ipairs(F.j_u32) do spec['u' .. i] = {off, 'uint32'} end
        for i, off in ipairs(F.j_u8) do spec['c' .. i] = {off, 'uint8'} end
        for off = F.j_f0, F.j_f1, 4 do spec['f' .. off] = {off, 'float'} end
        local v = mr.read_struct(j, 0, spec)
        if num(v.u1) ~= F.j_done then return end
        R.jobs[s] = true
        R.njobs = R.njobs + 1
        local out = {job = s, u = {}, c = {}, f = {}, nodes = {}}
        for i = 1, #F.j_u32 do out.u[i] = num(v['u' .. i]) end
        for i = 1, #F.j_u8 do out.c[i] = num(v['c' .. i]) end
        for off = F.j_f0, F.j_f1, 4 do out.f[#out.f + 1] = r3(v['f' .. off]) end
        local nspec = {nx = {F.j_next, 'pointer'}, pv = {F.j_prev, 'pointer'}}
        for off = F.n_f0, F.n_f1, 4 do nspec['f' .. off] = {off, 'float'} end
        -- from the head both ways along the links, at most NODES_MAX nodes
        local function walk(p, link, sign)
            local n = 0
            while p and not mr.is_null(p) and #out.nodes < M.NODES_MAX do
                n = n + 1
                local w = mr.read_struct(p, 0, nspec)
                local row = {sign * n}
                for off = F.n_f0, F.n_f1, 4 do row[#row + 1] = r3(w['f' .. off]) end
                out.nodes[#out.nodes + 1] = row
                p = w[link]
            end
        end
        if v.head and not mr.is_null(v.head) then
            walk(v.head, 'nx', 1)
            local h = mr.read_struct(v.head, 0, {pv = {F.j_prev, 'pointer'}})
            walk(h.pv, 'pv', -1)
        end
        jobs[#jobs + 1] = out
    end

    local function poll_unit(U, new, jobs, c)
        local u = mr.read_struct(U.ptr, 0, uspec)
        local row = {n = num(u.n), ab = num(u.ab), spd = {r3(u.s0), r3(u.s1), r3(u.s2)}, rad = {r3(u.r0), r3(u.r1)}}
        if row.n > 0 and row.n < 16 then
            local b = mr.read_pointer(U.ptr, F.u_beh + (row.n - 1) * 8)
            if b and not mr.is_null(b) then
                local v = mr.read_struct(b, 0, bspec)
                row.vt, row.st, row.re = key(v.vt, new), key(v.st, new), key(v.re, new)
                row.be = {}
                for i = 1, #F.b_fl do row.be[i] = num(v['b' .. i]) end
                row.gs = (v.gr and u.gr and mr.eq(v.gr, u.gr)) and 1 or 0
            end
        end
        if u.gr and not mr.is_null(u.gr) then
            local g = mr.read_struct(u.gr, 0, gspec)
            row.gn, row.sn = num(g.gn), num(g.sn)
            row.gu, row.gb = {}, {}
            for i = 1, #F.g_u32 do row.gu[i] = num(g['u' .. i]) end
            for i = 1, #F.g_u8 do row.gb[i] = num(g['b' .. i]) end
            row.ov = key(mr.read_pointer(u.gr, F.g_order), new)
            if row.sn > 0 and row.sn < 32 then
                local s = mr.read_pointer(u.gr, F.g_sub + (row.sn - 1) * F.g_sub_step)
                if s and not mr.is_null(s) then
                    local v = mr.read_struct(s, 0, sspec)
                    row.sv = key(v.vt, new)
                    row.sf = {}
                    for i = 1, #F.s_fl do row.sf[i] = num(v['b' .. i]) end
                end
            end
            if c % M.JOB_EVERY == 0 and row.ov ~= 0 then
                local okj, errj = pcall(read_job, u.gr, jobs)
                if not okj and #R.errors < 20 then R.errors[#R.errors + 1] = 'job: ' .. tostring(errj) end
            end
        end
        if #F.m_u8 > 0 then
            local list = mr.read_vector(U.ptr, L.u_men, {0, 'pointer'}, 8)
            if type(list) == 'table' then
                local mc = {}
                for i = 1, #F.m_u8 do mc[i] = 0 end
                local rows = (c % M.ROWS_EVERY == 0) and {} or nil
                for i = 1, #list do
                    local p = list[i]
                    if p and not mr.is_null(p) then
                        local v = mr.read_struct(p, 0, mspec)
                        if rows then rows[#rows + 1] = (ids and ids[mr.tostring(p)]) or 0 end
                        for k = 1, #F.m_u8 do
                            local b = num(v['b' .. k])
                            if b ~= 0 then mc[k] = mc[k] + 1 end
                            if rows then rows[#rows + 1] = b end
                        end
                    end
                end
                row.mc = mc
                row.mr = rows
            end
        end
        return row
    end

    function R.poll(names)
        R.calls = R.calls + 1
        local out = {units = {}, new = {}, jobs = {}}
        for _, name in ipairs(names or R.order) do
            local U = R.units[name]
            if U then
                local ok, row = pcall(poll_unit, U, out.new, out.jobs, R.calls)
                if ok then
                    out.units[name] = row
                elseif #R.errors < 20 then
                    R.errors[#R.errors + 1] = name .. ': ' .. tostring(row)
                end
            end
        end
        if #R.errors > 0 then
            out.errors = R.errors
            R.errors = {}
        end
        return out
    end

    return R
end

return M

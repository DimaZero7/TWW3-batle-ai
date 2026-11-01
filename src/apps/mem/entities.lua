-- The men of battle units read from the game's memory (memreader Plus, workshop 3811098873; docs/ru/game/memreader.md).
-- PROBES ONLY: never in the network's battle bridge or in exams. Read only - nothing is written to the game's memory;
-- the one hook (a patcher prefix on the script's unit:ammo_left(), only while units are bound) reads and passes on.
--
-- The code sites and the structure offsets are NOT in this repository: the build (tools/build.py --mem) takes them
-- from build/mem/sites.json (outside Git) into the probe's config (cfg = {sites, layout}); this file knows only names:
--   site  ammo_left   the script's unit:ammo_left() (arguments: the script's unit object, the call context): its prefix
--                     reads the battle unit (i_unit) of the script unit the probe binds
--         dir_table   the game's table of facing vectors (x, z floats) by a man's facing number (uint16)
--   layout i_unit     the script's unit object -> the battle unit;  u_battle  battle unit -> battle;  b_world  battle ->
--                     battle world;  w_tick  the world's simulation tick counter (uint32; a tick is 0.1 s)
--          u_men      the battle unit's vector of men (capacity, size, data: pointers)
--          m_unit     a man -> his battle unit;  m_live  flags byte (bit 0x10: in the battle, alive)
--          m_pos, m_vel   feet (x, y up, z) and velocity (x, y, z), floats;  m_face  facing number (uint16)
--          m_hp       hit points now (uint32);  m_anim  animation state (int32);  m_mode  gait (0 walk,
--                     1 run, 2 charge);  m_react  reaction state (int32: 0xd step back, 0xe knocked back, 0xf knocked
--                     down, 0x10 flying, 0x11 getting up);  m_mark  the strike timer's mark (float, s of the tick clock:
--                     an attack starts -> the mark is now);  m_fit  flags byte (bit 0x2: fit as a target);  m_flags
--                     flags byte (bit 0x8: braced);  m_brace  the bracing mass multiplier B (float);  m_speed  own speed
--          m_carrier, m_charge (optional)  the entity carrying the man (a pointer; followed to the root) and the root's charge deadline (float, s of the tick clock: the charge multiplier is
--                     clamp((deadline - now) / 13 s, 0, 1) - set to now + 13 s every tick the man is in a charge)
--          m_target, m_target_b (optional)  pointers to men: the target of the man's current task (m_target) and a
--                     second one (m_target_b) - found by discover (two paths with the same hits), told apart offline
--
-- attach(mr, cfg) -> handle (or nil, status): checks memreader and the sites (apps.mem.projectiles.check), the read
-- types; hooks ammo_left. handle.bind(name, unit) binds a script unit (calls unit:ammo_left(): the prefix sees its battle
-- unit), checks the men vector (size, every man's m_unit = the unit); handle.unhook() removes the hook (bind first).
-- handle.poll(names) reads every man of the named bound units (all when nil) -> {c, tick, tick_end, cost, units =
-- {name = {n, rows}}, new = {{id, name, address}}}: rows flat, ROW fields a man (ids unique over the battle, by address).
-- handle.discover(from, to, max_men) -> the offsets of from's men (step 8, below cfg.discover_max) that hold a pointer
-- to a man of `to`: {hex offset = hits}. Only the men's own memory is read - no pointer found there is followed (the
-- first version followed every pointer one level on; after ~10^5 such reads the game's Lua state broke).
-- Row fields tgt, tgt2: the man at m_target / m_target_b as an id (-1 a non-null pointer to no man read yet, 0 none).
local P = require('apps.mem.projectiles')

local M = {}

-- a man's row, in this order (tools/nn/mem_probe.py MAN)
M.ROW = {'id', 'live', 'x', 'z', 'y', 'vx', 'vz', 'face', 'hp', 'anim', 'mode', 'react', 'mark', 'fit', 'flags',
    'brace', 'speed', 'tgt', 'tgt2', 'carried', 'chg'}
M.MEN_MAX = 400
M.DISCOVER_MAX = 0xc00

local function r1(v)
    if type(v) ~= 'number' or v ~= v then return 0 end
    return math.floor(v * 10 + 0.5) / 10
end
local function r2(v)
    if type(v) ~= 'number' or v ~= v then return 0 end
    return math.floor(v * 100 + 0.5) / 100
end
local function int(v)
    if type(v) ~= 'number' or v ~= v then return 0 end
    return v
end

-- the read_struct layout of a man (once)
function M.spec(L)
    local s = {
        live = {L.m_live, 'uint8'},
        x = {L.m_pos, 'float'}, y = {L.m_pos + 4, 'float'}, z = {L.m_pos + 8, 'float'},
        vx = {L.m_vel, 'float'}, vz = {L.m_vel + 8, 'float'},
        face = {L.m_face, 'uint16'},
        hp = {L.m_hp, 'uint32'},
        anim = {L.m_anim, 'int32'}, mode = {L.m_mode, 'int32'}, react = {L.m_react, 'int32'},
        mark = {L.m_mark, 'float'},
        fit = {L.m_fit, 'uint8'}, flags = {L.m_flags, 'uint8'},
        brace = {L.m_brace, 'float'}, speed = {L.m_speed, 'float'},
    }
    return s
end

-- The reader (pure over mr's reads; the probe's timer calls poll, the tests directly).
function M.reader(mr, L)
    local R = {units = {}, order = {}, ids = {}, next_id = 0, calls = 0, errors = {}}
    local spec = M.spec(L)

    local function id_of(key, name, new)
        local id = R.ids[key]
        if id then return id end
        R.next_id = R.next_id + 1
        id = R.next_id
        R.ids[key] = id
        new[#new + 1] = {id, name, key}
        return id
    end

    function R.add(name, unit)
        if not R.units[name] then R.order[#R.order + 1] = name end
        R.units[name] = {ptr = unit}
    end

    -- the men of a battle unit: the list of pointers (or nil, why)
    function R.men(unit)
        local list = mr.read_vector(unit, L.u_men, {0, 'pointer'}, 8)
        if type(list) ~= 'table' then return nil, 'no vector' end
        if #list > M.MEN_MAX then return nil, 'vector of ' .. #list end
        return list
    end

    function R.tick(unit)
        local b = mr.read_pointer(unit, L.u_battle)
        local w = mr.read_pointer(b, L.b_world)
        return mr.read_struct(w, 0, {t = {L.w_tick, 'uint32'}}).t
    end

    -- (carried: 1 if the man rides another entity, chg: the root entity's charge deadline, 0 without the layout)
    local function charge_of(p)
        if not (L.m_carrier and L.m_charge) then return 0, 0 end
        local root, carried = p, 0
        for _ = 1, 4 do
            local ok, q = pcall(mr.read_pointer, root, L.m_carrier)
            if not ok or not q or mr.is_null(q) then break end
            root, carried = q, 1
        end
        local ok, v = pcall(function() return mr.read_struct(root, 0, {c = {L.m_charge, 'float'}}).c end)
        return carried, ok and r2(v) or 0
    end

    local function target_of(p, off)
        if not off then return 0 end
        local ok, q = pcall(mr.read_pointer, p, off)
        if not ok or not q or mr.is_null(q) then return 0 end
        return R.ids[mr.tostring(q)] or -1
    end

    function R.poll(names)
        R.calls = R.calls + 1
        local out = {c = R.calls, units = {}, new = {}}
        local os_ = rawget(_G, 'os')
        local clock = type(os_) == 'table' and os_.clock or nil
        local t0 = clock and clock() or nil
        for _, name in ipairs(names or R.order) do
            local U = R.units[name]
            if U then
                local ok, err = pcall(function()
                    if not out.tick then out.tick = R.tick(U.ptr) end
                    local list, why = R.men(U.ptr)
                    if not list then error(why) end
                    local rows = {}
                    for i = 1, #list do
                        local p = list[i]
                        if p and not mr.is_null(p) then
                            local s = mr.read_struct(p, 0, spec)
                            local id = id_of(mr.tostring(p), name, out.new)
                            local n = #rows
                            rows[n + 1] = id
                            rows[n + 2] = (int(s.live) % 32 >= 16) and 1 or 0
                            rows[n + 3], rows[n + 4], rows[n + 5] = r1(s.x), r1(s.z), r1(s.y)
                            rows[n + 6], rows[n + 7] = r1(s.vx), r1(s.vz)
                            rows[n + 8], rows[n + 9] = int(s.face), int(s.hp)
                            rows[n + 10], rows[n + 11], rows[n + 12] = int(s.anim), int(s.mode), int(s.react)
                            rows[n + 13] = r2(s.mark)
                            rows[n + 14], rows[n + 15] = int(s.fit), int(s.flags)
                            rows[n + 16], rows[n + 17] = r2(s.brace), r1(s.speed)
                            rows[n + 18] = target_of(p, L.m_target)
                            rows[n + 19] = target_of(p, L.m_target_b)
                            rows[n + 20], rows[n + 21] = charge_of(p)
                        end
                    end
                    out.units[name] = {n = #list, rows = rows}
                end)
                if not ok and #R.errors < 20 then R.errors[#R.errors + 1] = name .. ': ' .. tostring(err) end
            end
        end
        if t0 then out.cost = math.floor((clock() - t0) * 1e5 + 0.5) / 100 end    -- ms of processor time
        -- the tick again: the simulation thread runs on while the script reads (a poll spans tick .. tick_end)
        for _, name in ipairs(names or R.order) do
            local U = R.units[name]
            if U then
                local ok, t = pcall(R.tick, U.ptr)
                if ok then out.tick_end = t end
                break
            end
        end
        return out
    end

    -- the pointers from `from`'s men to `to`'s men (their own memory only)
    function R.discover(from, to, max_men, max_off)
        local A, B = R.units[from], R.units[to]
        local hits = {}
        if not (A and B) then return hits end
        local la, lb = R.men(A.ptr), R.men(B.ptr)
        if not (la and lb) then return hits end
        local enemy = {}
        for i = 1, #lb do if lb[i] then enemy[mr.tostring(lb[i])] = true end end
        local function add(key) hits[key] = (hits[key] or 0) + 1 end
        local step = math.max(1, math.floor(#la / (max_men or 6)))
        local done = 0
        for i = 1, #la, step do
            if done >= (max_men or 6) then break end
            done = done + 1
            local p = la[i]
            for off = 0, (max_off or M.DISCOVER_MAX) - 8, 8 do
                local ok, q = pcall(mr.read_pointer, p, off)
                if ok and q and not mr.is_null(q) then
                    local k = mr.tostring(q)
                    if enemy[k] then add(string.format('%x', off)) end
                end
            end
        end
        return hits
    end

    return R
end

-- The checks, the read types, the hook. -> handle, status (handle nil when a check failed: the probe runs without it).
function M.attach(mr, cfg)
    cfg = cfg or {}
    local status = P.check(mr, cfg)
    if not status.ok then return nil, status end
    if not (cfg.sites.ammo_left and cfg.layout and cfg.layout.u_men) then
        status.ok, status.why = false, 'no ammo_left site or men layout in the config'
        return nil, status
    end
    local L = cfg.layout
    local R = M.reader(mr, L)
    local at_ammo = P.address(mr, cfg.sites.ammo_left)
    local last = nil
    local patcher = nil
    local okp, errp = pcall(function()
        patcher = assert(mr.patcher('tww3_bai_men'))
        patcher:prefix(at_ammo, cfg.sites.ammo_left.sig, 'ammo_left', function(_, obj)
            local ok, u = pcall(mr.read_pointer, obj, L.i_unit)
            last = ok and u or nil
        end)
    end)
    status.ammo_left_hooked = okp or tostring(errp)
    if not okp then return nil, status end
    if cfg.sites.dir_table then
        local okd, dirs = pcall(function()
            local at = P.address(mr, cfg.sites.dir_table)
            local out = {}
            for _, k in ipairs({0, 4096, 8192, 16384, 32768, 49152}) do
                local s = mr.read_struct(at, k * 8, {x = {0, 'float'}, z = {4, 'float'}})
                out[#out + 1] = {k, P.r1(s.x * 100) / 100, P.r1(s.z * 100) / 100}
            end
            return out
        end)
        status.dir_table = okd and dirs or tostring(dirs)
    end
    local handle = {reader = R}
    -- binds a script unit; -> ok, {n, ptr, tick} or why
    function handle.bind(name, unit)
        last = nil
        pcall(function() return unit:ammo_left() end)
        local u = last
        if not u or mr.is_null(u) then return false, 'the hook saw no unit' end
        local okm, list = pcall(R.men, u)
        if not okm or not list then return false, 'no men vector: ' .. tostring(list) end
        local bad = 0
        for i = 1, #list do
            local okq, q = pcall(mr.read_pointer, list[i], L.m_unit)
            if not okq or not mr.eq(q, u) then bad = bad + 1 end
        end
        if #list == 0 or bad > 0 then return false, 'men ' .. #list .. ', not of this unit ' .. bad end
        R.add(name, u)
        local okt, t = pcall(R.tick, u)
        return true, {n = #list, ptr = mr.tostring(u), tick = okt and t or tostring(t)}
    end
    function handle.unhook()
        if patcher then pcall(function() patcher:remove_all() end) end
        patcher = nil
    end
    function handle.poll(names) return R.poll(names) end
    function handle.discover(from, to, max_men) return R.discover(from, to, max_men, cfg.discover_max) end
    function handle.errors()
        local e = R.errors
        R.errors = {}
        return e
    end
    return handle, status
end

return M

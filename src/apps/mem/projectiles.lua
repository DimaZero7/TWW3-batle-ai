-- The projectiles in flight read from the game's memory (memreader Plus, workshop 3811098873; docs/ru/game/memreader.md).
-- PROBES ONLY: never in the network's battle bridge or in exams. Read only - nothing is written to the game's memory;
-- the two hooks only count, read and pass the call on to the game's function.
--
-- The code sites and the structure offsets are NOT in this repository: the build (tools/build.py --mem) takes them
-- from build/mem/sites.json (outside Git) into the probe's config as cfg = {sites = {name = {hi, lo, sig, expect,
-- pattern}}, layout = {name = offset}, snapshot_max}; this file knows only their names:
--   site proj_update    the game's update of every projectile, once a simulation tick (argument: the projectile
--                       manager). It runs on the simulation thread: memreader runs such calls past the Lua callbacks
--                       (Lua is single-threaded), so its hook only counts them (hook_info: other_thread_calls)
--        ammo_left      the script's unit:ammo_left() (arguments: the script's unit object, the call context): on the
--                       script thread - the hook that runs Lua; it reads the unit's ammo from memory (the check
--                       against the value the script gets) for the first hook_max calls
--        battle_ref     code that reads battle_global (its bytes: the game's build)
--        battle_global  the battle's pointer: battle -> b_world -> w_proj = the projectile manager (world -> w_battle
--                       = the battle again: the chain's check)
--   layout mgr_active   the manager's vector of active projectiles (capacity, size, data)
--          p_state, p_pos, p_prev, p_vel, p_time, p_record, p_shooter   a projectile's state (int), position (x, y, z;
--                       y up), previous position, velocity, flight time (s), its database record, the shooting man
--          s_feet       the shooting man's feet (x, y, z);  r_category  the record's category (int)
--          i_unit, u_ammo   the script's unit object -> the battle unit, its ammo (int)
--
-- attach(mr, cfg) -> handle (or nil, status): checks memreader (present, its version), each site's bytes (the game's
-- build) and its byte pattern (exactly one match at the same address), the chain to the manager, then hooks.
-- handle.poll() (the probe's own callback, every 100 ms of battle: the script thread, between simulation ticks) walks
-- the active vector and buffers, with the poll's number c:
--   spawn {id, c, st, x, y, z, vx, vy, vz, t, rec, sx, sy, sz}        a projectile seen for the first time
--   state {id, c, st, x, y, z, t}                                      its state changed (the last one: 7, removed)
--   gone  {id, c}                                                      no longer in the vector
-- handle.drain() -> {calls, n_live, spawn, state, gone, fly, errors, records, ammo} since the last drain; fly: the
-- projectiles in the vector now that are not lying (states 5, 6) or done (7), up to cfg.snapshot_max
-- {id, st, x, y, z, vx, vy, vz, t}; records: the new database records {rec, ptr, cat}; ammo: the ammo_left hook's
-- calls {unit, ammo} (unit: the battle unit's address as text, ammo read from memory).
-- handle.stats() -> the counters (memreader's hook_info of proj_update); handle.detach() removes both hooks.
-- The scanner's two options (the charge probe's shots, apps.mem.hits): S.track - every poll the position of each
-- flying projectile is read and kept, and its gone row carries the last one {id, c, x, y, z, vx, vy, vz, t} (where it
-- stopped: the game frees a done projectile within a tick, so its state 7 is often never seen); S.on_spawn(p, st) -
-- called for each new projectile, its value (if any) is the spawn row's 15th field (what a projectile 'in a man' is in).
local M = {}

local LYING = {[5] = true, [6] = true}
M.MISS_POLLS = 3
local DONE = 7

local function r1(v)
    if type(v) ~= 'number' or v ~= v then return false end
    return math.floor(v * 10 + 0.5) / 10
end
local function r3(v)
    if type(v) ~= 'number' or v ~= v then return false end
    return math.floor(v * 1000 + 0.5) / 1000
end
M.r1 = r1

-- 'de ad be ef' -> the bytes. Pure.
function M.hex_bytes(hex)
    local out = {}
    for byte in string.gmatch(hex or '', '%x%x') do out[#out + 1] = string.char(tonumber(byte, 16)) end
    return table.concat(out)
end

-- base + hi * 2^20 + lo: Lua's numbers in the game are 32-bit floats (exact to 2^24), so the build splits each
-- relative address in two exact parts.
function M.address(mr, site)
    return mr.add(mr.add(mr.base, site.hi * 1048576), site.lo)
end

-- {ok, why, version, sites = {name = {ok, why}}}: memreader, the game's build by each site's bytes and pattern.
function M.check(mr, cfg)
    local out = {ok = false, sites = {}}
    if type(mr) ~= 'table' then
        out.why = 'memreader Plus is not loaded (_G.memreader_plus is nil; lua_mod_log.txt names the reason)'
        out.load_error = _G and _G.memreader_plus_load_error or nil
        return out
    end
    out.version, out.plus_version = mr.version, mr.plus_version
    if type(mr.hook) ~= 'function' or type(mr.hook_next) ~= 'function' or type(mr.read_vector) ~= 'function' then
        out.why = 'memreader without hook / hook_next / read_vector'
        return out
    end
    local ok = true
    for name, site in pairs(cfg.sites or {}) do
        local row = {ok = false}
        local good, why = pcall(function()
            local at = M.address(mr, site)
            row.at = mr.tostring(at)
            local want = M.hex_bytes(site.expect)
            if #want > 0 and mr.read(at, 0, #want) ~= want then
                row.why = 'bytes differ (another game build?)'
                return
            end
            if site.pattern then
                local found, count = mr.find_pattern(site.pattern)
                row.matches = tonumber(count) or count
                if not found or row.matches ~= 1 or not mr.eq(found, at) then
                    row.why = 'pattern: ' .. tostring(row.matches) .. ' matches or another address'
                    return
                end
            end
            row.ok = true
        end)
        if not good then row.why = 'check failed: ' .. tostring(why) end
        out.sites[name] = row
        ok = ok and row.ok
    end
    out.ok = ok and cfg.sites ~= nil and cfg.sites.proj_update ~= nil
    if ok and not out.ok then out.why = 'no proj_update site in the config' end
    if not ok then out.why = 'a site failed its check' end
    return out
end

-- The scanner (pure over mr's reads; the hook calls it once a simulation tick, the tests directly).
function M.scanner(mr, L, snapshot_max)
    local S = {calls = 0, next_id = 0, live = {}, spawn = {}, state = {}, gone = {}, errors = {}, rec = {},
               rec_new = {}, n_rec = 0, mgr = nil, n_live = 0}

    local function record_id(rp)
        if not rp then return false end
        local key = mr.tostring(rp)
        local id = S.rec[key]
        if id then return id end
        S.n_rec = S.n_rec + 1
        id = S.n_rec
        S.rec[key] = id
        local ok, cat = pcall(mr.read_int32, rp, L.r_category)
        S.rec_new[#S.rec_new + 1] = {rec = id, ptr = key, cat = ok and cat or false}
        return id
    end

    local function read_full(p)
        local s = mr.read_struct(p, 0, {
            st = {L.p_state, 'int32'}, x = {L.p_pos, 'float'}, y = {L.p_pos + 4, 'float'}, z = {L.p_pos + 8, 'float'},
            vx = {L.p_vel, 'float'}, vy = {L.p_vel + 4, 'float'}, vz = {L.p_vel + 8, 'float'},
            t = {L.p_time, 'float'}, rp = {L.p_record, 'pointer'}, sp = {L.p_shooter, 'pointer'}})
        return s
    end

    -- (S.track) the last place seen of a projectile, and its gone row
    function S.keep(rec, f)
        rec.x, rec.y, rec.z, rec.vx, rec.vy, rec.vz = f.x, f.y, f.z, f.vx, f.vy, f.vz
    end
    function S.gone_row(rec, c)
        if not (S.track and rec.x) then return {rec.id, c} end
        return {rec.id, c, r1(rec.x), r1(rec.y), r1(rec.z), r1(rec.vx), r1(rec.vy), r1(rec.vz), r3(rec.t)}
    end

    function S.scan(mgr)
        S.calls = S.calls + 1
        S.mgr = mgr
        local c = S.calls
        local list = mr.read_vector(mgr, L.mgr_active, {0, 'pointer'}, 8) or {}
        local n = 0
        for i = 1, #list do
            local p = list[i]
            if p then
                n = n + 1
                local key = mr.tostring(p)
                local st = mr.read_int32(p, L.p_state)
                local rec = S.live[key]
                local t = nil
                if rec and rec.st == DONE and st ~= DONE then
                    -- the slot was freed and reused for a new projectile within one tick
                    S.gone[#S.gone + 1] = S.gone_row(rec, c)
                    S.live[key], rec = nil, nil
                end
                local f_now = nil
                if rec and st ~= DONE and not LYING[st] then
                    if S.track then
                        f_now = read_full(p)
                        t = f_now.t
                    else
                        t = mr.read_float(p, L.p_time)
                    end
                    if rec.t and t + 1e-3 < rec.t then           -- the flight clock went back: a new projectile
                        S.gone[#S.gone + 1] = S.gone_row(rec, c)
                        S.live[key], rec, f_now = nil, nil, nil
                    end
                end
                if not rec then
                    local f = read_full(p)
                    S.next_id = S.next_id + 1
                    rec = {id = S.next_id, st = f.st, t = f.t, c = c}
                    S.live[key] = rec
                    local sx, sy, sz = false, false, false
                    if f.sp then
                        local ok, feet = pcall(mr.read_struct, f.sp, 0, {x = {L.s_feet, 'float'},
                            y = {L.s_feet + 4, 'float'}, z = {L.s_feet + 8, 'float'}})
                        if ok and feet then sx, sy, sz = r1(feet.x), r1(feet.y), r1(feet.z) end
                    end
                    local row = {rec.id, c, f.st, r1(f.x), r1(f.y), r1(f.z), r1(f.vx), r1(f.vy), r1(f.vz),
                        r3(f.t), record_id(f.rp), sx, sy, sz}
                    if S.on_spawn then
                        local okw, w = pcall(S.on_spawn, p, f.st)
                        if okw and w ~= nil then row[15] = w end
                        if not okw and #S.errors < 20 then S.errors[#S.errors + 1] = 'on_spawn: ' .. tostring(w) end
                    end
                    S.spawn[#S.spawn + 1] = row
                    if S.track then S.keep(rec, f) end
                elseif st ~= rec.st then
                    local f = read_full(p)
                    S.state[#S.state + 1] = {rec.id, c, st, r1(f.x), r1(f.y), r1(f.z), r3(f.t)}
                    rec.st = st
                    rec.t = f.t
                    if S.track then S.keep(rec, f) end
                elseif t then
                    rec.t = t
                    if f_now then S.keep(rec, f_now) end
                end
                rec.c = c
            end
        end
        S.n_live = n
        -- the poll reads the vector while the simulation thread may be removing from it (the last element moved into
        -- a freed slot): one projectile can be missed in one poll, so it is gone only after MISS_POLLS misses in a row
        -- (the gone row names the first poll it was missing)
        for key, rec in pairs(S.live) do
            if rec.c ~= c then
                rec.miss_c = rec.miss_c or c
                if c - rec.miss_c + 1 >= M.MISS_POLLS then
                    S.gone[#S.gone + 1] = S.gone_row(rec, rec.miss_c)
                    S.live[key] = nil
                end
            else
                rec.miss_c = nil
            end
        end
    end

    -- the projectiles in the vector now that fly (script thread, between simulation ticks)
    function S.fly()
        local out = {}
        if not S.mgr then return out end
        local list = mr.read_vector(S.mgr, L.mgr_active, {0, 'pointer'}, 8) or {}
        for i = 1, #list do
            local p = list[i]
            if p then
                local rec = S.live[mr.tostring(p)]
                local st = mr.read_int32(p, L.p_state)
                if rec and st ~= DONE and not LYING[st] then
                    if #out >= (snapshot_max or 300) then break end
                    local f = read_full(p)
                    out[#out + 1] = {rec.id, f.st, r1(f.x), r1(f.y), r1(f.z), r1(f.vx), r1(f.vy), r1(f.vz), r3(f.t)}
                end
            end
        end
        return out
    end

    function S.drain()
        local out = {calls = S.calls, n_live = S.n_live, spawn = S.spawn, state = S.state, gone = S.gone,
                     errors = S.errors, records = S.rec_new}
        local ok, fly = pcall(S.fly)
        out.fly = ok and fly or {}
        if not ok then out.errors[#out.errors + 1] = 'fly: ' .. tostring(fly) end
        S.spawn, S.state, S.gone, S.errors, S.rec_new = {}, {}, {}, {}, {}
        return out
    end
    return S
end

-- battle -> b_world -> w_proj, where world -> w_battle must lead back to the battle; -> mgr or nil. diag[tag..]: the
-- pointers read on the way, as text.
local function via(mr, L, diag, tag, battle)
    local function s(p) return p and mr.tostring(p) or false end
    diag[tag] = s(battle)
    if not battle or mr.is_null(battle) then return nil end
    local okw, world = pcall(mr.read_pointer, battle, L.b_world)
    diag[tag .. '_world'] = okw and s(world) or tostring(world)
    if not okw or not world or mr.is_null(world) then return nil end
    local okb, back = pcall(mr.read_pointer, world, L.w_battle)
    diag[tag .. '_back'] = okb and s(back) or tostring(back)
    if not okb or not mr.eq(back, battle) then return nil end
    local okm, mgr = pcall(mr.read_pointer, world, L.w_proj)
    diag[tag .. '_mgr'] = okm and s(mgr) or tostring(mgr)
    if not okm or not mgr or mr.is_null(mgr) then return nil end
    return mgr
end

-- The manager. The battle comes from battle_global or, failing that, from a battle unit at u_battle (cfg) or at the
-- offset discover() found (the ammo_left hook gives the unit). -> mgr or nil, diag.
function M.manager(mr, cfg, unit, u_off)
    local L = cfg.layout
    local diag = {}
    local okg, g = pcall(mr.read_pointer, M.address(mr, cfg.sites.battle_global), 0)
    local mgr = via(mr, L, diag, 'g', okg and g or nil)
    if mgr then diag.route = 'global' return mgr, diag end
    local off = u_off or L.u_battle
    if unit and off then
        local oku, b = pcall(mr.read_pointer, unit, off)
        mgr = via(mr, L, diag, 'u', oku and b or nil)
        if mgr then diag.route = 'unit+' .. string.format('%x', off) return mgr, diag end
    end
    return nil, diag
end

-- The offsets (step 8, below max_off) of `from` that hold a battle - a pointer B with B -> b_world -> W, W ->
-- w_battle = B - and of those that hold a world (W -> w_battle -> B, B -> b_world = W). Read only, every read guarded.
function M.discover(mr, L, from, max_off)
    local out = {battle = {}, world = {}}
    for off = 0, (max_off or 0x4000) - 8, 8 do
        local ok, p = pcall(mr.read_pointer, from, off)
        if ok and p and not mr.is_null(p) then
            local okw, w = pcall(mr.read_pointer, p, L.b_world)
            if okw and w and not mr.is_null(w) then
                local okb, b = pcall(mr.read_pointer, w, L.w_battle)
                if okb and b and mr.eq(b, p) then out.battle[#out.battle + 1] = off end
            end
            local okb, b = pcall(mr.read_pointer, p, L.w_battle)
            if okb and b and not mr.is_null(b) then
                local okw2, w2 = pcall(mr.read_pointer, b, L.b_world)
                if okw2 and w2 and mr.eq(w2, p) then out.world[#out.world + 1] = off end
            end
        end
    end
    return out
end

local function num(mr, v)
    if type(v) == 'number' or v == nil then return v end
    return mr.tonumber and tonumber(mr.tonumber(v)) or tostring(v)
end

-- The checks, then the hooks; the manager is found by poll() (at the load the battle may not be built yet). Returns
-- handle, status (handle nil when a check failed: the probe runs without it).
function M.attach(mr, cfg)
    local status = M.check(mr, cfg or {})
    if not status.ok then return nil, status end
    for _, name in ipairs({'ammo_left', 'battle_ref', 'battle_global'}) do
        if not cfg.sites[name] then
            status.ok, status.why = false, 'no ' .. name .. ' site in the config'
            return nil, status
        end
    end
    local L = cfg.layout
    local okm, _, diag = pcall(M.manager, mr, cfg)
    status.chain = okm and diag or nil
    local S = M.scanner(mr, L, cfg.snapshot_max)
    local last_unit, chain, found, u_off = nil, nil, nil, nil
    local at_update = M.address(mr, cfg.sites.proj_update)
    local at_ammo = M.address(mr, cfg.sites.ammo_left)
    local hook_max = cfg.hook_max or 40
    local ammo, ammo_calls = {}, 0
    -- proj_update: a plain hook (its callback never runs - the simulation thread - so it never has to call the game)
    local function on_update(m)
        S.update_callbacks = (S.update_callbacks or 0) + 1
        return mr.hook_next(at_update, m)
    end
    -- ammo_left is a script binding: it checks the Lua call's arguments, so the game's function must not be called from
    -- inside a Lua callback (hook_next: "bad argument #1 to 'hook_next' (expecting zero arguments)"); a patcher prefix
    -- reads and returns, memreader then calls the game's function itself
    local function on_ammo(_, obj)
        ammo_calls = ammo_calls + 1
        if ammo_calls <= hook_max then
            local ok, row = pcall(function()
                local unit = mr.read_pointer(obj, L.i_unit)
                last_unit = unit
                return {mr.tostring(unit), mr.read_int32(unit, L.u_ammo)}
            end)
            ammo[#ammo + 1] = ok and row or {false, tostring(row)}
        end
    end
    local hooked, patcher = {}, nil
    local okh, errh = pcall(mr.hook, at_update, cfg.sites.proj_update.sig, on_update)
    if okh then hooked[#hooked + 1] = at_update end
    status.proj_update_hooked = okh or tostring(errh)
    if type(mr.patcher) == 'function' then
        local okp, errp = pcall(function()
            patcher = assert(mr.patcher('tww3_bai_probe'))
            patcher:prefix(at_ammo, cfg.sites.ammo_left.sig, 'ammo_left', on_ammo)
        end)
        status.ammo_left_hooked = okp or tostring(errp)
    else
        status.ammo_left_hooked = 'memreader without patcher'
    end
    status.hooked = okh
    local handle = {scanner = S}
    function handle.poll()
        if last_unit and not found then             -- once: where the unit keeps its battle (or its world)
            local okd, d = pcall(M.discover, mr, L, last_unit, cfg.discover_max)
            found = okd and d or {error = tostring(d)}
            u_off = okd and d.battle[1] or nil
        end
        local okc, m, d = pcall(M.manager, mr, cfg, last_unit, u_off)
        if not okc then m, d = nil, {error = tostring(m)} end
        chain = d
        if not m then return end
        local ok, err = pcall(S.scan, m)
        if not ok and #S.errors < 20 then S.errors[#S.errors + 1] = tostring(err) end
    end
    function handle.drain()
        local d = S.drain()
        d.ammo, ammo = ammo, {}
        d.chain, d.discover = chain, found
        return d
    end
    function handle.stats()
        local info = mr.hook_info(at_update) or {}
        return {scans = S.calls, projectiles = S.next_id, ammo_calls = ammo_calls,
                update_callbacks = S.update_callbacks or 0,
                proj_update = {calls = num(mr, info.calls), other_thread_calls = num(mr, info.other_thread_calls),
                               attached = info.attached, error = info.error and tostring(info.error) or nil}}
    end
    function handle.detach()
        for _, at in ipairs(hooked) do pcall(mr.unhook, at) end
        hooked = {}
        if patcher then pcall(function() patcher:remove_all() end) end
        patcher = nil
    end
    return handle, status
end

return M

-- What a projectile 'in a man' is in, read from the game's memory (memreader Plus; bridge/docs/apps/memreader.md). PROBES
-- ONLY, read only. The game puts a NEW projectile object where a projectile struck a body (first state 2, its shooter
-- kept; apps.mem.projectiles); this finds the struck body among the pointers of that object's own memory.
--
-- new(mr, L, ids, opts) -> H; ids: the men's table address text -> id (apps.mem.entities reader R.ids: every man of the
-- bound units seen by a poll). H.who(p) -> a flat list {offset, id, offset, id, ...} of p's own fields (0 .. max_off,
-- step 8) that hold a pointer to a known man; a pointer at a LEARNED offset that is no known man is an 'other' body
-- (an engine: a gun of an artillery unit is no man of its unit's vector) -> id -k (k its number). Learning: the first
-- learn_n objects are read whole; then only the offsets that held a known man in at least half of them.
-- An other body is read ONCE when first seen (its own memory, as a man's: m_unit -> the bound unit's name, m_hp,
-- m_pos), at most other_max of them -> H.drain_other() {{k, address, unit name or false, hp, x, y, z}}; H.poll_other()
-- reads their hp again (the probe calls it once a second) -> {{k, hp}}. No pointer found in a projectile is followed
-- further than that one body (the first men's search that followed every pointer broke the game's Lua after ~10^5
-- reads: apps.mem.entities).
local M = {}

M.MAX_OFF = 0x200
M.LEARN_N = 200
M.OTHER_MAX = 16

local function r1(v)
    if type(v) ~= 'number' or v ~= v then return false end
    return math.floor(v * 10 + 0.5) / 10
end

function M.new(mr, L, ids, opts)
    opts = opts or {}
    local H = {seen = 0, count = {}, learned = nil, other = {}, n_other = 0, other_new = {}, units = opts.units or {}}
    local max_off = opts.max_off or M.MAX_OFF
    local learn_n = opts.learn_n or M.LEARN_N
    local other_max = opts.other_max or M.OTHER_MAX

    -- the bound unit's name of a battle unit's address (opts.units: {name = address text})
    local function unit_name(u)
        if not u then return false end
        local key = mr.tostring(u)
        for name, a in pairs(H.units) do if a == key then return name end end
        return false
    end

    local function read_other(q, key)
        H.n_other = H.n_other + 1
        local k = H.n_other
        local row = {k = k, ptr = q}
        H.other[key] = row
        local okr, s = pcall(mr.read_struct, q, 0, {u = {L.m_unit, 'pointer'}, hp = {L.m_hp, 'uint32'},
            x = {L.m_pos, 'float'}, y = {L.m_pos + 4, 'float'}, z = {L.m_pos + 8, 'float'}})
        if okr and s then
            H.other_new[#H.other_new + 1] = {k, key, unit_name(s.u), s.hp or false, r1(s.x), r1(s.y), r1(s.z)}
        else
            H.other_new[#H.other_new + 1] = {k, key, false, false, false, false, false}
        end
        return k
    end

    local function learn()
        local out = {}
        for off, n in pairs(H.count) do
            if n * 2 >= H.seen then out[#out + 1] = off end
        end
        table.sort(out)
        H.learned = out
    end

    function H.who(p)
        local out = {}
        if H.learned == nil then
            H.seen = H.seen + 1
            for off = 0, max_off - 8, 8 do
                local ok, q = pcall(mr.read_pointer, p, off)
                if ok and q and not mr.is_null(q) then
                    local id = ids[mr.tostring(q)]
                    if id then
                        out[#out + 1] = off
                        out[#out + 1] = id
                        H.count[off] = (H.count[off] or 0) + 1
                    end
                end
            end
            if H.seen >= learn_n then learn() end
            return out
        end
        for _, off in ipairs(H.learned) do
            local ok, q = pcall(mr.read_pointer, p, off)
            if ok and q and not mr.is_null(q) then
                local key = mr.tostring(q)
                local id = ids[key]
                if not id then
                    local o = H.other[key]
                    if o then
                        id = -o.k
                    elseif H.n_other < other_max then
                        id = -read_other(q, key)
                    end
                end
                if id then
                    out[#out + 1] = off
                    out[#out + 1] = id
                end
            end
        end
        return out
    end

    function H.drain_other()
        local out = H.other_new
        H.other_new = {}
        return out
    end

    function H.poll_other()
        local out = {}
        for _, o in pairs(H.other) do
            local ok, s = pcall(mr.read_struct, o.ptr, 0, {hp = {L.m_hp, 'uint32'}})
            out[#out + 1] = {o.k, ok and s and s.hp or false}
        end
        table.sort(out, function(a, b) return a[1] < b[1] end)
        return out
    end

    function H.offsets() return H.learned end
    return H
end

return M

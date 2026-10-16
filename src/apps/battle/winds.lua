-- The Winds of Magic of an army (the pool its spellcasters pay from; docs/en/game/mechanics/magic.md).
-- The engine's army readers seen in the game's own scripts: winds_of_magic_current (benchmark
-- blackfire_pass) and winds_of_magic_remaining_recharge_rate (wh_battle_advice); the others in
-- M.READERS are guesses tried once (M.discover) and kept only if they answer with a number.
-- The writers (lib_generated_battle: add_winds_of_magic_on_message / _reserve_on_message) add
-- an amount: modify_winds_of_magic_current(x), modify_winds_of_magic_reserve(x).
local M = {}

-- engine reader -> the short name in logs and state documents
M.READERS = {
    {'winds_of_magic_current', 'cur'},
    {'winds_of_magic_remaining_recharge_rate', 'rate'},
    {'winds_of_magic_reserve', 'res'},
    {'winds_of_magic_reserve_current', 'res2'},
    {'winds_of_magic_max', 'max'},
    {'winds_of_magic_capacity', 'cap'},
    {'winds_of_magic_recharge_rate', 'rr'},
}
-- (the reserve: winds_of_magic_reserve answers in battle, the magic probe build/magic/probe1.md; the network sees it)
M.DEFAULT = {'cur', 'rate', 'res'}

local function call(army, method)
    local ok, v = pcall(function() return army[method](army) end)
    if ok and type(v) == 'number' and v == v then return v end
    return nil
end

-- The short names of the readers that answer on this army now (a list in M.READERS order).
function M.discover(army)
    local out = {}
    for _, r in ipairs(M.READERS) do
        if army and call(army, r[1]) ~= nil then out[#out + 1] = r[2] end
    end
    return out
end

-- {short name = value} of the given readers (default M.DEFAULT); nil when none answers.
function M.read(army, names)
    if not army then return nil end
    local want, out, any = {}, {}, false
    for _, n in ipairs(names or M.DEFAULT) do want[n] = true end
    for _, r in ipairs(M.READERS) do
        if want[r[2]] then
            local v = call(army, r[1])
            if v ~= nil then
                out[r[2]] = math.floor(v * 1000 + 0.5) / 1000
                any = true
            end
        end
    end
    if any then return out end
    return nil
end

-- Adds current / reserve (either may be nil or 0) to the army's winds; returns true, or false and the error.
function M.modify(army, current, reserve)
    local ok, e = pcall(function()
        if current and current ~= 0 then army:modify_winds_of_magic_current(current) end
        if reserve and reserve ~= 0 then army:modify_winds_of_magic_reserve(reserve) end
    end)
    if ok then return true end
    return false, tostring(e)
end

return M

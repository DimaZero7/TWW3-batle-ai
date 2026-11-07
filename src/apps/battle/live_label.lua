-- The live labelling (tools.build human --live-label; tools/label/live.py, build/diag_chain/live_label.md): a person plays
-- the battle as usual - no pause, his own orders - and labels the operative's level at the same time: his own groups are
-- the game's (Ctrl+G or the group button: read from the cards panel), the enemy's groups are made on the enemy's cards
-- (apps.battle.enemy_cards). Every second of battle time the state document (the bridge's format,
-- tww3_bai_nn_state.json: every unit's row) goes out with
--   groups = {own = {{k, {names}}}, enemy = {{k, {names}}}}, selection = {own unit names selected}, live = true
-- and the companion's recorder cuts it into 15 s windows. Nothing here commands a unit.
-- Own groups (read_own): the cards panel's tab_group_over_card has group_frame0..9 over the cards of each group; the own
-- cards (review_DY's children) are named by the unit's unique ui id; a card under a frame is in that group. Found by
-- name only (find_uicomponent): a ChildCount / Find walk breaks the game's script bindings for good.
-- Every shortcut pressed goes out as the event live_shortcut with the own selection then (the control groups Ctrl+1..9:
-- their shortcut names are learnt from it).
-- Test commands (config.live_test; the scripted check, no desktop input): tww3_bai_live_cmd.txt, a line each -
--   click_own <unit> | dbl_own <unit> | click_enemy <unit> | button_group | move <unit> <x> <z> (a script order through
--   a controller of its own, released at once: the stream of the player's orders shows it).
local exchange = require('apps.bridge.exchange_adapter')
local json = require('apps.core.json')
local services = require('apps.bridge.services')
local enemy_cards = require('apps.battle.enemy_cards')

local M = {}

M.STATE = 'tww3_bai_nn_state.json'
M.CMD = 'tww3_bai_live_cmd.txt'
M.FRAMES = 10
-- the unit's highlight on the field (the game's own: a strip under the unit in its side's colour, as on a hover): the
-- CCO command ToggleHighlight shows it, SetHighlight(true) / Highlight(true) do nothing (the check battle 07.11 17:29,
-- build/diag_chain/ingame_live_6_outline_1.png); IsHighlighted says whether it is on
M.HIGHLIGHT = 'ToggleHighlight'
M.HIGHLIGHTED = 'IsHighlighted'

local function box(uic)
    local okp, x, y = pcall(function() return uic:Position() end)
    local okb, w, h = pcall(function() return uic:Bounds() end)
    return {x = okp and x or 0, y = okp and y or 0, w = okb and w or 0, h = okb and h or 0}
end

-- {{k, {names}}} of the own groups the cards panel shows (module doc); own = {{name, unit}}
function M.read_own(env, own)
    local f = function(...) return env.find_uicomponent(...) end
    local root = env.core:get_ui_root()
    local review, over = f(root, 'cards_panel', 'review_DY'), f(root, 'cards_panel', 'tab_group_over_card')
    if not (review and over) then return {} end
    local cards = {}
    for _, it in ipairs(own) do
        local c = f(review, tostring(it.unit:unique_ui_id()))
        local vis = c and select(2, pcall(function() return c:Visible() end))
        if c and vis ~= false then
            local b = box(c)
            b.name = it.name
            if b.w > 0 then cards[#cards + 1] = b end
        end
    end
    local out = {}
    for k = 0, M.FRAMES - 1 do
        local fr = f(over, 'group_frame' .. k)
        local vis = fr and select(2, pcall(function() return fr:Visible() end))
        if fr and vis ~= false then
            local b = box(fr)
            local names = {}
            for _, c in ipairs(cards) do
                if c.y >= b.y - 3 and c.y <= b.y + b.h + 3 and c.x >= b.x - 3 and c.x + c.w <= b.x + b.w + 3 then
                    names[#names + 1] = c.name
                end
            end
            if #names > 0 then out[#out + 1] = {k + 1, names} end
        end
    end
    return out
end

-- opts: bm, env {core, UIComponent, find_uicomponent, cco, common}, own = {{name, unit, key}}, enemy = {...},
-- rows() (net_rows: every unit's row), meta (the state's meta), emit, now_ms(), test (bool), orders (apps.orders.adapter),
-- army (the own army: a test order's controller), vec(x, z)
function M.start(opts)
    local env = opts.env
    local h = {move = 0}
    local by_name_all = {}
    for _, it in ipairs(opts.own) do by_name_all[it.name] = it end
    for _, it in ipairs(opts.enemy) do by_name_all[it.name] = it end
    -- the highlight of the picked enemy units on the field: toggled where it is not as wanted (IsHighlighted; unread -
    -- our own memory of it), the results logged (event live_highlight)
    local lit = {}
    local function highlight(names, on)
        local res = {}
        for _, name in ipairs(names) do
            local it = by_name_all[name]
            if it then
                local id = tostring(it.unit:unique_ui_id())
                local okr, now = pcall(function() return env.common.get_context_value('CcoBattleUnit', id, M.HIGHLIGHTED) end)
                if not okr or type(now) ~= 'boolean' then now = lit[name] == true end
                if now ~= on then
                    local ok, err = pcall(function() return env.common.call_context_command('CcoBattleUnit', id, M.HIGHLIGHT) end)
                    if ok then lit[name] = on end
                    res[#res + 1] = name .. ' ' .. M.HIGHLIGHT .. ': ' .. (ok and 'ok' or tostring(err))
                else
                    res[#res + 1] = name .. ' already ' .. tostring(on)
                end
            end
        end
        opts.emit('live_highlight', {t = opts.now_ms(), on = on, results = res})
    end
    h.cards = enemy_cards.start({bm = opts.bm, env = env, own = opts.own, enemy = opts.enemy, emit = opts.emit,
        now_ms = opts.now_ms, highlight = highlight})
    local by_name = {}
    for _, it in ipairs(opts.own) do by_name[it.name] = it end
    for _, it in ipairs(opts.enemy) do by_name[it.name] = it end
    exchange.remove(M.STATE)
    exchange.remove(M.CMD)

    local function selection()
        local out = {}
        for _, it in ipairs(opts.own) do
            local ok, v = pcall(function()
                return env.common.get_context_value('CcoBattleUnit', tostring(it.unit:unique_ui_id()), 'IsSelected')
            end)
            if ok and v == true then out[#out + 1] = it.name end
        end
        return out
    end

    local function run(line)
        local w = {}
        for x in line:gmatch('%S+') do w[#w + 1] = x end
        local verb, row = w[1], {t = opts.now_ms(), line = line}
        local f = function(...) return env.find_uicomponent(...) end
        local function click(uic, dbl)
            if not uic then return 'no component' end
            local ok, err = pcall(function() if dbl then uic:SimulateDblLClick() else uic:SimulateLClick() end end)
            return ok and 'ok' or tostring(err)
        end
        if verb == 'click_own' or verb == 'dbl_own' then
            local it = by_name[w[2]]
            local card = it and f(env.core:get_ui_root(), 'cards_panel', 'review_DY', tostring(it.unit:unique_ui_id()))
            row.result = click(card, verb == 'dbl_own')
        elseif verb == 'click_enemy' then
            row.result = h.cards.click(w[2], w[3])          -- (w[3]: a child of the card - a person's click lands there)
        elseif verb == 'cco' then
            -- a CCO command on a unit: cco <unit> <command> (the outlines' probe)
            local it = by_name[w[2]]
            local ok, err = pcall(function()
                return env.common.call_context_command('CcoBattleUnit', tostring(it.unit:unique_ui_id()),
                    line:match('^cco%s+%S+%s+(.+)$'))
            end)
            row.result = ok and ('ok ' .. tostring(err)) or tostring(err)
        elseif verb == 'eval' then
            local it = by_name[w[2]]
            local ok, v = pcall(function()
                return env.common.get_context_value('CcoBattleUnit', tostring(it.unit:unique_ui_id()),
                    line:match('^eval%s+%S+%s+(.+)$'))
            end)
            row.result = ok and tostring(v) or ('error: ' .. tostring(v))
        elseif verb == 'click_tab' then
            local fr = h.cards.frames[tonumber(w[2])]
            row.result = click(fr and fr.tab)
        elseif verb == 'button_group' then
            row.result = click(f(env.core:get_ui_root(), 'battle_orders', 'button_group'))
        elseif verb == 'move' then
            local it = by_name[w[2]]
            local ok, err = pcall(function()
                local uc = opts.orders.take_control(opts.army, it.unit)
                opts.orders.move(uc, opts.vec(tonumber(w[3]), tonumber(w[4])), false)
                opts.orders.release(uc)
            end)
            row.result = ok and 'ok' or tostring(err)
        end
        opts.emit('live_cmd', row)
    end

    -- every shortcut the player presses, with the own selection at that moment (event live_shortcut): the control
    -- groups Ctrl+1..9 are not in the UI or the CCO, but the key's shortcut and the selection it binds are seen here
    pcall(function()
        env.core:add_listener('bai_live_shortcuts', 'ShortcutPressed', true, function(c)
            opts.emit('live_shortcut', {t = opts.now_ms(), string = tostring(c.string), selection = selection()})
        end, true)
    end)

    function h.tick()
        if opts.test then
            local text = exchange.read(M.CMD)
            if text and text ~= '' then
                exchange.remove(M.CMD)
                for line in text:gmatch('[^\r\n]+') do pcall(run, line) end
            end
        end
        h.cards.tick()
        h.move = h.move + 1
        local okg, own = pcall(M.read_own, env, opts.own)
        local enemy = {}
        for k = 1, enemy_cards.GROUPS do
            local g = h.cards.groups[k]
            if g and #g > 0 then enemy[#enemy + 1] = {k, g} end
        end
        local doc = services.state_document(opts.meta, h.move, opts.now_ms(), opts.rows(), false, {}, {}, nil)
        doc.live = true
        doc.groups = {own = okg and own or {}, enemy = enemy}
        doc.selection = selection()
        if not okg then doc.groups_error = tostring(own) end
        exchange.write(M.STATE, json.encode(doc))
        return doc
    end

    function h.finish()
        local doc = services.state_document(opts.meta, h.move, opts.now_ms(), opts.rows(), true, {}, {}, nil)
        doc.live = true
        exchange.write(M.STATE, json.encode(doc))
    end

    return h
end

return M

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
local box_pick = require('apps.battle.box_pick')

local M = {}

M.STATE = 'tww3_bai_nn_state.json'
M.CMD = 'tww3_bai_live_cmd.txt'
M.FRAMES = 10
-- the unit's highlight on the field (the game's own: a strip under the unit in its side's colour, as on a hover): the
-- CCO command ToggleHighlight shows it, SetHighlight(true) / Highlight(true) do nothing (the check battle 07.11 17:29,
-- build/diag_chain/ingame_live_6_outline_1.png); IsHighlighted says whether it is on
M.HIGHLIGHT = 'ToggleHighlight'
M.HIGHLIGHTED = 'IsHighlighted'
-- the enemy's pick on the field: 'ring' - a yellow ring over the unit's banner (an icon under its script_ping_parent, as
-- CA's script_unit:add_ping_icon; steady). ToggleHighlight ('cco') and a unit controller's highlight ('uc') both pulse
-- on and off about once a second - the picked enemies blinked (the check battles 07.11 18:30, 18:33:
-- build/diag_chain/strip_cco.png, strip_uc2.png)
M.LIT_MODE = 'ring'
M.RING_IMAGE = 'ui/bai/label/pick_ring.png'
M.CLEAR_IMAGE = 'ui/bai/label/clear.png'
M.PING_LAYOUT = 'ui/battle ui/unit_ping_indicator.twui.xml'

-- The modifier keys held now (the host's helper tools/label/keys.ps1 writes M.KEYS: 'ctrl=0|1 shift=0|1 t=<unix s>'
-- on every change and twice a second; a script sees no keys of its own). No file or an old one (the helper gone): none.
M.KEYS = 'tww3_bai_keys.txt'
-- (no clock check: the game's os.time is not the helper's - a file in force read as stale, the check battle 07.11 18:30;
-- the helper removes its file when it ends, live.ps1 after the game)
function M.keys()
    local text = exchange.read(M.KEYS)
    if not text or text == '' then return {} end
    return {ctrl = text:match('ctrl=(%d)') == '1', shift = text:match('shift=(%d)') == '1', text = text}
end

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
    -- the mode: 'cco' (ToggleHighlight: the strip pulses - 5 shots in 1.25 s, one without it: the check battle 07.11
    -- 18:24) or 'uc' (a unit controller of the enemy's army, never taking control: highlight(true))
    h.lit_mode = M.LIT_MODE
    local ucs, rings = {}, {}
    local function uc_of(it)
        if not ucs[it.name] then
            local uc = opts.enemy_army:create_unit_controller()
            uc:add_units(it.unit)
            ucs[it.name] = uc
        end
        return ucs[it.name]
    end
    local function highlight(names, on)
        local res = {}
        for _, name in ipairs(names) do
            local it = by_name_all[name]
            if it and h.lit_mode == 'ring' then
                local ok, err = pcall(function()
                    local r = rings[name]
                    if not r then
                        local f = env.find_uicomponent
                        local holder = f(env.core:get_ui_root(), 'unit_id_holder', tostring(it.unit:unique_ui_id()),
                            'script_ping_parent')
                        assert(holder, 'no script_ping_parent')
                        r = env.UIComponent(holder:CreateComponent('bai_pick_ring_' .. name, M.PING_LAYOUT))
                        if env.cco then pcall(function() r:SetContextObject(env.cco('CcoBattleUnit', it.unit:unique_ui_id())) end) end
                        local icon = f(r, 'icon')
                        assert(icon, 'no icon')
                        -- (the indicator's icon has 3 images - an oval's base, its yellow fill, the picture: ours as 0, the
                        -- others clear, else a yellow pin stands over the unit - the check battle 07.11 18:37)
                        icon:SetImagePath(M.RING_IMAGE, 0)
                        local okn, n = pcall(function() return icon:NumImages() end)
                        for i = 1, (okn and type(n) == 'number' and n or 1) - 1 do
                            pcall(function() icon:SetImagePath(M.CLEAR_IMAGE, i) end)
                        end
                        local a = f(r, 'arrow')
                        if a then pcall(function() a:SetVisible(false) end) end
                        pcall(function() r:SetInteractive(false); icon:SetInteractive(false) end)
                        rings[name] = r
                    end
                    r:SetVisible(on)
                end)
                res[#res + 1] = name .. ' ring ' .. tostring(on) .. ': ' .. (ok and 'ok' or tostring(err))
            elseif it and h.lit_mode == 'uc' and opts.enemy_army then
                local ok, err = pcall(function() uc_of(it):highlight(on) end)
                res[#res + 1] = name .. ' uc ' .. tostring(on) .. ': ' .. (ok and 'ok' or tostring(err))
            elseif it then
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
        now_ms = opts.now_ms, highlight = highlight, keys = M.keys})
    -- the box over the field picks enemies too (apps.battle.box_pick)
    local all = {}
    for _, it in ipairs(opts.own) do all[#all + 1] = it end
    for _, it in ipairs(opts.enemy) do all[#all + 1] = it end
    local okb, box = pcall(box_pick.start, {bm = opts.bm, env = env, units = opts.enemy, all = all, emit = opts.emit,
        now_ms = opts.now_ms, on_box = function(names) h.cards.box_pick(names, 'box') end,
        on_clear = function(why) h.cards.clear_pick(why) end,
        on_unit = function(name) h.cards.click_unit(name, 'field banner', true) end})
    if okb then h.box = box else opts.emit('box_probe', {error = tostring(box)}) end
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
        elseif verb == 'reval' then
            -- a CcoBattleRoot expression: reval <expr> (the cursor's probe)
            local ok, v = pcall(function()
                return env.common.get_context_value('CcoBattleRoot', '', line:match('^reval%s+(.+)$'))
            end)
            row.result = ok and tostring(v) or ('error: ' .. tostring(v))
        elseif verb == 'ucall' then
            -- ucall <unit> <method> [args]: a method of the script unit (the probe: select_in_ui, mark_as_ally,
            -- add_animated_selection_proxy, is_selectable, is_controllable, is_player_controlled ...)
            local it = by_name[w[2]]
            local args = {}
            for i = 4, #w do
                local a = w[i]
                if a == 'true' then a = true elseif a == 'false' then a = false elseif tonumber(a) then a = tonumber(a) end
                args[#args + 1] = a
            end
            local ok, a1, a2 = pcall(function() return it.unit[w[3]](it.unit, unpack(args)) end)
            row.result = ok and (tostring(a1) .. (a2 ~= nil and (' ' .. tostring(a2)) or '')) or ('error: ' .. tostring(a1))
        elseif verb == 'bcall' then
            -- bcall <method> [args]: a method of the battle (bm.battle)
            local args = {}
            for i = 3, #w do
                local a = w[i]
                if a == 'true' then a = true elseif a == 'false' then a = false elseif tonumber(a) then a = tonumber(a) end
                args[#args + 1] = a
            end
            local ok, a1 = pcall(function() local b = opts.bm.battle; return b[w[2]](b, unpack(args)) end)
            row.result = ok and tostring(a1) or ('error: ' .. tostring(a1))
        elseif verb == 'lit_mode' then
            h.lit_mode = w[2]
            row.result = 'ok'
        elseif verb == 'uc_lit' then
            -- uc_lit <unit> on|off: the unit controller's highlight alone (the check of its look)
            local it = by_name[w[2]]
            local ok, err = pcall(function()
                h.test_ucs = h.test_ucs or {}
                local uc = h.test_ucs[w[2]]
                if not uc then
                    uc = opts.enemy_army:create_unit_controller()
                    uc:add_units(it.unit)
                    h.test_ucs[w[2]] = uc
                end
                uc:highlight(w[3] == 'on')
            end)
            row.result = ok and 'ok' or tostring(err)
        elseif verb == 'fit' and h.box then
            local okc, b = pcall(h.box.calibrate, 'test')
            if not okc then row.error = tostring(b); b = nil end
            row.result = (not okc) and 'error' or b and string.format('fov %.1f hb %.1f dx %.1f dy %.1f err %.1f flip %d n %d',
                b.fov, b.hb, b.dx, b.dy, b.err, b.flip, b.n) or 'no fit'
        elseif verb == 'poly' and h.box then
            -- poly <unit>...: the units' formations on the screen and their banners (the check over a screenshot)
            local out = {}
            local pts = {}
            for _, p in ipairs(h.box.points(all)) do pts[p[1]] = {math.floor(p[2]), math.floor(p[3])} end
            for i = 2, #w do
                local q = h.box.polygon(w[i])
                out[#out + 1] = {unit = w[i], box = q and q.box, centre = q and q.centre, banner_fit = q and q.banner_fit,
                    banner = pts[w[i]]}
            end
            row.result = json.encode(out)
        elseif verb == 'box_rect' and h.box then
            -- box_rect x y w h: a box's end with this rectangle (the drag is the person's)
            local got = h.box.finish({x = tonumber(w[2]), y = tonumber(w[3]), w = tonumber(w[4]), h = tonumber(w[5])},
                'test')
            row.result = table.concat(got, ' ')
        elseif verb == 'click_banner' then
            -- click_banner <unit>: the unit's banner over the field (unit_id_holder's component of its id)
            local it = by_name[w[2]]
            row.result = click(it and f(env.core:get_ui_root(), 'unit_id_holder', tostring(it.unit:unique_ui_id())))
        elseif verb == 'root_click' then
            row.result = click(env.core:get_ui_root())
        elseif verb == 'box_units' and h.box then
            -- a box around these units' banners (the drag itself is the person's: a script cannot drag) -> the pick
            local want, x0, y0, x1, y1 = {}, nil, nil, nil, nil
            for i = 2, #w do want[w[i]] = true end
            for _, p in ipairs(h.box.points()) do
                if want[p[1]] then
                    x0, y0 = math.min(x0 or p[2], p[2]), math.min(y0 or p[3], p[3])
                    x1, y1 = math.max(x1 or p[2], p[2]), math.max(y1 or p[3], p[3])
                end
            end
            if x0 then
                local got = h.box.finish({x = x0 - 6, y = y0 - 6, w = x1 - x0 + 12, h = y1 - y0 + 12}, 'test')
                row.result = table.concat(got, ' ')
            else
                row.result = 'no banners'
            end
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

    -- the game's own battle events to a script (battle:register_command_handler / register_unit_selection_handler through
    -- CA's battle_manager wrappers, which keep CA's own): the player's commands (Group Created, Attack Unit, Entity Hit,
    -- Double Click ...) and a unit's selection, every unit - own and enemy (events game_cmd, game_sel)
    local told = 0
    local function tell(kind, row)
        if told >= 400 then return end
        told = told + 1
        opts.emit(kind, row)
    end
    local uid_name = {}
    for _, it in ipairs(opts.own) do uid_name[tostring(it.unit:unique_ui_id())] = it.name end
    for _, it in ipairs(opts.enemy) do uid_name[tostring(it.unit:unique_ui_id())] = it.name end
    local function unit_name(u)
        local ok, id = pcall(function() return tostring(u:unique_ui_id()) end)
        return ok and (uid_name[id] or id) or nil
    end
    M.COMMANDS = {'Double Click', 'Entity Hit', 'Double Click Unit Card', 'Group Created', 'Group Destroyed',
        'Move Orientation Width', 'Move Rotation Angle', 'Change Speed', 'Attack Unit', 'Change Skirmish', 'Change Melee',
        'Change Formation', 'Special Ability', 'Fire At Will'}
    h.game_events = {}
    -- (the check only: Entity Hit is the combat's hits, some 170 a minute; the selection - the own units only; an enemy's
    -- select_in_ui is ignored - the check battle 07.11 18:43)
    for _, name in ipairs(opts.test and M.COMMANDS or {}) do
        local ok, err = pcall(function()
            opts.bm:register_command_handler_callback(name, function(ctx)
                local row = {t = opts.now_ms(), name = name}
                pcall(function() row.bool1 = ctx:get_bool1() end)
                pcall(function() row.string1 = ctx:get_string1() end)
                pcall(function() row.unit = unit_name(ctx:get_unit()) end)
                tell('game_cmd', row)
                if h.on_game_cmd then pcall(h.on_game_cmd, row) end
            end, 'bai_live_' .. name)
        end)
        h.game_events[name] = ok and 'ok' or tostring(err)
    end
    for _, it in ipairs(opts.test and all or {}) do
        local ok = pcall(function()
            opts.bm:register_unit_selection_callback(it.unit, function(_, selected)
                tell('game_sel', {t = opts.now_ms(), unit = it.name, selected = selected})
                if h.on_game_sel then pcall(h.on_game_sel, it.name, selected) end
            end)
        end)
        h.game_events.selection = (h.game_events.selection or 0) + (ok and 1 or 0)
    end
    if opts.test then opts.emit('game_events', {t = opts.now_ms(), registered = h.game_events}) end

    function h.tick()
        if opts.test then
            local text = exchange.read(M.CMD)
            if text and text ~= '' then
                exchange.remove(M.CMD)
                for line in text:gmatch('[^\r\n]+') do
                    local ok, err = pcall(run, line)
                    if not ok then opts.emit('live_cmd', {t = opts.now_ms(), line = line, result = 'error', error = tostring(err)}) end
                end
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
        if h.box then h.box.stop() end
        local doc = services.state_document(opts.meta, h.move, opts.now_ms(), opts.rows(), true, {}, {}, nil)
        doc.live = true
        exchange.write(M.STATE, json.encode(doc))
    end

    return h
end

return M

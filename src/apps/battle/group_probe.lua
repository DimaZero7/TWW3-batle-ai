-- The probe of the game's own grouping and of the enemy's cards (research, build/diag_chain/live_label.md; tools.build
-- nn-arena --group-probe). No walk over the UI tree: a ChildCount / Find loop breaks the game's script bindings for good
-- (the Malakai project, runs 20261104-16xxxx: every Id() and ChildCount() returns a function afterwards) - components
-- are found by name only (find_uicomponent).
--   every second (a real-time timer): each unit's IsSelected, the selection (CcoBattleRoot SelectionContext), the own
--   groups as the cards panel draws them (ui_groups: tab_group_over_card's group_frame0..9 and the own cards - named by
--   the unit's unique ui id - under each), the enemy groups made here; events 'group_probe' with what changed
--   the player's inputs as the UI reports them: ShortcutPressed / ShortcutTriggered (context.string: e.g.
--   toggle_group = Ctrl+G) and ComponentLClickUp (the component's id) - event 'group_probe_input'
--   the enemy's groups: Ctrl+G (toggle_group) or the group button (button_group) while enemy units are selected
--   (IsSelected) makes them enemy group N (the next free number); a ring with N over each (ui/bai/label/t<N>.png)
--   commands from the host, a line each in tww3_bai_probe_cmd.txt (read and removed every second):
--     cards <enemy unit> ... | cards all_enemies   the game's own land unit card bound to each enemy unit (cco with
--                                         the numeric unique ui id), laid on the own cards' line (layout below)
--     positions                           every card's place and the screen's resolution (event group_probe_positions)
--     click_own <unit> / dbl_own <unit>   SimulateLClick / SimulateDblLClick on the own unit's card
--     click_enemy <unit>                  SimulateLClick on our card of that enemy unit
--     button_group                        SimulateLClick on the orders panel's group button (Ctrl+G's button)
--     key <name>                          ui_root:SimulateKey(name)
--     select <unit>                       the CCO command Select on the unit
--     eval <type> <id or -> <expr>        one CCO expression
-- Every engine call is in pcall; nothing here gives an order.
local exchange = require('apps.bridge.exchange_adapter')

local M = {}

M.CMD = 'tww3_bai_probe_cmd.txt'
M.CARD_LAYOUT = 'ui/common ui/land_unit_card'
M.SEP_IMAGE = 'ui/bai/label/sep.png'
M.RING_IMAGE = 'ui/bai/label/t%d.png'
M.PING_LAYOUT = 'ui/battle ui/unit_ping_indicator.twui.xml'
M.GAP = 26           -- px between the last own card and the first enemy card (the separator in the middle)
M.MARGIN = 510       -- the game's cards strip: screen width less this (review_DY fullwidth_margin)
M.ROW_SHIFT = 127    -- a row above (review_DY double_row_shift)
M.FRAMES = 10        -- group_frame0 .. group_frame9

-- opts: bm, emit, flush, env {common, core, UIComponent, find_uicomponent, cco}, units = {{name, unit, side}},
-- now_ms(), finished(), guard
function M.start(opts)
    local bm, env = opts.bm, opts.env
    local guard = opts.guard or function(fn) return fn end
    local last = {}
    local by_name, by_uid = {}, {}
    for _, it in ipairs(opts.units) do
        by_name[it.name] = it
        by_uid[tostring(it.unit:unique_ui_id())] = it
    end
    local f = function(...) return env.find_uicomponent(...) end
    local root = function() return env.core:get_ui_root() end
    local enemy_row, enemy_card_of = {}, {}
    local enemy_groups, enemy_group_of, rings = {}, {}, {}
    local sep, placed = nil, nil
    local made = 0

    local function value(kind, id, expr)
        local ok, v = pcall(function() return env.common.get_context_value(kind, id, expr) end)
        if not ok then return 'error: ' .. tostring(v) end
        if v == nil then return 'nil' end
        return tostring(v)
    end

    local function box(uic)
        local okp, x, y = pcall(function() return uic:Position() end)
        local okb, w, h = pcall(function() return uic:Bounds() end)
        return {x = okp and x or 0, y = okp and y or 0, w = okb and w or 0, h = okb and h or 0}
    end

    local function review()
        return f(root(), 'cards_panel', 'review_DY')
    end

    local function own_card(it)
        local r = review()
        return r and f(r, tostring(it.unit:unique_ui_id()))
    end

    local function own_cards()
        local out = {}
        for _, it in ipairs(opts.units) do
            if it.side == 1 then
                local c = own_card(it)
                if c then
                    local b = box(c)
                    b.id, b.name = tostring(it.unit:unique_ui_id()), it.name
                    if b.w > 0 then out[#out + 1] = b end
                end
            end
        end
        table.sort(out, function(a, b) return a.y < b.y or (a.y == b.y and a.x < b.x) end)
        return out
    end

    -- 'g<k>:<name>+<name>;...' - the own groups the cards panel shows (module doc)
    local function ui_groups()
        local over = f(root(), 'cards_panel', 'tab_group_over_card')
        if not over then return 'no tab_group_over_card' end
        local cards = own_cards()
        local out = {}
        for k = 0, M.FRAMES - 1 do
            local fr = f(over, 'group_frame' .. k)
            local vis = fr and select(2, pcall(function() return fr:Visible() end))
            if fr and vis ~= false then
                local b = box(fr)
                local members = {}
                for _, c in ipairs(cards) do
                    if c.y >= b.y - 3 and c.y <= b.y + b.h + 3 and c.x >= b.x - 3 and c.x + c.w <= b.x + b.w + 3 then
                        members[#members + 1] = c.name
                    end
                end
                out[#out + 1] = 'g' .. (k + 1) .. ':' .. table.concat(members, '+')
            end
        end
        return table.concat(out, ';')
    end

    local function selected_enemies()
        local out = {}
        for _, it in ipairs(opts.units) do
            if it.side == 2 and value('CcoBattleUnit', tostring(it.unit:unique_ui_id()), 'IsSelected') == 'true' then
                out[#out + 1] = it
            end
        end
        return out
    end

    local function ring_over(it, k)
        local holder = f(root(), 'unit_id_holder')
        local uid = holder and f(holder, tostring(it.unit:unique_ui_id()))
        local parent = uid and f(uid, 'script_ping_parent')
        assert(parent, 'no script_ping_parent')
        if rings[it.name] then pcall(function() rings[it.name]:Destroy() end) end
        made = made + 1
        local uic = env.UIComponent(parent:CreateComponent('bai_ring_' .. made, M.PING_LAYOUT))
        local icon = f(uic, 'icon')
        icon:SetImagePath(string.format(M.RING_IMAGE, k))
        rings[it.name] = uic
    end

    -- the enemy units selected now -> enemy group N (the next number; each unit leaves the group it was in)
    local function make_enemy_group(why)
        local sel = selected_enemies()
        if #sel == 0 then return end
        local k = #enemy_groups + 1
        if k > 6 then k = 6 end
        enemy_groups[k] = enemy_groups[k] or {}
        local names = {}
        for _, it in ipairs(sel) do
            local was = enemy_group_of[it.name]
            if was and enemy_groups[was] then
                for i, n in ipairs(enemy_groups[was]) do
                    if n == it.name then table.remove(enemy_groups[was], i) break end
                end
            end
            enemy_group_of[it.name] = k
            table.insert(enemy_groups[k], it.name)
            names[#names + 1] = it.name
            local ok, err = pcall(ring_over, it, k)
            if not ok then names[#names] = it.name .. ' (ring: ' .. tostring(err) .. ')' end
        end
        opts.emit('group_probe_enemy_group', {t = opts.now_ms(), why = why, group = k, units = names})
    end

    local function enemy_groups_text()
        local out = {}
        for k, g in ipairs(enemy_groups) do
            if #g > 0 then out[#out + 1] = 'e' .. k .. ':' .. table.concat(g, '+') end
        end
        return table.concat(out, ';')
    end

    local function read_all()
        local changed = {}
        local function put(u, e, v)
            local key = u .. '|' .. e
            if last[key] ~= v then
                last[key] = v
                changed[#changed + 1] = {u = u, e = e, v = v}
            end
        end
        local okg, g = pcall(ui_groups)
        put('ui', 'ui_groups', okg and g or ('error: ' .. tostring(g)))
        put('ui', 'enemy_groups', enemy_groups_text())
        for _, it in ipairs(opts.units) do
            put(it.name, 'IsSelected', value('CcoBattleUnit', tostring(it.unit:unique_ui_id()), 'IsSelected'))
        end
        put('root', 'selection', value('CcoBattleRoot', '', 'SelectionContext.UnitList.JoinString(UniqueUiId, ",")'))
        if #changed > 0 then opts.emit('group_probe', {t = opts.now_ms(), changed = changed}) end
    end

    local function resolution()
        local ok, w, h = pcall(function() return env.core:get_screen_resolution() end)
        if ok and w then return w, h end
        return 1600, 1200
    end

    -- The enemy's cards where the game's own strip rule puts cards (hud_battle.twui.xml review_DY, BattleCardPanel:
    -- the strip no wider than the screen less fullwidth_margin 510 px - the general's portrait on the left, the winds
    -- of magic circle on the right - a second row double_row_shift up): after the last own card on the own cards'
    -- bottom row with a separator between, when that row stays inside; else rows of their own above the own cards
    -- (a separator bar under them). The enemy cards take the own cards' size. Every second (the strip changes).
    local function layout()
        if #enemy_row == 0 then return nil end
        local own = own_cards()
        if #own == 0 then return 'no own cards' end
        local resw = resolution()
        local margin = M.MARGIN / 2
        local right = resw - margin
        local wom = f(root(), 'winds_of_magic')
        if wom then
            local wb = box(wom)
            if wb.x > resw / 2 then right = math.min(right, wb.x - 6) end
        end
        local left = margin
        local w, h = own[1].w, own[1].h > 0 and own[1].h or 130
        local step = w
        for k = 2, #own do
            if own[k].y == own[1].y and own[k].x > own[k - 1].x then step = own[k].x - own[k - 1].x break end
        end
        local top, bottom = own[1].y, own[1].y
        for _, c in ipairs(own) do top = math.min(top, c.y); bottom = math.max(bottom, c.y) end
        local lastx = -1
        for _, c in ipairs(own) do if c.y == bottom then lastx = math.max(lastx, c.x) end end
        local n = #enemy_row
        local places, sep_at = {}, nil
        local x0 = lastx + step + M.GAP
        if x0 + n * step <= right then
            for k = 1, n do places[k] = {x0 + (k - 1) * step, bottom} end
            sep_at = {x0 - M.GAP / 2 - 5, bottom, 10, h, 'row'}
        else
            local per = math.max(1, math.floor((right - left) / step))
            local rows = math.ceil(n / per)
            local shift = math.max(M.ROW_SHIFT, h + 12)
            for k = 1, n do
                local r = math.floor((k - 1) / per)
                local in_row = math.min(per, n - r * per)
                local cx = (left + right) / 2 - in_row * step / 2
                places[k] = {cx + ((k - 1) % per) * step, top - shift * (rows - r)}
            end
            local width = math.min(n, per) * step
            sep_at = {(left + right) / 2 - width / 2, top - 9, width, 6, 'above'}
        end
        local key = table.concat({w, h, step, sep_at[1], sep_at[2], places[1][1], places[1][2], n}, ',')
        if key == placed then return nil end
        placed = key
        for k, e in ipairs(enemy_row) do
            local c = e.uic
            local b = box(c)
            if b.w > 0 and math.abs(b.w - w) > 1 then
                pcall(function() c:SetCanResizeWidth(true); c:SetCanResizeHeight(true) end)
                pcall(function() c:Resize(w, h) end)
            end
            pcall(function() c:MoveTo(places[k][1], places[k][2]) end)
        end
        if sep then
            local icon = f(sep, 'icon')
            pcall(function() icon:Resize(sep_at[3], sep_at[4]) end)
            pcall(function() sep:MoveTo(sep_at[1], sep_at[2]) end)
            pcall(function() icon:MoveTo(sep_at[1], sep_at[2]) end)
        end
        return string.format('%s: card %dx%d step %d, own rows y %d..%d, right edge %d (screen %d), enemy at %d,%d',
            sep_at[5], w, h, step, top, bottom, right, resw, places[1][1], places[1][2])
    end

    local function enemy_cards(names)
        local res = {}
        local oks, serr = pcall(function()
            sep = env.UIComponent(root():CreateComponent('bai_enemy_sep', M.PING_LAYOUT))
            local icon = f(sep, 'icon')
            local arrow = f(sep, 'arrow')
            if arrow then pcall(function() arrow:SetVisible(false) end) end
            icon:SetImagePath(M.SEP_IMAGE)
            pcall(function() icon:SetCanResizeWidth(true); icon:SetCanResizeHeight(true) end)
            pcall(function() sep:RegisterTopMost() end)
        end)
        res[#res + 1] = 'separator: ' .. (oks and 'ok' or tostring(serr))
        for _, name in ipairs(names) do
            local it = by_name[name]
            local ok, err = pcall(function()
                made = made + 1
                local c = env.UIComponent(root():CreateComponent('bai_enemy_card_' .. made, M.CARD_LAYOUT))
                -- the unit's numeric unique ui id, as CA's script_unit:add_ping_icon gives cco
                c:SetContextObject(env.cco('CcoBattleUnit', it.unit:unique_ui_id()))
                c:SetVisible(true)
                pcall(function() c:RegisterTopMost() end)
                enemy_row[#enemy_row + 1] = {name = name, uic = c}
                enemy_card_of[name] = c
            end)
            res[#res + 1] = name .. ': ' .. (ok and 'ok' or tostring(err))
        end
        local okl, l = pcall(layout)
        res[#res + 1] = 'layout: ' .. tostring(okl and l or ('error: ' .. tostring(l)))
        return res
    end

    local function positions()
        local w, h = resolution()
        local row = {res = {w, h}, own = own_cards(), enemy = {}}
        for _, e in ipairs(enemy_row) do
            local b = box(e.uic)
            b.name = e.name
            row.enemy[#row.enemy + 1] = b
        end
        opts.emit('group_probe_positions', row)
        return 'ok'
    end

    local function click(uic, dbl)
        if not uic then return 'no component' end
        local ok, err = pcall(function() if dbl then uic:SimulateDblLClick() else uic:SimulateLClick() end end)
        return ok and 'ok' or tostring(err)
    end

    local function run(line)
        local w = {}
        for x in line:gmatch('%S+') do w[#w + 1] = x end
        local verb, row = w[1], {t = opts.now_ms(), line = line}
        if verb == 'cards' then
            local names = {}
            if w[2] == 'all_enemies' then
                for _, it in ipairs(opts.units) do if it.side == 2 then names[#names + 1] = it.name end end
            else
                for i = 2, #w do names[#names + 1] = w[i] end
            end
            row.results = enemy_cards(names)
        elseif verb == 'positions' then
            row.result = positions()
        elseif verb == 'click_own' or verb == 'dbl_own' then
            local it = by_name[w[2]]
            row.result = it and click(own_card(it), verb == 'dbl_own') or 'unknown unit'
        elseif verb == 'click_enemy' then
            row.result = click(enemy_card_of[w[2]])
        elseif verb == 'button_group' then
            row.result = click(f(root(), 'battle_orders', 'button_group'))
        elseif verb == 'key' then
            local ok, err = pcall(function() root():SimulateKey(w[2]) end)
            row.result = ok and 'ok' or tostring(err)
        elseif verb == 'select' then
            local it = by_name[w[2]]
            local ok, err = pcall(function()
                return env.common.call_context_command('CcoBattleUnit', tostring(it.unit:unique_ui_id()), 'Select')
            end)
            row.result = ok and 'ok' or tostring(err)
        elseif verb == 'eval' then
            local expr = line:match('^eval%s+%S+%s+%S+%s+(.+)$')
            row.result = value(w[2], w[3] == '-' and '' or w[3], expr or '')
        end
        opts.emit('group_probe_cmd', row)
    end

    -- the player's inputs as the UI reports them (and the enemy groups: module doc)
    pcall(function()
        for _, ev in ipairs({'ShortcutPressed', 'ShortcutTriggered', 'ComponentLClickUp'}) do
            env.core:add_listener('bai_group_probe_' .. ev, ev, true, guard(function(c)
                local s = tostring(c.string)
                opts.emit('group_probe_input', {t = opts.now_ms(), kind = ev, string = s})
                if (ev == 'ShortcutPressed' and s == 'toggle_group') or (ev == 'ComponentLClickUp' and s == 'button_group') then
                    make_enemy_group(ev .. ' ' .. s)
                end
            end), true)
        end
    end)

    local function tick()
        if opts.finished and opts.finished() then return end
        local text = exchange.read(M.CMD)
        if text and text ~= '' then
            exchange.remove(M.CMD)
            for line in text:gmatch('[^\r\n]+') do
                local ok, err = pcall(run, line)
                if not ok then opts.emit('group_probe_cmd', {t = opts.now_ms(), line = line, error = tostring(err)}) end
            end
        end
        read_all()
        local okl, l = pcall(layout)
        if not okl or l then opts.emit('group_probe_layout', {t = opts.now_ms(), result = tostring(l)}) end
        if opts.flush then opts.flush() end
        bm:real_callback(guard(tick), 1000, 'tww3_bai_group_probe')
    end

    exchange.remove(M.CMD)
    bm:real_callback(guard(tick), 1000, 'tww3_bai_group_probe')
    return {}
end

return M

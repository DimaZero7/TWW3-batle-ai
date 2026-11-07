-- The enemy's cards for the live labelling (build/diag_chain/live_label.md): the game's own unit card layout
-- (ui/common ui/land_unit_card) for every enemy unit, on the own cards' line after a separator, filled by the script -
-- the game fills a card only for the player's own units (its BattleCardPanel / UnitCard callbacks; a context object set
-- on the card is not taken: runs 20261107-153706 .. -16xxxx), so the picture (ui/bai/units/<key>.png in our pack:
-- tools/label/gameart's card of the key), the men (unit_strength_number) and the health bar (health_fill) are set here
-- every second.
--   A click on an enemy card toggles that unit in the enemy pick (the card's 'selected' state); Ctrl+G (the shortcut
--   toggle_group) or the group button (button_group) with a pick makes it enemy group N (the next free number of 1..6;
--   a unit leaves its old group); a click on the separator clears the pick. Nothing here gives an order: an enemy unit
--   never joins the player's selection (SelectionContext.UnitList stays the own units), the cards only feed our groups.
--   Every component is found by name (find_uicomponent), never by a ChildCount / Find walk: that breaks the game's
--   script bindings for good (the Malakai project's finding).
local M = {}

M.CARD_LAYOUT = 'ui/common ui/land_unit_card'
M.PING_LAYOUT = 'ui/battle ui/unit_ping_indicator.twui.xml'
M.SEP_IMAGE = 'ui/bai/label/sep.png'
M.UNIT_IMAGE = 'ui/bai/units/%s.png'
M.GAP = 26           -- px between the last own card and the first enemy card (the separator in the middle)
M.MARGIN = 510       -- the game's cards strip: screen width less this (hud_battle review_DY fullwidth_margin)
M.ROW_SHIFT = 127    -- a row above (review_DY double_row_shift)
M.GROUPS = 6

-- The places of n enemy cards (pure: tests): own = {{x, y, w, h}} the own cards, right / left the strip's edges.
-- -> places {{x, y}}, separator {x, y, w, h, kind}, the card's w, h.
function M.places(own, n, left, right)
    local w, h = own[1].w, own[1].h > 0 and own[1].h or 130
    local step = w
    table.sort(own, function(a, b) return a.y < b.y or (a.y == b.y and a.x < b.x) end)
    for k = 2, #own do
        if own[k].y == own[1].y and own[k].x > own[k - 1].x then step = own[k].x - own[k - 1].x break end
    end
    local top, bottom = own[1].y, own[1].y
    for _, c in ipairs(own) do top = math.min(top, c.y); bottom = math.max(bottom, c.y) end
    local lastx = -1
    for _, c in ipairs(own) do if c.y == bottom then lastx = math.max(lastx, c.x) end end
    local places = {}
    local x0 = lastx + step + M.GAP
    if x0 + n * step <= right then
        for k = 1, n do places[k] = {x0 + (k - 1) * step, bottom} end
        return places, {x0 - M.GAP / 2 - 5, bottom, 10, h, 'row'}, w, h
    end
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
    return places, {(left + right) / 2 - width / 2, top - 9, width, 6, 'above'}, w, h
end

-- opts: bm, env {core, UIComponent, find_uicomponent}, own = {{name, unit}}, enemy = {{name, unit, key}}, emit, now_ms
function M.start(opts)
    local env = opts.env
    local f = function(...) return env.find_uicomponent(...) end
    local root = function() return env.core:get_ui_root() end
    local h = {cards = {}, pick = {}, groups = {}, group_of = {}, made = 0}
    local by_card = {}
    local placed, w0 = nil, {}

    local function box(uic)
        local okp, x, y = pcall(function() return uic:Position() end)
        local okb, w, hh = pcall(function() return uic:Bounds() end)
        return {x = okp and x or 0, y = okp and y or 0, w = okb and w or 0, h = okb and hh or 0}
    end

    local function own_cards()
        local r = f(root(), 'cards_panel', 'review_DY')
        local out = {}
        if not r then return out end
        for _, it in ipairs(opts.own) do
            local c = f(r, tostring(it.unit:unique_ui_id()))
            if c then
                local vis = select(2, pcall(function() return c:Visible() end))
                local b = box(c)
                if b.w > 0 and vis ~= false then out[#out + 1] = b end
            end
        end
        return out
    end

    local function resolution()
        local ok, w, hh = pcall(function() return env.core:get_screen_resolution() end)
        if ok and w then return w, hh end
        return 1600, 1200
    end

    function h.layout()
        local own = own_cards()
        if #own == 0 or #h.cards == 0 then return nil end
        local resw = resolution()
        local right, left = resw - M.MARGIN / 2, M.MARGIN / 2
        local wom = f(root(), 'winds_of_magic')
        if wom then
            local wb = box(wom)
            if wb.x > resw / 2 then right = math.min(right, wb.x - 6) end
        end
        local places, sep, w, hh = M.places(own, #h.cards, left, right)
        local key = table.concat({w, hh, sep[1], sep[2], places[1][1], places[1][2], #h.cards}, ',')
        if key == placed then return nil end
        placed = key
        for k, e in ipairs(h.cards) do
            local b = box(e.uic)
            if b.w > 0 and math.abs(b.w - w) > 1 then
                pcall(function() e.uic:SetCanResizeWidth(true); e.uic:SetCanResizeHeight(true) end)
                pcall(function() e.uic:Resize(w, hh) end)
            end
            pcall(function() e.uic:MoveTo(places[k][1], places[k][2]) end)
        end
        if h.sep then
            local icon = f(h.sep, 'icon')
            pcall(function() icon:Resize(sep[3], sep[4]) end)
            pcall(function() h.sep:MoveTo(sep[1], sep[2]) end)
            pcall(function() icon:MoveTo(sep[1], sep[2]) end)
        end
        return string.format('%s: card %dx%d, screen %d, right %d, first enemy at %d,%d', sep[5], w, hh, resw, right,
            places[1][1], places[1][2])
    end

    -- the card's content: the picture once, the men and the health every call
    local function fill(e)
        local u = e.unit
        if not e.pictured then
            local img = f(e.uic, 'card_image_holder')
            e.pictured = img and pcall(function() img:SetImagePath(string.format(M.UNIT_IMAGE, e.key)) end)
        end
        local men = select(2, pcall(function() return u:number_of_men_alive() end))
        local hp = select(2, pcall(function() return u:unary_hitpoints() end))
        local num = f(e.uic, 'unit_strength_number')
        if num and type(men) == 'number' then pcall(function() num:SetStateText(tostring(men)) end) end
        local bar = f(e.uic, 'health_fill')
        if bar and type(hp) == 'number' then
            if not w0[e.name] then
                local b = box(bar)
                w0[e.name] = {b.w, b.h}
                pcall(function() bar:SetCanResizeWidth(true) end)
            end
            local ww = w0[e.name]
            if ww[1] > 0 then pcall(function() bar:Resize(math.max(1, math.floor(ww[1] * hp + 0.5)), ww[2]) end) end
        end
        local gone = type(men) == 'number' and men <= 0
        pcall(function() e.uic:SetOpacity(gone and 90 or 255, true) end)
    end

    local function show_pick(e)
        local on = h.pick[e.name] == true
        pcall(function() e.uic:SetState(on and 'selected' or 'available') end)
        local frame = f(e.uic, 'selected_frame')
        if frame then pcall(function() frame:SetVisible(on) end) end
    end

    function h.toggle(name)
        local e = h.by_name[name]
        if not e then return end
        h.pick[name] = not h.pick[name] or nil
        show_pick(e)
        opts.emit('enemy_pick', {t = opts.now_ms(), unit = name, on = h.pick[name] == true})
    end

    function h.clear_pick()
        for name in pairs(h.pick) do
            h.pick[name] = nil
            show_pick(h.by_name[name])
        end
    end

    -- the pick -> enemy group N (the next number not in use; 6 at most: then the last)
    function h.group_pick(why)
        local names = {}
        for _, e in ipairs(h.cards) do if h.pick[e.name] then names[#names + 1] = e.name end end
        if #names == 0 then return nil end
        local k = 1
        while h.groups[k] and #h.groups[k] > 0 and k < M.GROUPS do k = k + 1 end
        for _, name in ipairs(names) do
            local was = h.group_of[name]
            if was and h.groups[was] then
                for i, n in ipairs(h.groups[was]) do
                    if n == name then table.remove(h.groups[was], i) break end
                end
            end
        end
        h.groups[k] = names
        for _, name in ipairs(names) do h.group_of[name] = k end
        h.clear_pick()
        opts.emit('enemy_group', {t = opts.now_ms(), why = why, group = k, units = names})
        return k
    end

    function h.tick()
        for _, e in ipairs(h.cards) do pcall(fill, e) end
        local ok, l = pcall(h.layout)
        if not ok or l then opts.emit('enemy_cards_layout', {t = opts.now_ms(), result = tostring(l)}) end
    end

    -- the cards
    h.by_name = {}
    local oks, serr = pcall(function()
        h.sep = env.UIComponent(root():CreateComponent('bai_enemy_sep', M.PING_LAYOUT))
        local icon = f(h.sep, 'icon')
        local arrow = f(h.sep, 'arrow')
        if arrow then pcall(function() arrow:SetVisible(false) end) end
        icon:SetImagePath(M.SEP_IMAGE)
        pcall(function() icon:SetCanResizeWidth(true); icon:SetCanResizeHeight(true) end)
        pcall(function() h.sep:RegisterTopMost() end)
    end)
    local made = {separator = oks and 'ok' or tostring(serr)}
    for i, it in ipairs(opts.enemy) do
        local ok, err = pcall(function()
            local id = 'bai_enemy_card_' .. i
            local c = env.UIComponent(root():CreateComponent(id, M.CARD_LAYOUT))
            c:SetVisible(true)
            pcall(function() c:RegisterTopMost() end)
            local e = {name = it.name, unit = it.unit, key = it.key, uic = c, id = id}
            h.cards[#h.cards + 1] = e
            h.by_name[it.name] = e
            by_card[id] = e
        end)
        if not ok then made[it.name] = tostring(err) end
    end
    h.tick()
    opts.emit('enemy_cards', {t = opts.now_ms(), cards = #h.cards, made = made})

    -- the clicks and the group shortcut
    pcall(function()
        env.core:add_listener('bai_enemy_cards_click', 'ComponentLClickUp', true, function(c)
            local s = tostring(c.string)
            local e = by_card[s]
            if e then
                h.toggle(e.name)
            elseif s == 'bai_enemy_sep' then
                h.clear_pick()
            elseif s == 'button_group' then
                h.group_pick('button_group')
            end
        end, true)
        env.core:add_listener('bai_enemy_cards_shortcut', 'ShortcutPressed', true, function(c)
            if tostring(c.string) == 'toggle_group' then h.group_pick('toggle_group') end
        end, true)
    end)
    -- (a click on the card itself: a SimulateLClick from a test goes the same way)
    function h.click(name)
        local e = h.by_name[name]
        if e then pcall(function() e.uic:SimulateLClick() end) end
    end
    return h
end

return M

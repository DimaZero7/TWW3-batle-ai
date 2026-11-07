-- The enemy's cards for the live labelling (build/diag_chain/live_label.md): the game's own unit card layout
-- (ui/common ui/land_unit_card) for every enemy unit, on the own cards' line after a separator, filled by the script -
-- the game fills a card only for the player's own units (its BattleCardPanel / UnitCard callbacks; a context object set
-- on the card is not taken), so the picture (ui/bai/units/<key>.png in our pack: tools/label/gameart's card of the key),
-- the men (unit_strength_number) and the health bar (health_fill) are set here every second.
--   The pick: a click on an enemy card toggles that unit in it (the card's 'selected' state). A person's click lands on
--   a child of the card (its picture, its bar): the clicked component's parents are followed up to our card
--   (UIComponent:Parent, never a ChildCount / Find walk over the tree - that breaks the game's script bindings for good).
--   Grouping: our group button next to the enemy cards (bai_enemy_group_button, active while the pick has a unit),
--   Ctrl+G (the shortcut toggle_group) or the game's group button with a pick -> enemy group N (the next free of 1..6;
--   a unit leaves its old group). A group is drawn like the game's own: a frame in the group's colour on each of its
--   cards and its number on a tab over the first (our images on the card: the game's group_frame layout places itself
--   by its own callback); a click on the tab picks the whole group. The pick: a yellow frame on the card.
--   opts.highlight(names, on): the units' outlines on the field (a pick or a picked group), when the game lets a script.
--   Nothing here gives an order: the pick never joins the player's selection (SelectionContext stays the own units).
local M = {}

M.CARD_LAYOUT = 'ui/common ui/land_unit_card'
M.FRAME_LAYOUT = 'ui/battle ui/group_frame'
M.BUTTON_LAYOUT = 'ui/templates/round_small_button'
M.PING_LAYOUT = 'ui/battle ui/unit_ping_indicator.twui.xml'
M.SEP_IMAGE = 'ui/bai/label/sep.png'
M.PICK_IMAGE = 'ui/bai/label/pick.png'     -- the pick's frame over a card (the card's own 'selected' look is the game's,
M.HOVER_IMAGE = 'ui/bai/label/hover.png'   -- bound to its unit: it does not show on ours - the person's battle 07.11)
M.GROUP_FRAME_IMAGE = 'ui/bai/label/gf%d.png'   -- a group's frame in its colour (tools/label/icons.py)
M.GROUP_TAB_IMAGE = 'ui/bai/label/g%d.png'      -- its number
M.TAB = 26
M.CLEAR_IMAGE = 'ui/bai/label/clear.png'   -- a clear pixel
M.OVER = 50         -- an overlay's priority over the card's (drawn over it)
M.UNIT_IMAGE = 'ui/bai/units/%s.png'
M.GAP = 26           -- px between the last own card and the first enemy card (the separator in the middle)
M.MARGIN = 510       -- the game's cards strip: screen width less this (hud_battle review_DY fullwidth_margin)
M.FRAME_H = 48       -- px over a row of cards kept free for the group frames' tabs (the game's tab: 45 px over a card)
M.GROUPS = 6
M.PARENTS = 8        -- how far up from a clicked component our card is looked for

-- The places of n enemy cards (pure: tests): own = {{x, y, w, h}} the own cards, right / left the strip's edges.
-- -> places {{x, y}}, separator {x, y, w, h, kind}, the card's w, h. Above the own cards a band of FRAME_H stays free
-- (the own groups' frames), and above the enemy rows another (theirs).
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
    local shift = h + M.FRAME_H
    for k = 1, n do
        local r = math.floor((k - 1) / per)
        local in_row = math.min(per, n - r * per)
        local cx = (left + right) / 2 - in_row * step / 2
        places[k] = {cx + ((k - 1) % per) * step, top - M.FRAME_H - shift * (rows - r) + M.FRAME_H}
    end
    for k = 1, n do places[k][2] = places[k][2] - M.FRAME_H end
    local width = math.min(n, per) * step
    return places, {(left + right) / 2 - width / 2, top - M.FRAME_H - 6, width, 4, 'above'}, w, h
end

-- opts: bm, env {core, UIComponent, find_uicomponent}, own = {{name, unit}}, enemy = {{name, unit, key}}, emit, now_ms,
-- highlight(names, on) (optional)
function M.start(opts)
    local env = opts.env
    local f = function(...) return env.find_uicomponent(...) end
    local root = function() return env.core:get_ui_root() end
    local h = {cards = {}, pick = {}, groups = {}, group_of = {}, frames = {}}
    local by_card = {}
    local placed, w0 = nil, {}
    local geom = {}          -- the cards' places now (the frames are drawn over them)

    local function box(uic)
        local okp, x, y = pcall(function() return uic:Position() end)
        local okb, w, hh = pcall(function() return uic:Bounds() end)
        return {x = okp and x or 0, y = okp and y or 0, w = okb and w or 0, h = okb and hh or 0}
    end

    -- a component's size and its state's images' too (a resized component keeps its images' own size otherwise:
    -- the check battle 07.11 17:38 - frames 54x130 by Bounds, drawn as a small mark in the corner)
    local function size(c, w, hh)
        if not c then return end
        w, hh = math.floor(w + 0.5), math.floor(hh + 0.5)
        pcall(function() c:SetCanResizeWidth(true); c:SetCanResizeHeight(true) end)
        pcall(function() c:Resize(w, hh) end)
        local okn, n = pcall(function() return c:NumImages() end)
        for i = 0, (okn and type(n) == 'number' and n or 1) - 1 do
            pcall(function()
                c:SetCanResizeCurrentStateImageWidth(i, true)
                c:SetCanResizeCurrentStateImageHeight(i, true)
                c:ResizeCurrentStateImage(i, w, hh)
            end)
        end
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

    local function picked_names()
        local out = {}
        for _, e in ipairs(h.cards) do if h.pick[e.name] then out[#out + 1] = e.name end end
        return out
    end

    local function show_button()
        if not h.button then return end
        local any = next(h.pick) ~= nil
        pcall(function() h.button:SetState(any and 'active' or 'inactive') end)
        pcall(function() h.button:SetDisabled(not any) end)
    end

    -- the group frames over the enemy cards: a frame over the group's cards and its number on a tab above them (our
    -- images: the game's own group_frame layout repositions itself by its callback - a white block over the own tabs,
    -- run 20261107-1425xx)
    -- every card of group k gets the group's colour frame (a child of the card, as the pick's), the group's first card
    -- its number on a tab above it (on the root: a child would be cut by its card); a click on the tab picks the group
    local function draw_frames()
        for _, fr in pairs(h.frames) do pcall(function() fr.tab:SetVisible(false) end) end
        for _, e in ipairs(h.cards) do
            if e.gf_uic and e.gf_k ~= h.group_of[e.name] then pcall(function() e.gf_uic:SetVisible(false) end) end
        end
        for k = 1, M.GROUPS do
            local names = h.groups[k]
            if names and #names > 0 and h.overlay then
                local x0, y0, first = nil, nil, nil
                for _, n in ipairs(names) do
                    local g, e = geom[n], h.by_name[n]
                    if g and e then
                        if not (x0 and (g.y > y0 or (g.y == y0 and g.x >= x0))) then x0, y0, first = g.x, g.y, e end
                        if e.gf_uic and e.gf_k ~= k then
                            pcall(function() e.gf_uic:DestroyChildren() end)
                            pcall(function() e.gf_uic:Destroy() end)
                            e.gf_uic = nil
                        end
                        if not e.gf_uic then
                            local ok, c = pcall(h.overlay, 'bai_enemy_gf' .. k .. '_' .. e.id:match('(%d+)$'),
                                string.format(M.GROUP_FRAME_IMAGE, k), e.uic)
                            if ok then e.gf_uic, e.gf_k = c, k end
                            if not ok then opts.emit('enemy_group_frame', {group = k, unit = n, error = tostring(c)}) end
                        end
                        if e.gf_uic then
                            h.put(e.gf_uic, g)
                            pcall(function() e.gf_uic:SetVisible(true) end)
                        end
                    end
                end
                if x0 then
                    local fr = h.frames[k]
                    if fr and fr.parent ~= first then
                        pcall(function() fr.tab:DestroyChildren() end)
                        pcall(function() fr.tab:Destroy() end)
                        fr, h.frames[k] = nil, nil
                    end
                    if not fr then
                        local ok, tab = pcall(h.overlay, 'bai_enemy_group_tab_' .. k, string.format(M.GROUP_TAB_IMAGE, k),
                            first.uic)
                        if ok then
                            fr = {tab = tab, parent = first}
                            h.frames[k] = fr
                            pcall(function() tab:SetInteractive(true) end)
                            pcall(function() h.find(tab, 'icon'):SetInteractive(true) end)
                        else
                            opts.emit('enemy_group_frame', {group = k, error = tostring(tab)})
                        end
                    end
                    if fr then
                        h.put(fr.tab, {x = x0, y = y0 - M.TAB - 2, w = M.TAB, h = M.TAB})
                        pcall(function() fr.tab:SetVisible(true) end)
                        if h.where then opts.emit('enemy_group_tab', {group = k, at = h.where(fr.tab)}) end
                    end
                end
            end
        end
    end

    function h.layout(force)
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
        if key == placed and not force then return nil end
        placed = key
        for k, e in ipairs(h.cards) do
            local b = box(e.uic)
            if b.w > 0 and math.abs(b.w - w) > 1 then
                pcall(function() e.uic:SetCanResizeWidth(true); e.uic:SetCanResizeHeight(true) end)
                pcall(function() e.uic:Resize(w, hh) end)
            end
            pcall(function() e.uic:MoveTo(places[k][1], places[k][2]) end)
            geom[e.name] = {x = places[k][1], y = places[k][2], w = w, h = hh}
            if e.pick_uic and h.put then h.put(e.pick_uic, geom[e.name]) end
        end
        if h.sep then
            local icon = f(h.sep, 'icon')
            size(icon, sep[3], sep[4])
            pcall(function() h.sep:MoveTo(sep[1], sep[2]) end)
            pcall(function() icon:MoveTo(sep[1], sep[2]) end)
        end
        if h.button then
            local last = places[#places]
            pcall(function() h.button:MoveTo(last[1] + w + 8, last[2] + hh / 2 - 18) end)
        end
        draw_frames()
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
            if ww[1] > 0 then size(bar, math.max(1, ww[1] * hp), ww[2]) end
        end
        local gone = type(men) == 'number' and men <= 0
        pcall(function() e.uic:SetOpacity(gone and 90 or 255, true) end)
    end

    -- an image over a card (ours: it takes no click - the click goes to the card), on the root or a child of the card
    -- (parent). Over a card it must be the card's child and NOT topmost (a topmost one, on the root or a child, went
    -- under the card: the check battles 07.11 17:44, 17:47 - only the frame's bottom edge seen); the priority over the
    -- card's is kept but did not help by itself
    local function priority(x)
        local ok, v = pcall(function() return x:Priority() end)
        return ok and type(v) == 'number' and v or nil
    end

    local function overlay(name, image, parent)
        local c = env.UIComponent((parent or root()):CreateComponent(name, M.PING_LAYOUT))
        local icon = f(c, 'icon')
        local arrow = f(c, 'arrow')
        if arrow then pcall(function() arrow:SetVisible(false) end) end
        pcall(function() icon:SetCanResizeWidth(true); icon:SetCanResizeHeight(true) end)
        -- the indicator's icon has 3 images (an oval's base, its yellow fill, the picture: stretched with ours they covered
        -- the card's picture - the check battles 07.11 17:52, 17:57): ours as image 0, the others clear
        local okn, n = pcall(function() return c:NumImages() end)
        local oki, ni = pcall(function() return icon:NumImages() end)
        icon:SetImagePath(image, 0)
        for i = 1, (oki and type(ni) == 'number' and ni or 1) - 1 do
            pcall(function() icon:SetImagePath(M.CLEAR_IMAGE, i) end)
        end
        if not h.told_images then
            h.told_images = true
            opts.emit('enemy_overlay_images', {indicator = okn and n or tostring(n), icon = oki and ni or tostring(ni)})
        end
        for _, x in ipairs({c, icon}) do pcall(function() x:SetInteractive(false) end) end
        pcall(function() c:SetVisible(false) end)
        -- on the root it must be topmost (the cards are); a card's child must not: a topmost child went under its card
        -- (the check battles 07.11 17:44 and 17:47 - only the frame's bottom edge seen)
        if not parent then pcall(function() c:RegisterTopMost() end) end
        local card = h.cards[1] and h.cards[1].uic
        local base = 0
        for _, x in ipairs({card, card and f(card, 'card_image_holder'), parent}) do
            base = math.max(base, x and priority(x) or 0)
        end
        local okp = pcall(function() c:PropagatePriority(base + M.OVER) end)
        if not h.told_priority then
            h.told_priority = true
            opts.emit('enemy_priority', {card = card and priority(card), picture = card and priority(f(card, 'card_image_holder')),
                overlay = priority(c), icon = priority(icon), set = okp})
        end
        return c
    end

    local function put(c, g)
        if not (c and g) then return end
        local icon = f(c, 'icon')
        size(icon, g.w, g.h)
        pcall(function() c:MoveTo(g.x, g.y) end)
        pcall(function() icon:MoveTo(g.x, g.y) end)
    end

    -- where an overlay really is (the check: placed, sized, visible)
    local function where(c)
        if not c then return nil end
        local icon = f(c, 'icon')
        local out = {}
        for k, x in pairs({c = c, icon = icon}) do
            local b = box(x)
            local vis = select(2, pcall(function() return x:Visible() end))
            out[k] = {b.x, b.y, b.w, b.h, tostring(vis)}
        end
        return out
    end

    h.put = put
    h.overlay = overlay
    h.find = f
    h.where = where

    -- a card's own overlay (a child of the card): the pick's frame, the hover's
    local function card_overlay(e, field, kind, image)
        if not e[field] then
            local ok, c = pcall(overlay, 'bai_enemy_' .. kind .. '_' .. e.id:match('(%d+)$'), image, e.uic)
            if ok then e[field] = c end
            opts.emit('enemy_overlay', {unit = e.name, kind = kind, ok = ok, error = (not ok) and tostring(c) or nil})
        end
        return e[field]
    end

    local function show_pick(e)
        local on = h.pick[e.name] == true
        if on then card_overlay(e, 'pick_uic', 'pick', M.PICK_IMAGE) end
        if e.pick_uic then
            put(e.pick_uic, geom[e.name])
            pcall(function() e.pick_uic:SetVisible(on) end)
            if on then opts.emit('enemy_pick_frame', {unit = e.name, at = where(e.pick_uic)}) end
        end
    end

    function h.hover(name, on)
        local e = h.by_name[name]
        if not e then return end
        local c = card_overlay(e, 'hover_uic', 'hover', M.HOVER_IMAGE)
        if not c then return end
        if on and geom[name] then put(c, geom[name]) end
        pcall(function() c:SetVisible(on) end)
        if opts.highlight and not h.pick[name] then pcall(opts.highlight, {name}, on) end
    end

    local function highlight(names, on)
        if opts.highlight and #names > 0 then pcall(opts.highlight, names, on) end
    end

    function h.toggle(name, why)
        local e = h.by_name[name]
        if not e then return end
        h.pick[name] = not h.pick[name] or nil
        show_pick(e)
        highlight({name}, h.pick[name] == true)
        show_button()
        opts.emit('enemy_pick', {t = opts.now_ms(), unit = name, on = h.pick[name] == true, why = why})
    end

    function h.clear_pick()
        local names = picked_names()
        for _, name in ipairs(names) do
            h.pick[name] = nil
            show_pick(h.by_name[name])
        end
        highlight(names, false)
        show_button()
    end

    -- a group's frame clicked: the pick is that group
    function h.pick_group(k, why)
        h.clear_pick()
        for _, name in ipairs(h.groups[k] or {}) do
            h.pick[name] = true
            show_pick(h.by_name[name])
        end
        highlight(h.groups[k] or {}, true)
        show_button()
        opts.emit('enemy_pick_group', {t = opts.now_ms(), group = k, why = why})
    end

    -- the pick -> enemy group N (the next number not in use; 6 at most: then the last)
    function h.group_pick(why)
        local names = picked_names()
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
        draw_frames()
        opts.emit('enemy_group', {t = opts.now_ms(), why = why, group = k, units = names})
        return k
    end

    function h.tick()
        for _, e in ipairs(h.cards) do pcall(fill, e) end
        local ok, l = pcall(h.layout)
        if not ok or l then opts.emit('enemy_cards_layout', {t = opts.now_ms(), result = tostring(l)}) end
    end

    -- which of ours a clicked component is: itself or one of its parents (module doc)
    local function ours(component, s)
        if by_card[s] or s == 'bai_enemy_sep' or s == 'bai_enemy_group_button' or s:match('^bai_enemy_group_') then
            return s
        end
        if not component then return nil end
        local ok, uic = pcall(function() return env.UIComponent(component) end)
        for _ = 1, M.PARENTS do
            if not ok or not uic then return nil end
            local okp, parent = pcall(function() return env.UIComponent(uic:Parent()) end)
            if not okp or not parent then return nil end
            local oki, id = pcall(function() return parent:Id() end)
            if not oki or type(id) ~= 'string' then return nil end
            if by_card[id] or id:match('^bai_enemy_group_') then return id end
            if id == 'root' then return nil end
            uic = parent
        end
        return nil
    end

    -- the cards, the separator, our group button
    h.by_name = {}
    local made = {}
    local oks, serr = pcall(function()
        h.sep = env.UIComponent(root():CreateComponent('bai_enemy_sep', M.PING_LAYOUT))
        local icon = f(h.sep, 'icon')
        local arrow = f(h.sep, 'arrow')
        if arrow then pcall(function() arrow:SetVisible(false) end) end
        icon:SetImagePath(M.SEP_IMAGE)
        pcall(function() icon:SetCanResizeWidth(true); icon:SetCanResizeHeight(true) end)
        pcall(function() h.sep:RegisterTopMost() end)
    end)
    made.separator = oks and 'ok' or tostring(serr)
    for i, it in ipairs(opts.enemy) do
        local ok, err = pcall(function()
            local id = 'bai_enemy_card_' .. i
            local c = env.UIComponent(root():CreateComponent(id, M.CARD_LAYOUT))
            c:SetVisible(true)
            pcall(function() c:SetInteractive(true) end)
            pcall(function() c:RegisterTopMost() end)
            local e = {name = it.name, unit = it.unit, key = it.key, uic = c, id = id}
            h.cards[#h.cards + 1] = e
            h.by_name[it.name] = e
            by_card[id] = e
        end)
        if not ok then made[it.name] = tostring(err) end
    end
    local okb, berr = pcall(function()
        h.button = env.UIComponent(root():CreateComponent('bai_enemy_group_button', M.BUTTON_LAYOUT))
        pcall(function() h.button:SetTooltipText('Вражеская группа из выбранных карточек (как Ctrl+G)', true) end)
        pcall(function() h.button:RegisterTopMost() end)
    end)
    made.button = okb and 'ok' or tostring(berr)
    show_button()
    h.tick()
    opts.emit('enemy_cards', {t = opts.now_ms(), cards = #h.cards, made = made})

    -- the clicks and the group shortcut
    pcall(function()
        env.core:add_listener('bai_enemy_cards_click', 'ComponentLClickUp', true, function(c)
            local s = tostring(c.string)
            local id = ours(c.component, s)
            if id then
                opts.emit('enemy_click', {t = opts.now_ms(), string = s, ours = id})
            end
            if id and by_card[id] then
                h.toggle(by_card[id].name, 'click ' .. s)
            elseif id == 'bai_enemy_sep' then
                h.clear_pick()
            elseif id == 'bai_enemy_group_button' then
                h.group_pick('our button')
            elseif id and (id:match('^bai_enemy_group_frame_') or id:match('^bai_enemy_group_tab_')) then
                h.pick_group(tonumber(id:match('(%d+)$')), 'frame')
            elseif s == 'button_group' then
                h.group_pick('button_group')
            end
        end, true)
        for _, ev in ipairs({'ComponentMouseOn', 'ComponentMouseOff'}) do
            env.core:add_listener('bai_enemy_cards_' .. ev, ev, true, function(c)
                local id = ours(c.component, tostring(c.string))
                if id and by_card[id] then h.hover(by_card[id].name, ev == 'ComponentMouseOn') end
            end, true)
        end
        env.core:add_listener('bai_enemy_cards_shortcut', 'ShortcutPressed', true, function(c)
            if tostring(c.string) == 'toggle_group' then h.group_pick('toggle_group') end
        end, true)
    end)
    -- a test's click: SimulateLClick on the card or on one of its children (a person's click lands on a child)
    function h.click(name, child)
        local e = h.by_name[name]
        if not e then return 'unknown unit' end
        local target = child and f(e.uic, child) or e.uic
        if not target then return 'no ' .. tostring(child) end
        local ok, err = pcall(function() target:SimulateLClick() end)
        return ok and 'ok' or tostring(err)
    end
    return h
end

return M

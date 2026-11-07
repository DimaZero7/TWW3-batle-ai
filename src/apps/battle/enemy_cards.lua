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
M.GROUP_ICON = 'ui/skins/default/icon_group.png'   -- the game's own group icon
M.PING_LAYOUT = 'ui/battle ui/unit_ping_indicator.twui.xml'
M.SEP_IMAGE = 'ui/bai/label/sep.png'
M.PICK_IMAGE = 'ui/bai/label/pick.png'     -- the pick's frame over a card (the card's own 'selected' look is the game's,
M.HOVER_IMAGE = 'ui/bai/label/hover.png'   -- bound to its unit: it does not show on ours - the person's battle 07.11)
M.GROUP_FRAME_IMAGE = 'ui/bai/label/gf%d.png'   -- a group's frame in its colour (tools/label/icons.py)
M.GROUP_TAB_IMAGE = 'ui/bai/label/gn%d.png'     -- its number (gn1..gn40, tools/label/icons.py TABS)
M.TAB = 26
M.EDGE = 3           -- px: a group frame's edge
M.GROUP_EDGE_IMAGE = 'ui/bai/label/gb%d.png'   -- its colour, solid (tools/label/icons.py)
M.CLEAR_IMAGE = 'ui/bai/label/clear.png'   -- a clear pixel
M.OVER = 50         -- an overlay's priority over the card's (drawn over it)
M.UNIT_IMAGE = 'ui/bai/units/%s.png'
M.GAP = 26           -- px between the last own card and the first enemy card (the separator in the middle)
M.MARGIN = 510       -- the game's cards strip: screen width less this (hud_battle review_DY fullwidth_margin)
M.FRAME_H = 48       -- px over a row of cards kept free for the group frames' tabs (the game's tab: 45 px over a card)
M.COLORS = 6         -- the group colours (config/label.json group_colors): group k's the ((k - 1) % 6 + 1)-th - neighbours
                     -- never alike; as many groups as the enemy has units (the person's 07.11: a 7th merged into the 6th)
M.TABS = 40          -- number tabs drawn (a group above: no tab, its frame still)
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

-- The cards' order (pure: tests): names in the cards' first order, groups = {[k] = {names}} -> the names in order:
-- group 1's cards, group 2's, ... (each in the first order), then the units in no group in the first order. As the game
-- orders its own cards: a unit grouped moves up to its group, before the units in none (the check battle 07.11 17:29:
-- own greatsword 2 grouped as 2 moved from the 3rd card to the 2nd, before the lord in none).
-- the group numbers in use, in order (pure)
function M.numbers(groups)
    local ks = {}
    for k, v in pairs(groups) do if type(k) == 'number' and v and #v > 0 then ks[#ks + 1] = k end end
    table.sort(ks)
    return ks
end

-- a group's colour number 1..M.COLORS (pure)
function M.colour(k)
    return (k - 1) % M.COLORS + 1
end

function M.order(names, groups)
    local out, used = {}, {}
    for _, k in ipairs(M.numbers(groups)) do
        local set = {}
        for _, n in ipairs(groups[k] or {}) do set[n] = true end
        for _, n in ipairs(names) do
            if set[n] and not used[n] then out[#out + 1] = n; used[n] = true end
        end
    end
    for _, n in ipairs(names) do if not used[n] then out[#out + 1] = n end end
    return out
end

-- A click's pick, as the files in Windows Explorer (pure: tests): pick {[name] = true}, order the cards' names as they
-- stand, mods {ctrl, shift} -> the new pick, the new anchor (the range's start for Shift):
--   a click - only this unit; Ctrl - this unit added or taken out, the rest kept; Shift - the cards from the anchor to this
--   one (with Ctrl: added to the pick); Shift with no anchor - as Ctrl.
function M.click(pick, order, name, mods, anchor)
    local out = {}
    local ia, ib = nil, nil
    for i, n in ipairs(order) do
        if n == anchor then ia = i end
        if n == name then ib = i end
    end
    if mods.shift and ia and ib then
        if mods.ctrl then for n in pairs(pick) do out[n] = true end end
        for i = math.min(ia, ib), math.max(ia, ib) do out[order[i]] = true end
        return out, anchor
    end
    if mods.ctrl or mods.shift then
        for n in pairs(pick) do out[n] = true end
        out[name] = (not pick[name]) or nil
        return out, name
    end
    out[name] = true
    return out, name
end

-- A box's or a group's pick (pure): add (Ctrl / Shift held) - joined to the pick; else the pick is these.
function M.merge(pick, names, add)
    local out = {}
    if add then for n in pairs(pick) do out[n] = true end end
    for _, n in ipairs(names) do out[n] = true end
    return out
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

    -- A group's frame, as the game's own: one frame around the group's cards (they stand in a row: M.order) in its colour
    -- and its number on a tab over the first. Drawn as pieces on each card (its children: a card's child over another
    -- card would go under it) - the top and the bottom edge on every card of the group, the left on the first of a row
    -- of them, the right on the last; the tab a child of the first card. A click on the tab picks the group.
    local EDGES = {'top', 'bottom', 'left', 'right'}
    local function edge_box(g, side, first, last)
        local t = M.EDGE
        if side == 'top' then return {x = g.x, y = g.y, w = g.w, h = t} end
        if side == 'bottom' then return {x = g.x, y = g.y + g.h - t, w = g.w, h = t} end
        if side == 'left' and first then return {x = g.x, y = g.y, w = t, h = g.h} end
        if side == 'right' and last then return {x = g.x + g.w - t, y = g.y, w = t, h = g.h} end
        return nil
    end

    local function draw_frames()
        -- each card: its group's edges
        for _, e in ipairs(h.cards) do
            local k, g = h.group_of[e.name], geom[e.name]
            e.edges = e.edges or {}
            local first, last = false, false
            if k and g then
                local l, r = nil, nil
                for _, n in ipairs(h.groups[k] or {}) do
                    local o = geom[n]
                    if o and o.y == g.y then
                        if math.abs(o.x + o.w - g.x) < 2 then l = n end
                        if math.abs(g.x + g.w - o.x) < 2 then r = n end
                    end
                end
                first, last = l == nil, r == nil
            end
            for _, side in ipairs(EDGES) do
                local b = k and g and edge_box(g, side, first, last) or nil
                local c = e.edges[side]
                if b and not c then
                    local ok, made = pcall(h.overlay, 'bai_enemy_edge_' .. side .. '_' .. e.id:match('(%d+)$'),
                        string.format(M.GROUP_EDGE_IMAGE, M.colour(k)), e.uic)
                    if ok then c = made; e.edges[side] = c; e.edge_k = k end
                    if not ok then opts.emit('enemy_group_frame', {group = k, unit = e.name, error = tostring(made)}) end
                end
                if c then
                    if b then
                        if e.edge_k ~= k then
                            pcall(function() h.find(c, 'icon'):SetImagePath(string.format(M.GROUP_EDGE_IMAGE, M.colour(k)), 0) end)
                        end
                        h.put(c, b)
                    end
                    pcall(function() c:SetVisible(b ~= nil) end)
                end
            end
            if k then e.edge_k = k end
        end
        -- each group: the tab over its first card (a child of that card: remade when the first card changes)
        local ks, seen = {}, {}
        for _, k in ipairs(M.numbers(h.groups)) do ks[#ks + 1] = k; seen[k] = true end
        for k in pairs(h.frames) do if not seen[k] then ks[#ks + 1] = k end end
        for _, k in ipairs(ks) do
            local names = h.groups[k]
            local fr = h.frames[k]
            local first = nil
            for _, n in ipairs(names or {}) do
                local g = geom[n]
                if g and (not first or g.y < geom[first].y or (g.y == geom[first].y and g.x < geom[first].x)) then
                    first = n
                end
            end
            local e = first and h.by_name[first]
            if fr and fr.parent ~= e then
                pcall(function() fr.tab:SetVisible(false) end)
                pcall(function() fr.tab:DestroyChildren() end)
                pcall(function() fr.tab:Destroy() end)
                fr, h.frames[k] = nil, nil
            end
            if e and h.overlay and k <= M.TABS then
                if not fr then
                    local ok, tab = pcall(h.overlay, 'bai_enemy_group_tab_' .. k, string.format(M.GROUP_TAB_IMAGE, k), e.uic)
                    if ok then
                        fr = {tab = tab, parent = e}
                        h.frames[k] = fr
                        pcall(function() tab:SetInteractive(true) end)
                        pcall(function() h.find(tab, 'icon'):SetInteractive(true) end)
                    else
                        opts.emit('enemy_group_frame', {group = k, error = tostring(tab)})
                    end
                end
                if fr then
                    local g = geom[first]
                    h.put(fr.tab, {x = g.x, y = g.y - M.TAB - 2, w = M.TAB, h = M.TAB})
                    pcall(function() fr.tab:SetVisible(true) end)
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
        local names = {}
        for _, e in ipairs(h.cards) do names[#names + 1] = e.name end
        local order = M.order(names, h.groups)
        h.order = order
        local key = table.concat({w, hh, sep[1], sep[2], places[1][1], places[1][2], #h.cards,
            table.concat(order, ' ')}, ',')
        if key == placed and not force then return nil end
        placed = key
        for k, name in ipairs(order) do
            local e = h.by_name[name]
            local b = box(e.uic)
            if b.w > 0 and math.abs(b.w - w) > 1 then
                pcall(function() e.uic:SetCanResizeWidth(true); e.uic:SetCanResizeHeight(true) end)
                pcall(function() e.uic:Resize(w, hh) end)
            end
            pcall(function() e.uic:MoveTo(places[k][1], places[k][2]) end)
            geom[e.name] = {x = places[k][1], y = places[k][2], w = w, h = hh}
            if e.pick_uic and h.put then h.put(e.pick_uic, geom[e.name]) end
            if e.hover_uic and h.put then h.put(e.hover_uic, geom[e.name]) end
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

    -- the highlight on the field follows the pick (and the card under the mouse): only the units whose state changes are
    -- touched (a unit picked again is not switched off and on - it blinked: the person's battle 07.11 18:16)
    -- One enemy picked: the game's own look of a selected enemy (opts.native(name): the white ring on its banner, the
    -- red outline under its men - the CCO Select; the own selection goes, as when the game's enemy is clicked). More than
    -- one: the ring over each (opts.highlight). The switch only when the native one changes (no blinking).
    h.lit = {}
    local function sync()
        local picked = {}
        for _, e in ipairs(h.cards) do if h.pick[e.name] then picked[#picked + 1] = e.name end end
        local native = (opts.native and #picked == 1) and picked[1] or nil
        if native ~= h.native then
            local ok = pcall(opts.native, native, h.native)
            h.native = ok and native or nil
        end
        if not opts.highlight then return end
        local on, off = {}, {}
        for _, e in ipairs(h.cards) do
            local want = (h.pick[e.name] == true or h.hovered == e.name) and e.name ~= h.native
            if want and not h.lit[e.name] then on[#on + 1] = e.name end
            if not want and h.lit[e.name] then off[#off + 1] = e.name end
            h.lit[e.name] = want or nil
        end
        if #off > 0 then pcall(opts.highlight, off, false) end
        if #on > 0 then pcall(opts.highlight, on, true) end
    end
    h.sync = sync

    function h.hover(name, on)
        local e = h.by_name[name]
        if not e then return end
        local c = card_overlay(e, 'hover_uic', 'hover', M.HOVER_IMAGE)
        if not c then return end
        if on and geom[name] then put(c, geom[name]) end
        pcall(function() c:SetVisible(on) end)
        if on then h.hovered = name elseif h.hovered == name then h.hovered = nil end
        sync()
    end

    local function unpick_all()
        for _, name in ipairs(picked_names()) do
            h.pick[name] = nil
            show_pick(h.by_name[name])
        end
    end

    -- the modifier keys now (opts.keys: the host's helper, tools/label/keys.ps1; none known - none held)
    local function keys()
        local ok, k = pcall(function() return opts.keys and opts.keys() end)
        return ok and type(k) == 'table' and k or {}
    end
    h.keys = keys

    -- the pick becomes new: only the cards whose state changes are redrawn
    local function apply(new)
        for _, e in ipairs(h.cards) do
            local want = new[e.name] == true
            if want ~= (h.pick[e.name] == true) then
                h.pick[e.name] = want or nil
                show_pick(e)
            end
        end
        sync()
        show_button()
    end

    -- a click on an enemy (its card, or its unit on the field: field - Shift as Ctrl), as Explorer (M.click)
    function h.click_unit(name, why, field)
        if not h.by_name[name] then return end
        local k = keys()
        local mods = {ctrl = k.ctrl == true, shift = k.shift == true and not field}
        if field and k.shift then mods.ctrl = true end
        local new, anchor = M.click(h.pick, h.order or {}, name, mods, h.anchor)
        h.anchor = anchor
        apply(new)
        opts.emit('enemy_pick', {t = opts.now_ms(), unit = name, on = h.pick[name] == true, why = why,
            ctrl = mods.ctrl or nil, shift = mods.shift or nil, picked = #picked_names(), keys = k.text})
    end

    function h.toggle(name, why)
        local e = h.by_name[name]
        if not e then return end
        local new = {}
        for n in pairs(h.pick) do new[n] = true end
        new[name] = (not h.pick[name]) or nil
        h.anchor = name
        apply(new)
        opts.emit('enemy_pick', {t = opts.now_ms(), unit = name, on = h.pick[name] == true, why = why})
    end

    function h.clear_pick(why)
        if why then
            -- a click on the field or an own unit: with Ctrl / Shift held the own selection grows - the pick stays
            local k = keys()
            if k.ctrl or k.shift then return end
        end
        local had = #picked_names()
        unpick_all()
        sync()
        show_button()
        if why and had > 0 then opts.emit('enemy_pick_clear', {t = opts.now_ms(), why = why, units = had}) end
    end

    -- a box over the field (apps.battle.box_pick): the pick is the box's enemies (none: the pick goes, as the own
    -- selection); with Ctrl / Shift held they are added
    function h.box_pick(names, why)
        local k = keys()
        local add = k.ctrl == true or k.shift == true
        apply(M.merge(h.pick, names, add))
        opts.emit('enemy_pick_box', {t = opts.now_ms(), units = names, why = why, add = add or nil,
            picked = #picked_names()})
    end

    -- a group's tab clicked: the pick is that group (with Ctrl / Shift: added)
    function h.pick_group(k, why)
        local m = keys()
        apply(M.merge(h.pick, h.groups[k] or {}, m.ctrl == true or m.shift == true))
        opts.emit('enemy_pick_group', {t = opts.now_ms(), group = k, why = why})
    end

    -- the pick -> enemy group N (the next number not in use; 6 at most: then the last); the pick being exactly one group
    -- -> that group dissolved (the game's group button on a group: 'ungroup'). The cards move to their order (M.order).
    function h.group_pick(why)
        local names = picked_names()
        if #names == 0 then return nil end
        local same = h.group_of[names[1]]
        if same and #(h.groups[same] or {}) == #names then
            for _, n in ipairs(names) do if h.group_of[n] ~= same then same = nil break end end
        else
            same = nil
        end
        if same then
            for _, n in ipairs(names) do h.group_of[n] = nil end
            h.groups[same] = nil
            h.clear_pick()
            pcall(h.layout, true)
            opts.emit('enemy_ungroup', {t = opts.now_ms(), why = why, group = same, units = names})
            return 0
        end
        local k = 1
        while h.groups[k] and #h.groups[k] > 0 and k < #h.cards do k = k + 1 end
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
        pcall(h.layout, true)
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
        pcall(function() h.button:SetImagePath(M.GROUP_ICON, 0) end)
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
                h.click_unit(by_card[id].name, 'click ' .. s)
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

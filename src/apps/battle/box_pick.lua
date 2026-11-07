-- The field's mouse for the enemy, as the game does it for the own units (the live labelling, build/diag_chain/
-- live_label.md). Found by name only (find_uicomponent): a ChildCount / Find walk breaks the game's script bindings.
--   The box: the game's box selection takes only the own units; its rectangle is a UI component of the game
--   (ui/battle ui/box_selection.twui.xml, ui3.pack: 'box_selection', in the tree while idle, hidden), watched every
--   M.POLL_MS of real time (the component kept: a frame's poll is Visible and Bounds only). While it shows its rectangle
--   is kept; when it goes, the enemies whose formation on the screen overlaps it are the box's -> opts.on_box(names).
--   The formation on the screen: apps.battle.screen_proj - the unit's ground rectangle (position, bearing, ordered width,
--   a depth of its men) projected with the camera; the projection's field of view and the banners' height are fitted to
--   the unit banners the game draws (unit_id_holder's component of each unit: a known unit, where it shows). Before a
--   fit (or a bad one) a unit's banner point stands for it.
--   A click on the field, not a box (the game: the own selection replaced or emptied) -> opts.on_clear(why): the own
--   selection changed with no box just ended, or the field clicked (the click's component 'root'). A click on an enemy's
--   banner (the component named by its unit's id) -> opts.on_unit(name), as a click on its card.
-- Shift / Ctrl are not seen by a script: a box replaces the pick (an empty one keeps it).
local proj = require('apps.battle.screen_proj')

local M = {}

M.NAMES = {'box_selection'}  -- the rectangle's component (the twui's name)
M.HOLDER = 'unit_id_holder'
M.POLL_MS = 16               -- every frame or so (a short drag lasts a few hundred ms)
M.SELECTION_MS = 100
M.MIN = 4                    -- px: a smaller rectangle is a click, not a box
M.AFTER_BOX_MS = 500         -- a selection change or a click this soon after a box is the box's own
M.FIT_EVERY_MS = 3000
M.FIT_GOOD_PX = 12           -- a fit this good (median px) is kept for longer
M.FIT_USE_PX = 0             -- the formations by the fit are not used yet: the fit matches the banners (2 px) but its
                             -- ground stands some 50 px over the men (the banner's height and its screen offset trade
                             -- off - the check battle 07.11 18:46, build/diag_chain/polys_54.png); the banner's region
                             -- below stands for the unit (M.REGION)
-- a unit's region on the screen from its banner component's centre (px of the game's screen): the icon sits ~45 px under
-- the centre, the men ~90-170 px under it (the check battle 07.11 18:46, polys_54.png)
M.REGION = {left = -40, right = 40, top = 15, bottom = 170}
M.FIELD = {root = true}      -- the click's component when the field itself is clicked
M.MAN_AREA = 1.6             -- m2 a man takes in a formation (a depth of the men for the ordered width)
M.BODY = 2.0                 -- m: the men's height (their box on the screen)

-- The names whose point stands inside rect (pure: tests): points = {{name, x, y}}, rect {x, y, w, h}.
function M.inside(rect, points)
    local out = {}
    for _, p in ipairs(points) do
        if p[2] >= rect.x and p[2] <= rect.x + rect.w and p[3] >= rect.y and p[3] <= rect.y + rect.h then
            out[#out + 1] = p[1]
        end
    end
    return out
end

-- The names whose region (M.REGION round its banner point) the rectangle overlaps (pure: tests): points {{name, x, y}}.
function M.covered_by_region(rect, points, reg)
    reg = reg or M.REGION
    local out = {}
    for _, p in ipairs(points) do
        local x0, x1, y0, y1 = p[2] + reg.left, p[2] + reg.right, p[3] + reg.top, p[3] + reg.bottom
        if rect.x <= x1 and x0 <= rect.x + rect.w and rect.y <= y1 and y0 <= rect.y + rect.h then out[#out + 1] = p[1] end
    end
    return out
end

-- a formation's depth (pure): men, ordered width -> m
function M.depth(men, width)
    if not (men and width and width > 0) then return 6 end
    return math.max(3, math.min(60, men * M.MAN_AREA / width))
end

-- opts: bm, env {core, common, find_uicomponent}, units = {{name, unit}} (the enemy), all = {{name, unit}} (every unit:
-- the fit), on_box(names, rect), on_clear(why), emit, now_ms, real_ms (a real-time clock in ms)
function M.start(opts)
    local env = opts.env
    local f = function(...) return env.find_uicomponent(...) end
    local root = function() return env.core:get_ui_root() end
    local h = {seen = false, last = nil, boxes = 0, box_ms = -1e9, fit = nil, fit_ms = -1e9, polls = 0}
    local real = opts.real_ms or function() return os.clock() * 1000 end
    h.by_uid = {}
    for _, it in ipairs(opts.units) do
        pcall(function() h.by_uid[tostring(it.unit:unique_ui_id())] = it.name end)
    end

    local function box(uic)
        local okp, x, y = pcall(function() return uic:Position() end)
        local okb, w, hh = pcall(function() return uic:Bounds() end)
        return {x = okp and x or 0, y = okp and y or 0, w = okb and w or 0, h = okb and hh or 0}
    end

    local function visible(uic)
        local ok, v = pcall(function() return uic:Visible() end)
        return ok and v ~= false
    end

    local function rectangle()
        if h.rect_uic then
            local ok = pcall(function() return h.rect_uic:Visible() end)
            if ok then return h.rect_uic, h.rect_name end
            h.rect_uic = nil
        end
        for _, name in ipairs(M.NAMES) do
            local c = f(root(), name)
            if c then h.rect_uic, h.rect_name = c, name; return c, name end
        end
        return nil
    end

    local function banner(holder, it)
        local c = f(holder, tostring(it.unit:unique_ui_id()))
        if c and visible(c) then
            local b = box(c)
            if b.w > 0 then return b.x + b.w / 2, b.y + b.h / 2 end
        end
        return nil
    end

    local function world(it)
        local ok, p = pcall(function() return it.unit:position() end)
        if not ok or not p then return nil end
        return {p:get_x(), p:get_y(), p:get_z()}
    end

    local function camera()
        local ok, c = pcall(function()
            local cam = opts.bm:camera()
            local p, t = cam:position(), cam:target()
            return {{p:get_x(), p:get_y(), p:get_z()}, {t:get_x(), t:get_y(), t:get_z()}}
        end)
        if ok and c then return c end
        return nil
    end

    local function screen()
        local ok, w, hh = pcall(function() return env.core:get_screen_resolution() end)
        if ok and w then return w, hh end
        return 1600, 1200
    end

    -- every unit's banner point on the screen now ({{name, x, y}}; the units the game hides have none)
    function h.points(units)
        local holder = f(root(), M.HOLDER)
        local out = {}
        if not holder then return out end
        for _, it in ipairs(units or opts.units) do
            local x, y = banner(holder, it)
            if x then out[#out + 1] = {it.name, x, y} end
        end
        return out
    end

    -- the projection's fit to the banners now (both screen handednesses tried)
    function h.calibrate(why)
        local cam = camera()
        local holder = f(root(), M.HOLDER)
        if not (cam and holder) then return nil end
        local w, hh = screen()
        local pairs = {}
        for _, it in ipairs(opts.all or opts.units) do
            local x, y = banner(holder, it)
            local p = x and world(it)
            if p then pairs[#pairs + 1] = {p[1], p[2], p[3], x, y} end
        end
        if #pairs < 6 then return nil end
        local best = nil
        for _, flip in ipairs({1, -1}) do
            local b = proj.basis(cam[1], cam[2])
            b.r = {b.r[1] * flip, b.r[2] * flip, b.r[3] * flip}
            local fov, hb, dx, dy, err = proj.fit(b, w, hh, pairs)
            if fov and (not best or err < best.err) then
                best = {fov = fov, hb = hb, dx = dx, dy = dy, err = err, flip = flip, n = #pairs}
            end
        end
        h.fit_ms = real()
        if best and (not h.fit or best.err <= h.fit.err * 1.5 or real() - (h.fit.at or 0) > 30000) then
            best.at = real()
            h.fit = best
        end
        opts.emit('box_fit', {t = opts.now_ms(), why = why, fov = best and best.fov, hb = best and best.hb,
            dx = best and best.dx, dy = best and best.dy, err = best and best.err, flip = best and best.flip, n = #pairs})
        return best
    end

    -- the camera's frame with the fit's handedness
    local function frame(fit)
        local cam = camera()
        if not (fit and cam) then return nil end
        local b = proj.basis(cam[1], cam[2])
        b.r = {b.r[1] * fit.flip, b.r[2] * fit.flip, b.r[3] * fit.flip}
        return b
    end

    -- a unit on the screen: its men's box (ground formation raised by M.BODY) and its centre
    local function unit_screen(b, fit, it)
        local p = world(it)
        if not p then return nil end
        local u = it.unit
        local bearing = select(2, pcall(function() return u:bearing() end))
        local width = select(2, pcall(function() return u:ordered_width() end))
        local men = select(2, pcall(function() return u:number_of_men_alive() end))
        width = type(width) == 'number' and math.max(width, 3) or 10
        local q = proj.quad(p[1], p[2], p[3], type(bearing) == 'number' and bearing or 0, width,
            M.depth(type(men) == 'number' and men or 0, width))
        local w, hh = screen()
        local box_ = proj.unit_box(b, fit.fov, w, hh, q, M.BODY)
        local cx, cy = proj.project(b, fit.fov, w, hh, p)
        return box_, cx and {cx, cy}, p
    end

    -- the enemies a screen rectangle covers: by their men's box on the screen (a fit) or their banner point
    function h.covered(rect)
        local fit = h.fit
        local b = fit and fit.err <= M.FIT_USE_PX and frame(fit)
        local holder = f(root(), M.HOLDER)
        local out, how = {}, 'region'
        if b and holder and M.FIT_USE_PX > 0 then
            how = 'formation'
            for _, it in ipairs(opts.units) do
                if banner(holder, it) then                       -- a unit the game hides is not taken
                    local ub = unit_screen(b, fit, it)
                    if ub and rect.x <= ub.x + ub.w and ub.x <= rect.x + rect.w and rect.y <= ub.y + ub.h
                        and ub.y <= rect.y + rect.h then
                        out[#out + 1] = it.name
                    end
                end
            end
        else
            out, how = M.covered_by_region(rect, h.points()), 'region'
        end
        return out, how
    end

    -- a unit on the screen now (the check: drawn over a screenshot): its men's box, its centre, its banner by the fit
    function h.polygon(name)
        local fit = h.fit
        local b = frame(fit)
        if not b then return nil end
        for _, it in ipairs(opts.all or opts.units) do
            if it.name == name then
                local ub, c, p = unit_screen(b, fit, it)
                local w, hh = screen()
                local bx, by = nil, nil
                if p then bx, by = proj.project(b, fit.fov, w, hh, {p[1], p[2] + fit.hb, p[3]}) end
                return {box = ub and {math.floor(ub.x), math.floor(ub.y), math.floor(ub.w), math.floor(ub.h)},
                    centre = c and {math.floor(c[1]), math.floor(c[2])},
                    banner_fit = bx and {math.floor(bx + fit.dx), math.floor(by + fit.dy)}}
            end
        end
        return nil
    end

    function h.finish(rect, why)
        local names, how = h.covered(rect)
        h.boxes = h.boxes + 1
        h.box_ms = real()
        opts.emit('box_pick', {t = opts.now_ms(), why = why, rect = {rect.x, rect.y, rect.w, rect.h}, units = names,
            how = how, fit_err = h.fit and h.fit.err})
        pcall(opts.on_box, names, rect)
        return names
    end

    local function selection()
        local ok, v = pcall(function()
            return env.common.get_context_value('CcoBattleRoot', '', 'SelectionContext.UnitList.JoinString(UniqueUiId, ",")')
        end)
        return ok and tostring(v) or nil
    end

    local function poll()
        if h.stopped then return end
        h.polls = h.polls + 1
        local c = rectangle()
        if c and visible(c) then
            local b = box(c)
            if b.w >= M.MIN and b.h >= M.MIN then
                h.last = b
                if not h.seen then
                    h.seen = true
                    opts.emit('box_seen', {t = opts.now_ms(), rect = {b.x, b.y, b.w, b.h}})
                end
            end
        elseif h.seen then
            h.seen = false
            -- the hidden component's rectangle: the drag's last one when it grew on from the last seen
            local r, b = h.last, c and box(c)
            if b and b.w >= r.w and b.h >= r.h and b.x <= r.x + 1 and b.y <= r.y + 1
                and b.x + b.w >= r.x + r.w - 1 and b.y + b.h >= r.y + r.h - 1 then r = b end
            h.finish(r, 'drag')
        end
        -- the own selection: changed by a click on the field (not a box) -> the enemy's pick goes
        local now = real()
        if now - (h.sel_ms or -1e9) >= M.SELECTION_MS then
            h.sel_ms = now
            local s = selection()
            if s and h.sel ~= nil and s ~= h.sel and not h.seen and now - h.box_ms > M.AFTER_BOX_MS
                and now - (h.card_ms or -1e9) > M.AFTER_BOX_MS then
                pcall(opts.on_clear, 'own selection changed')
            end
            h.sel = s or h.sel
        end
        if M.FIT_USE_PX > 0 and now - h.fit_ms >= (h.fit and h.fit.err <= M.FIT_GOOD_PX and 20000 or M.FIT_EVERY_MS) then
            h.fit_ms = now
            local ok, err = pcall(h.calibrate, 'timer')
            if not ok and not h.fit_told then
                h.fit_told = true
                opts.emit('box_fit', {t = opts.now_ms(), error = tostring(err)})
            end
        end
    end

    local function loop()
        pcall(poll)
        if not h.stopped then opts.bm:real_callback(loop, M.POLL_MS, 'bai_box_pick') end
    end

    function h.stop() h.stopped = true end
    -- a click on our enemy card (apps.battle.enemy_cards): the own selection may change with it, the pick must not go
    function h.card_clicked() h.card_ms = real() end

    -- a click on the field (the game's click component 'root'): the pick goes unless it ended a box
    pcall(function()
        env.core:add_listener('bai_box_pick_click', 'ComponentLClickUp', true, function(ctx)
            local s = tostring(ctx.string)
            if h.clicks_told == nil then h.clicks_told = 0 end
            if h.clicks_told < 40 then
                h.clicks_told = h.clicks_told + 1
                opts.emit('field_click', {t = opts.now_ms(), string = s})
            end
            if s:match('^bai_enemy_') then h.card_ms = real() end
            -- an enemy's banner over its unit clicked (the component is named by the unit's id): as its card
            local name = h.by_uid[s]
            if name then
                h.card_ms = real()
                pcall(opts.on_unit, name)
                return
            end
            if M.FIELD[s] then
                opts.bm:real_callback(function()
                    if not h.seen and real() - h.box_ms > M.AFTER_BOX_MS then pcall(opts.on_clear, 'field click') end
                end, 150, 'bai_box_pick_click')
            end
        end, true)
    end)

    -- what is there at the start (the check: the rectangle's component while idle, the banners)
    local c, name = rectangle()
    local idle = c and box(c) or nil
    opts.emit('box_probe', {t = opts.now_ms(), rectangle = name, idle = idle and {idle.x, idle.y, idle.w, idle.h},
        idle_visible = c and visible(c) or false, holder = f(root(), M.HOLDER) ~= nil, banners = #h.points()})
    opts.bm:real_callback(loop, M.POLL_MS, 'bai_box_pick')
    return h
end

return M

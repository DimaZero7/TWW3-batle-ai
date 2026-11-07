-- The box selection over the field for the enemy (the live labelling, build/diag_chain/live_label.md): the game's own
-- box selection (left button held and dragged over the field) takes only the own units. Its rectangle is a UI component
-- of the game (ui/battle ui/box_selection.twui.xml in ui3.pack, found by name), so it is watched every M.POLL_MS of real
-- time: while it shows, its rectangle is kept; when it goes, the enemy units whose unit banner (unit_id_holder's
-- component of the unit, where the game draws the unit's icon on the screen) stands inside it are the box's enemies ->
-- opts.on_box(names, rect). Shift / Ctrl are not seen by a script: a box always replaces the pick (an empty one keeps it).
-- Found by name only (find_uicomponent): a ChildCount / Find walk breaks the game's script bindings for good.
local M = {}

M.NAMES = {'box_selection'}  -- the rectangle's component (the twui's name)
M.HOLDER = 'unit_id_holder'
M.POLL_MS = 100
M.MIN = 4                    -- px: a smaller rectangle is a click, not a box

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

-- opts: bm, env {core, find_uicomponent}, units = {{name, unit}} (the enemy), on_box(names, rect), emit, now_ms
function M.start(opts)
    local env = opts.env
    local f = function(...) return env.find_uicomponent(...) end
    local root = function() return env.core:get_ui_root() end
    local h = {seen = false, last = nil, boxes = 0}

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
        for _, name in ipairs(M.NAMES) do
            local c = f(root(), name)
            if c then return c, name end
        end
        return nil
    end

    -- every unit's banner point on the screen now ({{name, x, y}}; the units the game hides have none)
    function h.points()
        local holder = f(root(), M.HOLDER)
        local out = {}
        if not holder then return out end
        for _, it in ipairs(opts.units) do
            local c = f(holder, tostring(it.unit:unique_ui_id()))
            if c and visible(c) then
                local b = box(c)
                out[#out + 1] = {it.name, b.x + b.w / 2, b.y + b.h / 2}
            end
        end
        return out
    end

    function h.finish(rect, why)
        local points = h.points()
        local names = M.inside(rect, points)
        h.boxes = h.boxes + 1
        opts.emit('box_pick', {t = opts.now_ms(), why = why, rect = {rect.x, rect.y, rect.w, rect.h}, units = names,
            banners = #points})
        pcall(opts.on_box, names, rect)
        return names
    end

    local function poll()
        if h.stopped then return end
        local c, name = rectangle()
        if c and visible(c) then
            local b = box(c)
            if b.w >= M.MIN and b.h >= M.MIN then
                h.last = b
                if not h.seen then
                    h.seen = true
                    opts.emit('box_seen', {t = opts.now_ms(), name = name, rect = {b.x, b.y, b.w, b.h}})
                end
            end
        elseif h.seen then
            h.seen = false
            h.finish(h.last, 'drag')
        end
    end

    local function loop()
        pcall(poll)
        if not h.stopped then opts.bm:real_callback(loop, M.POLL_MS, 'bai_box_pick') end
    end

    function h.stop() h.stopped = true end

    -- what is there at the start (the check: is the rectangle's component in the tree while idle, are the banners)
    local c, name = rectangle()
    local idle = c and box(c) or nil
    opts.emit('box_probe', {t = opts.now_ms(), rectangle = name, idle = idle and {idle.x, idle.y, idle.w, idle.h},
        idle_visible = c and visible(c) or nil, holder = f(root(), M.HOLDER) ~= nil, banners = #h.points()})
    opts.bm:real_callback(loop, M.POLL_MS, 'bai_box_pick')
    return h
end

return M

-- The labelling's marks over the live battle (tools/label/game.py writes them; build/diag_chain/ingame_ui.md, what the
-- game can draw): the overlay file tww3_bai_nn_overlay.txt, read on real-time timers (they run on a pause too):
--
--   tww3_bai_nn_overlay 1
--   version 7
--   text <a line shown on the screen: the subtitles component, UI/Common UI/scripted_subtitles.twui.xml>
--   mark <unit name> g|t <k>      an icon over the unit (its unit id's script_ping_parent, CA's script_unit:add_ping_icon
--                                 road, ui/battle ui/unit_ping_indicator.twui.xml) with our image ui/bai/label/g<k>.png
--                                 (my unit in group k) or t<k>.png (an enemy, group k's target): tools/label/icons.py
--   zone <k> <x> <z> <b> <front> <depth>   a group's zone: ground ping icons (bm:add_ping_icon) at its four corners
--   ping <type>                   the ground ping's type (a number of the engine's; default M.PING)
--   button 1                      a dialogue box (UI/Common UI/dialogue_box) with the text; its tick writes
--                                 tww3_bai_nn_label_click.txt 'keep <version>' (game.py: «Оставить как есть»)
--   camera <x> <z> <tx> <tz> <h>  the camera moved to look from (x, z) at (tx, tz), h m over the ground
--   catalogue <x> <z> <dx> <dz> <from> <to>   ground pings of the types from..to in a row (the probe of the types)
--   end
-- A new version: the old marks removed, the new drawn; every engine call through pcall, its result counted in the
-- event 'label_overlay' (ok / failed with the first errors). Nothing of this runs without config.label.
local exchange = require('apps.bridge.exchange_adapter')

local M = {}

M.FILE = 'tww3_bai_nn_overlay.txt'
M.CLICK = 'tww3_bai_nn_label_click.txt'
M.FORMAT = 'tww3_bai_nn_overlay 1'
M.PREFIX = 'tww3_bai_overlay_'
M.PING = 8
M.IMAGE = 'ui/bai/label/%s%d.png'
M.PING_LAYOUT = 'ui/battle ui/unit_ping_indicator.twui.xml'
M.TEXT_LAYOUT = 'UI/Common UI/scripted_subtitles.twui.xml'
M.DIALOG_LAYOUT = 'UI/Common UI/dialogue_box'
M.DIALOG_AT = {30, 110}

local function words(line)
    local out = {}
    for w in line:gmatch('%S+') do out[#out + 1] = w end
    return out
end

-- The overlay file's text -> {version, text, marks = {{unit, kind, k}}, zones = {{k, x, z, b, front, depth}}, ping,
-- button, camera, catalogue} or nil (not complete).
function M.parse(text)
    if type(text) ~= 'string' then return nil end
    local lines = {}
    for line in (text .. '\n'):gmatch('([^\n]*)\n') do lines[#lines + 1] = line:gsub('\r$', '') end
    while #lines > 0 and lines[#lines] == '' do lines[#lines] = nil end
    if lines[1] ~= M.FORMAT or lines[#lines] ~= 'end' then return nil end
    local doc = {marks = {}, zones = {}, ping = M.PING, button = false}
    for i = 2, #lines - 1 do
        local line = lines[i]
        local w = words(line)
        local k = w[1]
        if k == 'version' then
            doc.version = tonumber(w[2])
        elseif k == 'text' then
            doc.text = line:sub(6)
        elseif k == 'mark' and #w >= 4 then
            doc.marks[#doc.marks + 1] = {unit = w[2], kind = w[3], k = tonumber(w[4])}
        elseif k == 'zone' and #w >= 7 then
            doc.zones[#doc.zones + 1] = {k = tonumber(w[2]), x = tonumber(w[3]), z = tonumber(w[4]), b = tonumber(w[5]),
                front = tonumber(w[6]), depth = tonumber(w[7])}
        elseif k == 'ping' then
            doc.ping = tonumber(w[2]) or M.PING
        elseif k == 'button' then
            doc.button = w[2] == '1'
        elseif k == 'camera' and #w >= 6 then
            doc.camera = {x = tonumber(w[2]), z = tonumber(w[3]), tx = tonumber(w[4]), tz = tonumber(w[5]),
                h = tonumber(w[6])}
        elseif k == 'catalogue' and #w >= 7 then
            doc.catalogue = {x = tonumber(w[2]), z = tonumber(w[3]), dx = tonumber(w[4]), dz = tonumber(w[5]),
                from = tonumber(w[6]), to = tonumber(w[7])}
        end
    end
    if not doc.version then return nil end
    return doc
end

-- The four corners {x, z} of a zone (the simulator's bearings: 0 = +z, clockwise; front across, depth along).
function M.corners(z)
    local a = math.rad(z.b)
    local fx, fz = math.sin(a), math.cos(a)          -- forward
    local rx, rz = fz, -fx                           -- right
    local out = {}
    for _, s in ipairs({{-1, -1}, {1, -1}, {1, 1}, {-1, 1}}) do
        local u, v = s[1] * z.front / 2, s[2] * z.depth / 2
        out[#out + 1] = {x = z.x + u * rx + v * fx, z = z.z + u * rz + v * fz}
    end
    return out
end

-- opts: bm, emit(event, fields), unit_of(name) -> unit or nil, height(x, z), now_ms(), guard(fn), finished(),
-- poll_ms (default 250), env (the globals: core, UIComponent, find_uicomponent, cco, common; default _G).
function M.start(opts)
    local bm = opts.bm
    local env = opts.env or _G
    local guard = opts.guard or function(fn) return fn end
    local h = {version = nil, pings = {}, icons = {}, shown = 0, stats = {}}
    local count, errors

    local function try(what, fn, ...)
        local ok, err = pcall(fn, ...)
        count[what] = count[what] or {ok = 0, failed = 0}
        if ok then
            count[what].ok = count[what].ok + 1
        else
            count[what].failed = count[what].failed + 1
            if #errors < 6 then errors[#errors + 1] = what .. ': ' .. tostring(err) end
        end
        return ok, err
    end

    local function root()
        return env.core:get_ui_root()
    end

    local function clear()
        for _, p in ipairs(h.pings) do try('ping_remove', function() bm:remove_ping_icon(p.x, p.y, p.z) end) end
        h.pings = {}
        for _, uic in ipairs(h.icons) do
            try('icon_remove', function() uic:Destroy() end)
        end
        h.icons = {}
    end

    local function icon_over(unit, path, name)
        local f = env.find_uicomponent
        local holder = f(root(), 'unit_id_holder')
        assert(holder, 'no unit_id_holder')
        local uid = f(holder, tostring(unit:unique_ui_id()))
        assert(uid, 'no unit id component')
        local parent = f(uid, 'script_ping_parent')
        assert(parent, 'no script_ping_parent')
        local uic = env.UIComponent(parent:CreateComponent(name, M.PING_LAYOUT))
        if env.cco then pcall(function() uic:SetContextObject(env.cco('CcoBattleUnit', unit:unique_ui_id())) end) end
        local icon = f(uic, 'icon')
        assert(icon, 'no icon child')
        icon:SetImagePath(path)
        h.icons[#h.icons + 1] = uic
    end

    local function ping(x, z, kind)
        local y = opts.height(x, z)
        bm:add_ping_icon(x, y, z, kind, false, 0)
        h.pings[#h.pings + 1] = {x = x, y = y, z = z}
    end

    local function text(line)
        local f = env.find_uicomponent
        local holder = f(root(), 'bai_label_text')
        if not holder then
            root():CreateComponent('bai_label_text', M.TEXT_LAYOUT)
            holder = f(root(), 'bai_label_text')
        end
        local t = f(root(), 'bai_label_text', 'text_child')
        assert(t, 'no text_child')
        if line and line ~= '' then
            t:SetStateText(line, 'bai_label')
            t:SetVisible(true)
            pcall(function() t:RegisterTopMost() end)
        else
            t:SetVisible(false)
        end
    end

    local function dialog(doc)
        local f = env.find_uicomponent
        local old = f(root(), 'bai_label_next')
        if old then pcall(function() old:UnLockPriority() end); old:Destroy() end
        if not doc.button then return end
        local box = env.UIComponent(root():CreateComponent('bai_label_next', M.DIALOG_LAYOUT))
        local t = f(root(), 'bai_label_next', 'DY_text')
        if t then
            t:SetStateText((doc.text or '') .. '\n\nГалочка — оставить как есть, крестик — убрать окно', 'bai_label')
        end
        pcall(function() box:PropagatePriority(99) end)
        -- aside, not over the field's middle (MoveTo: the screen's pixels of the box's top left)
        if M.DIALOG_AT then pcall(function() box:MoveTo(M.DIALOG_AT[1], M.DIALOG_AT[2]) end) end
        if not h.listening then
            h.listening = true
            env.core:add_listener('bai_label_click', 'ComponentLClickUp', function(c)
                return c.string == 'button_tick' or c.string == 'button_cancel'
            end, function(c)
                if c.string == 'button_tick' then exchange.write(M.CLICK, 'keep ' .. tostring(h.version) .. '\n') end
                local b = f(root(), 'bai_label_next')
                if b then pcall(function() b:UnLockPriority() end); pcall(function() b:Destroy() end) end
                opts.emit('label_overlay_click', {t = opts.now_ms(), button = c.string, version = h.version})
            end, true)
        end
    end

    local function apply(doc)
        count, errors = {}, {}
        clear()
        for _, m in ipairs(doc.marks) do
            local unit = opts.unit_of(m.unit)
            if unit then
                try('mark_' .. m.kind, icon_over, unit, string.format(M.IMAGE, m.kind, m.k),
                    'bai_mark_' .. m.unit .. '_' .. tostring(doc.version))
            else
                try('mark_' .. m.kind, error, 'unknown unit ' .. tostring(m.unit))
            end
        end
        for _, z in ipairs(doc.zones) do
            for _, c in ipairs(M.corners(z)) do try('zone_ping', ping, c.x, c.z, doc.ping) end
        end
        if doc.catalogue then
            local c = doc.catalogue
            for k = c.from, c.to do try('catalogue', ping, c.x + (k - c.from) * c.dx, c.z + (k - c.from) * c.dz, k) end
            local paths = {}
            for k = 0, 30 do
                local ok, p = pcall(function() return env.common.PingIconPath(k) end)
                paths[#paths + 1] = ok and tostring(p) or ('error: ' .. tostring(p))
            end
            opts.emit('label_overlay_pings', {paths = paths})
        end
        try('text', text, doc.text)
        try('button', dialog, doc)
        if doc.camera then
            local c = doc.camera
            try('camera', function()
                local y = opts.height(c.tx, c.tz)
                bm:camera():move_to(opts.vec(c.x, y + c.h, c.z), opts.vec(c.tx, y, c.tz), 0, false, 0)
            end)
        end
        h.version = doc.version
        h.shown = h.shown + 1
        local row = {t = opts.now_ms(), version = doc.version, marks = #doc.marks, zones = #doc.zones,
            errors = #errors > 0 and errors or nil}
        for what, c in pairs(count) do row[what] = c end
        opts.emit('label_overlay', row)
        if opts.flush then opts.flush() end
    end

    local function poll()
        if opts.finished and opts.finished() then return end
        local doc = M.parse(exchange.read(M.FILE))
        if doc and doc.version ~= h.version then apply(doc) end
        bm:real_callback(guard(poll), opts.poll_ms or 250, M.PREFIX .. 'poll')
    end

    function h.stop()
        count, errors = {}, {}
        clear()
        pcall(text, nil)
    end

    exchange.remove(M.FILE)
    exchange.remove(M.CLICK)
    bm:real_callback(guard(poll), opts.poll_ms or 250, M.PREFIX .. 'poll')
    return h
end

return M

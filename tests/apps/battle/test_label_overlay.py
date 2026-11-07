"""apps.battle.label_overlay: the overlay file parsed (a file without its end is not taken), a zone's corners, and the
drawing with a fake engine: the marks over units, the zone's pings, the text; a new version removes the old marks; a
failing engine call is counted, not an error."""
from tests.lua_runtime import load, new_runtime


def test_the_overlay_file_and_the_corners():
    lua = new_runtime()
    lua.globals().ov = load(lua, "apps.battle.label_overlay")
    lua.execute("""
        local text = table.concat({ov.FORMAT, 'version 3', 'text Ты — оперативник. Пауза, t = 30 с',
            'mark own_spear_1 g 2', 'mark enemy_lord t 1', 'zone 2 10 20 90 80 48', 'ping 9', 'button 1',
            'camera 0 -100 0 0 120', 'end'}, '\\n') .. '\\n'
        local d = ov.parse(text)
        assert(d.version == 3 and d.text == 'Ты — оперативник. Пауза, t = 30 с' and d.ping == 9 and d.button)
        assert(#d.marks == 2 and d.marks[1].unit == 'own_spear_1' and d.marks[1].kind == 'g' and d.marks[1].k == 2)
        assert(#d.zones == 1 and d.zones[1].front == 80 and d.camera.h == 120)
        assert(ov.parse(ov.FORMAT .. '\\nversion 1\\n') == nil)          -- no end: not complete
        local c = ov.corners({x = 0, z = 0, b = 90, front = 80, depth = 40})
        -- facing +x: the front runs along z (+-40), the depth along x (+-20)
        for _, p in ipairs(c) do assert(math.abs(math.abs(p.x) - 20) < 1e-6 and math.abs(math.abs(p.z) - 40) < 1e-6) end
    """)


def test_the_marks_are_drawn_and_replaced():
    lua = new_runtime()
    lua.globals().ov = load(lua, "apps.battle.label_overlay")
    lua.globals().ex = load(lua, "apps.bridge.exchange_adapter")
    lua.execute("""
        local real, events, pings, removed, comps = {}, {}, {}, 0, {}
        local function comp(name)
            local c = {name = name, kids = {}}
            function c:CreateComponent(n, path) local k = comp(n); k.path = path; self.kids[n] = k; comps[#comps + 1] = k
                k.kids.icon = comp('icon'); k.kids.text_child = comp('text_child'); return k end
            function c:SetImagePath(p) self.image = p end
            function c:SetStateText(t) self.text = t end
            function c:SetVisible(v) self.visible = v end
            function c:Destroy() self.destroyed = true end
            function c:unique_ui_id() return 42 end
            return c
        end
        local ui = comp('root')
        local holder = ui:CreateComponent('unit_id_holder'); local uid = holder:CreateComponent('42')
        uid:CreateComponent('script_ping_parent')
        local function find(c, ...)
            for _, n in ipairs({...}) do
                if type(n) ~= 'string' then return nil end
                local found
                local function walk(x) if found then return end; if x.kids[n] then found = x.kids[n] return end
                    for _, k in pairs(x.kids) do walk(k) end end
                walk(c); c = found; if not c then return nil end
            end
            return c
        end
        local env = {core = {get_ui_root = function() return ui end, add_listener = function() end},
            find_uicomponent = find, UIComponent = function(c) return c end}
        local bm = {}
        function bm:add_ping_icon(x, y, z, k) pings[#pings + 1] = {x, z, k} end
        function bm:remove_ping_icon() removed = removed + 1 end
        function bm:real_callback(fn) real[#real + 1] = fn end
        local unit = {unique_ui_id = function() return 42 end}
        local files = {}
        local h = ov.start({bm = bm, env = env, emit = function(e, f) f.event = e; events[#events + 1] = f end,
            unit_of = function(n) if n == 'own_spear_1' then return unit end end, height = function() return 0 end,
            now_ms = function() return 0 end, vec = function() end})
        local function feed(text) ex.write(ov.FILE, text); table.remove(real, 1)() end
        feed(table.concat({ov.FORMAT, 'version 1', 'text привет', 'mark own_spear_1 g 2', 'mark nobody t 1',
            'zone 2 0 0 0 80 48', 'end'}, '\\n'))
        local e = events[#events]
        assert(e.version == 1 and e.mark_g.ok == 1 and e.mark_t.failed == 1 and e.zone_ping.ok == 4, e.errors and e.errors[1])
        assert(#pings == 4)
        local icon_found = false
        for _, c in ipairs(comps) do if c.kids.icon.image == 'ui/bai/label/g2.png' then icon_found = true end end
        assert(icon_found)
        feed(table.concat({ov.FORMAT, 'version 2', 'zone 1 5 5 0 80 48', 'end'}, '\\n'))
        assert(removed == 4 and #pings == 8 and events[#events].version == 2)
        h.stop()
        os.remove(ov.FILE)
    """)

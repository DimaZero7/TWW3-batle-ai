"""apps.mem.entities: the men of battle units from the game's memory, on a fake memreader (no game). The offsets here
are made up (the real ones live outside Git: build/mem/sites.json)."""
import pytest

from tests.lua_runtime import load, new_runtime

FAKE = """
    L = {i_unit = 0x10, u_battle = 0x30, b_world = 0x100, w_tick = 0x98, u_men = 0x180, m_unit = 0x2e8,
         m_live = 0x108, m_pos = 0x88, m_vel = 0xe0, m_face = 0xb0, m_hp = 0xb1c, m_anim = 0x26c, m_mode = 0x270,
         m_react = 0x324, m_mark = 0xa30, m_fit = 0xb13, m_flags = 0x68, m_brace = 0xbc, m_speed = 0x23c}
    MEM, CODE = {}, {}
    mr = {base = 4096, version = 1.2, plus_version = '0.15.1'}
    function mr.add(p, n) return p + n end
    function mr.eq(a, b) return a == b end
    function mr.tostring(p) return string.format('%x', p) end
    function mr.read(p, off, n) return string.sub(CODE[p + off] or '', 1, n) end
    function mr.find_pattern(pat) return 4096 + 256, 1 end
    function mr.read_pointer(p, off)
        local m = MEM[p]; if not m then error('failed to read memory') end
        return m[off or 0] or 0
    end
    function mr.is_null(p) return p == 0 or p == false or p == nil end
    function mr.read_vector(p, off, field, stride) return MEM[p][off] end
    function mr.read_struct(p, off, layout)
        local m = MEM[p]; if not m then error('failed to read memory') end
        local out = {}
        for k, f in pairs(layout) do out[k] = m[off + f[1]] or 0 end
        return out
    end
    function mr.hook() end
    function mr.hook_next() end
    function mr.read_vector_ok() end
    PREFIX = {}
    function mr.patcher(owner)
        return {prefix = function(self, at, sig, name, fn) PREFIX[name] = fn end,
                remove_all = function(self) PREFIX = {} end}
    end
    BATTLE, WORLD = 600, 650
    MEM[BATTLE] = {[L.b_world] = WORLD}
    MEM[WORLD] = {[L.w_tick] = 123}
    function man(p, unit, x, z, react, mark, live)
        MEM[p] = {[L.m_unit] = unit, [L.m_live] = live or 0x13, [L.m_pos] = x, [L.m_pos + 4] = 1, [L.m_pos + 8] = z,
                  [L.m_vel] = 2, [L.m_vel + 8] = -3, [L.m_face] = 16384, [L.m_hp] = 60, [L.m_anim] = 0x37,
                  [L.m_mode] = 2, [L.m_react] = react or 0, [L.m_mark] = mark or -100, [L.m_fit] = 2,
                  [L.m_flags] = 8, [L.m_brace] = 4, [L.m_speed] = 10.84}
    end
    -- two battle units (2000: knights, 3000: clanrats) behind their script objects 20 / 30
    MEM[2000] = {[L.u_battle] = BATTLE, [L.u_men] = {2100, 2200}}
    MEM[3000] = {[L.u_battle] = BATTLE, [L.u_men] = {3100, 3200, 3300}}
    man(2100, 2000, 1, 2); man(2200, 2000, 3, 4, 0, 11.5)
    man(3100, 3000, 5, 6, 0xf); man(3200, 3000, 7, 8, 0x11); man(3300, 3000, 9, 10, 0, -100, 0x03)
    MEM[20] = {[L.i_unit] = 2000}; MEM[30] = {[L.i_unit] = 3000}
    function script_unit(obj)
        return {ammo_left = function(self) if PREFIX.ammo_left then PREFIX.ammo_left(nil, obj) end return 0 end}
    end
    CODE[4096 + 256] = string.char(0x48, 0x89, 0x5c, 0x24)
    CODE[4096 + 512] = string.char(0x48, 0x89, 0x5c, 0x24, 0x08)
    CFG = {sites = {proj_update = {hi = 0, lo = 256, sig = 'void(pointer)', expect = '48 89 5c'},
                    ammo_left = {hi = 0, lo = 512, sig = 'int32(pointer, pointer)', expect = '48 89 5c 24 08'},
                    battle_ref = {hi = 0, lo = 768, sig = '', expect = ''},
                    battle_global = {hi = 0, lo = 1024, sig = '', expect = ''}},
           layout = L}
"""


@pytest.fixture
def lua():
    runtime = new_runtime()
    runtime.globals().E = load(runtime, "apps.mem.entities")
    runtime.execute(FAKE)
    return runtime


def test_bind_reads_the_battle_unit_through_the_hook_and_checks_its_men(lua):
    lua.execute("""
        H = assert(E.attach(mr, CFG))
        OK1, I1 = H.bind('kn', script_unit(20))
        MEM[3200][L.m_unit] = 2000                          -- a man of another unit in the vector: refused
        OK2, I2 = H.bind('cr', script_unit(30))
        MEM[3200][L.m_unit] = 3000
        OK3, I3 = H.bind('cr', script_unit(30))
        H.unhook()
        OK4, I4 = H.bind('x', script_unit(30))              -- no hook any more: nothing seen
    """)
    g = lua.globals()
    assert g.OK1 is True and g.I1.n == 2 and g.I1.tick == 123
    assert g.OK2 is False and "not of this unit 1" in g.I2
    assert g.OK3 is True and g.I3.n == 3
    assert g.OK4 is False and "no unit" in g.I4


def test_poll_gives_each_man_a_row_and_a_battle_wide_id(lua):
    lua.execute("""
        H = assert(E.attach(mr, CFG))
        H.bind('kn', script_unit(20)); H.bind('cr', script_unit(30))
        D1 = H.poll()
        MEM[2000][L.u_men] = {2200}                          -- a man left the vector
        D2 = H.poll({'kn'})
    """)
    g = lua.globals()
    d1, d2 = g.D1, g.D2
    n = len(lua.eval("E.ROW"))
    kn, cr = list(d1.units.kn.rows.values()), list(d1.units.cr.rows.values())
    assert d1.tick == 123 and d1.units.kn.n == 2 and len(kn) == 2 * n and len(cr) == 3 * n
    row = dict(zip(lua.eval("E.ROW").values(), kn[n:2 * n]))
    assert row == {"id": 2, "live": 1, "x": 3, "z": 4, "y": 1, "vx": 2, "vz": -3, "face": 16384, "hp": 60,
                   "anim": 0x37, "mode": 2, "react": 0, "mark": 11.5, "fit": 2, "flags": 8, "brace": 4,
                   "speed": 10.8, "tgt": 0, "tgt2": 0, "carried": 0, "chg": 0}
    assert [cr[k * n + 11] for k in range(3)] == [0xf, 0x11, 0]          # reactions: knocked down, getting up, none
    assert [cr[k * n + 1] for k in range(3)] == [1, 1, 0]                # bit 0x10 of the live byte
    assert len(d1.new) == 5 and d1.new[1][2] == "kn"
    rows2 = list(d2.units.kn.rows.values())
    assert len(rows2) == n and rows2[0] == 2 and len(d2.new or {}) == 0 and d2.units.cr is None


def test_discover_finds_pointers_to_enemies_and_the_rows_carry_the_targets(lua):
    lua.execute("""
        L.m_target, L.m_target_b = 0x368, 0x3d8
        H = assert(E.attach(mr, CFG))
        H.bind('kn', script_unit(20)); H.bind('cr', script_unit(30))
        MEM[2100][0x368] = 3100; MEM[2200][0x368] = 3200; MEM[2200][0x3d8] = 3300
        MEM[2100][0x500] = 5000; MEM[5000] = {[0x38] = 3100}  -- a pointer to an object holding an enemy: not followed
        HITS = H.discover('kn', 'cr', 6)
        D1 = H.poll({'kn'})                                   -- the clanrats not read yet: unknown pointers
        H.poll({'cr'}); D2 = H.poll({'kn'})
    """)
    g = lua.globals()
    assert dict(g.HITS) == {"368": 2, "3d8": 1}
    n = len(lua.eval("E.ROW"))
    kn1, kn2 = list(g.D1.units.kn.rows.values()), list(g.D2.units.kn.rows.values())
    t1, t2 = list(lua.eval("E.ROW").values()).index("tgt"), list(lua.eval("E.ROW").values()).index("tgt2")
    assert (kn1[t1], kn1[t2], kn1[n + t1], kn1[n + t2]) == (-1, 0, -1, -1)
    assert (kn2[t1], kn2[t2], kn2[n + t1], kn2[n + t2]) == (3, 0, 4, 5)


def test_the_charge_deadline_is_the_carrying_root_entitys(lua):
    lua.execute("""
        L.m_carrier, L.m_charge = 0x2e0, 0xa3c
        H = assert(E.attach(mr, CFG))
        H.bind('kn', script_unit(20))
        MEM[2100][0x2e0] = 9000; MEM[9000] = {[0xa3c] = 31.25}   -- a rider on his horse: the horse's deadline
        MEM[2200][0xa3c] = 12.5                                    -- a man on foot: his own
        D = H.poll({'kn'})
    """)
    names = list(lua.eval("E.ROW").values())
    n = len(names)
    rows = list(lua.globals().D.units.kn.rows.values())
    c, g = names.index("carried"), names.index("chg")
    assert (rows[c], rows[g], rows[n + c], rows[n + g]) == (1, 31.25, 0, 12.5)


def test_attach_refuses_without_the_men_layout(lua):
    lua.execute("CFG.layout = {i_unit = 0x10}")
    st = lua.eval("select(2, E.attach(mr, CFG))")
    assert st.ok is False and "men layout" in st.why

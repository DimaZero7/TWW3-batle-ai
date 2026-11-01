"""apps.mem.hits: which body a projectile 'in a man' is in, on a fake memreader (made-up offsets; the real ones live
outside Git: build/mem/sites.json)."""
import pytest

from tests.lua_runtime import load, new_runtime

FAKE = """
    L = {m_unit = 0x10, m_hp = 0x18, m_pos = 0x20}
    MEM = {}
    mr = {}
    function mr.tostring(p) return string.format('%x', p) end
    function mr.is_null(p) return p == 0 or p == false end
    function mr.read_pointer(p, off)
        local m = MEM[p]; if not m then error('failed to read memory') end
        return m[off or 0] or 0
    end
    function mr.read_struct(p, off, layout)
        local m = MEM[p]; if not m then error('failed to read memory') end
        local out = {}
        for k, f in pairs(layout) do out[k] = m[off + f[1]] or (f[2] == 'pointer' and false or 0) end
        return out
    end
    READS = 0
    local rp = mr.read_pointer
    mr.read_pointer = function(p, off) READS = READS + 1 return rp(p, off) end
    -- men 501, 502 (ids 1, 2) of unit 900; a gun 600 of unit 900 (no man of its vector); a shooter 700 (id 3)
    IDS = {['1f5'] = 1, ['1f6'] = 2, ['2bc'] = 3}
    MEM[600] = {[L.m_unit] = 900, [L.m_hp] = 540, [L.m_pos] = 12.34, [L.m_pos + 4] = 1, [L.m_pos + 8] = -5}
    function obj(p, target, shooter) MEM[p] = {[0x20] = shooter, [0x48] = target, [0x50] = 77} end
"""


@pytest.fixture
def lua():
    runtime = new_runtime()
    runtime.globals().M = load(runtime, "apps.mem.hits")
    runtime.execute(FAKE)
    return runtime


def test_learns_the_offsets_then_names_other_bodies(lua):
    lua.execute("""
        H = M.new(mr, L, IDS, {learn_n = 2, units = {gun_unit = '384'}})
        obj(1000, 501, 700); W1 = H.who(1000)
        obj(1001, 502, 700); W2 = H.who(1001)              -- learned now: 0x20 (shooter), 0x48 (target)
        R0 = READS
        obj(1002, 600, 700); W3 = H.who(1002)              -- a body that is no known man: other 1
        R1 = READS - R0
        obj(1003, 600, 700); W4 = H.who(1003)              -- the same body: read once only
        O = H.drain_other()
        HP = H.poll_other()
    """)
    g = lua.globals()
    assert list(g.W1.values()) == [0x20, 3, 0x48, 1]
    assert list(g.H.offsets().values()) == [0x20, 0x48]
    assert list(g.W3.values()) == [0x20, 3, 0x48, -1] and list(g.W4.values()) == [0x20, 3, 0x48, -1]
    assert g.R1 == 2                                      # only the learned offsets are read
    o = [list(r.values()) for r in g.O.values()]
    assert o == [[1, "258", "gun_unit", 540, 12.3, 1, -5]]
    assert [list(r.values()) for r in g.HP.values()] == [[1, 540]]


def test_other_bodies_are_capped_and_unreadable_ones_are_guarded(lua):
    lua.execute("""
        H = M.new(mr, L, IDS, {learn_n = 1, other_max = 1})
        obj(1000, 501, 700); H.who(1000)
        obj(1001, 4242, 700); A = H.who(1001)              -- unreadable body: kept as other 1, its fields false
        obj(1002, 600, 700); B = H.who(1002)               -- over the cap: not named
        O = H.drain_other()
    """)
    g = lua.globals()
    assert list(g.A.values()) == [0x20, 3, 0x48, -1]
    assert list(g.B.values()) == [0x20, 3]
    assert list(g.O[1].values())[2:] == [False] * 5

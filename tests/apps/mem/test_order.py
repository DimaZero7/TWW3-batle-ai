"""apps.mem.order: the formation's state of battle units from the game's memory, on a fake memreader (no game). The offsets
here are made up (the real ones live outside Git: build/mem/sites.json)."""
import pytest

from tests.lua_runtime import load, new_runtime

FAKE = """
    L = {u_men = 0x11, form = {
        u_beh_n = 0x10, u_beh = 0x20, u_group = 0x30, u_abil_n = 0x34, u_stat = 0x40, u_rad = 0x50,
        b_state = 0x60, b_react = 0x68, b_group = 0x70, b_fl = {0x78},
        g_n = 0x12, g_u32 = {0x60, 0x64}, g_u8 = {0x68}, g_order = 0x100, g_sub_n = 0x70, g_sub = 0x80, g_sub_step = 0x10,
        o_job = 0x90, j_u32 = {0x20}, j_u8 = {0x24}, j_head = 0x58, j_next = 0x10, j_prev = 0x8, j_f0 = 0x60, j_f1 = 0x64,
        j_done = 4, n_f0 = 0x18, n_f1 = 0x1c, s_fl = {0x18}, m_u8 = {0x5c}}}
    MEM = {}
    mr = {}
    function mr.eq(a, b) return a == b end
    function mr.tostring(p) return string.format('%x', p) end
    function mr.is_null(p) return p == 0 or p == false or p == nil end
    function mr.read_pointer(p, off)
        local m = MEM[p]; if not m then error('failed to read memory') end
        return m[off or 0] or 0
    end
    function mr.read_vector(p, off, field, stride) return MEM[p][off] end
    function mr.read_struct(p, off, layout)
        local m = MEM[p]; if not m then error('failed to read memory') end
        local out = {}
        for k, f in pairs(layout) do out[k] = m[off + f[1]] or 0 end
        return out
    end
    -- a unit (1000) with a behaviour (1100) in a state (77), its group (1200), the order's job (1500) and two nodes
    MEM[1000] = {[L.form.u_beh_n] = 1, [L.form.u_beh] = 1100, [L.form.u_group] = 1200, [L.form.u_stat] = 1.0,
                 [L.form.u_stat + 4] = -0.1, [L.u_men] = {1300, 1400}}
    MEM[1100] = {[0] = 55, [L.form.b_state] = 77, [L.form.b_react] = 0, [L.form.b_group] = 1200, [0x78] = 1}
    MEM[1200] = {[L.form.g_n] = 2, [0x60] = 123, [0x64] = 7, [0x68] = 1, [L.form.g_order] = 66, [L.form.g_sub_n] = 2,
                 [L.form.g_sub + 0x10] = 1600}
    MEM[1600] = {[0] = 88, [0x18] = 1}
    MEM[1300] = {[0x5c] = 1}; MEM[1400] = {[0x5c] = 0}
    IDS = {['514'] = 5, ['578'] = 6}
"""


@pytest.fixture
def lua():
    runtime = new_runtime()
    runtime.globals().O = load(runtime, "apps.mem.order")
    runtime.execute(FAKE)
    return runtime


def test_poll_reads_the_machine_the_groups_marks_and_the_men(lua):
    lua.execute("""
        R = O.reader(mr, L, IDS)
        R.add('u', 1000)
        for i = 1, 10 do D = R.poll() end
        ROW = D.units.u
    """)
    r = lua.globals().ROW
    assert r.n == 1 and r.gs == 1 and r.gn == 2 and r.sn == 2
    assert list(r.gu.values()) == [123, 7] and list(r.gb.values()) == [1] and list(r.be.values()) == [1]
    assert list(r.mc.values()) == [1]                     # one man of two holds the byte
    assert list(r.mr.values()) == [5, 1, 6, 0]            # every ROWS_EVERY polls: the rows {id, byte}
    assert r.spd[1] == 1 and abs(r.spd[2] + 0.1) < 1e-6


def test_a_pointer_is_a_small_number_told_once(lua):
    lua.execute("""
        R = O.reader(mr, L, IDS)
        R.add('u', 1000)
        D1 = R.poll(); D2 = R.poll()
    """)
    g = lua.globals()
    assert len(g.D1.new) == 4 and len(g.D2.new) == 0      # the behaviour's table, its state, the order's, the suborder's
    assert g.D1.units.u.st == g.D2.units.u.st != 0 and g.D1.units.u.re == 0


def test_the_job_is_read_once_when_done_and_only_at_a_heap_address(lua):
    lua.execute("""
        R = O.reader(mr, L, IDS)
        R.add('u', 1000)
        F = L.form
        MEM[1200][F.g_order + F.o_job] = 1500
        MEM[1500] = {[0x20] = 2, [0x58] = 1510}             -- not done yet
        MEM[1510] = {[0x10] = 1520, [0x8] = 0, [0x18] = 11.5, [0x1c] = 22.5}
        MEM[1520] = {[0x8] = 1510, [0x18] = 33.5, [0x1c] = 44.5}
        for i = 1, 5 do D1 = R.poll() end                   -- the upper half of the address is 0: not an address
        MEM[1200][F.g_order + F.o_job + 4] = 0x1f3
        for i = 1, 5 do D2 = R.poll() end                   -- an address, the job not done
        MEM[1500][0x20] = 4
        for i = 1, 5 do D3 = R.poll() end
        for i = 1, 5 do D4 = R.poll() end                   -- once
    """)
    g = lua.globals()
    assert len(g.D1.jobs) == 0 and len(g.D2.jobs) == 0 and len(g.D4.jobs) == 0
    job = g.D3.jobs[1]
    nodes = [list(job.nodes[k].values()) for k in sorted(job.nodes)]
    assert nodes == [[1, 11.5, 22.5], [2, 33.5, 44.5]]


def test_a_failed_read_is_an_error_not_a_stop(lua):
    lua.execute("""
        R = O.reader(mr, L, IDS)
        R.add('u', 1000); R.add('gone', 9999)
        D = R.poll()
    """)
    d = lua.globals().D
    assert d.units.u.n == 1 and d.units.gone is None and "gone" in d.errors[1]

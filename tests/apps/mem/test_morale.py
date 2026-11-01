"""apps.mem.morale: a unit's morale, its effects' list and the unit's own fields from the game's memory, on a fake
memreader (no game). The offsets here are made up (the real ones live outside Git: build/mem/sites.json)."""
import pytest

from tests.apps.mem.test_entities import FAKE
from tests.lua_runtime import load, new_runtime

MORALE = """
    L.u_morale, L.o_unit, L.o_state, L.o_points, L.o_target = 0x3000, 0x10, 0x40, 0x48, 0x4c
    L.o_fx_n, L.o_fx, L.n_next, L.n_id, L.n_val = 0x58, 0x60, 0x8, 0x10, 0x14
    L.o_timed, L.t_stride, L.o_contacts, L.c_id, L.c_side = 0x30, 12, 0x20, 0x10, 0x14
    L.o_surprise, L.o_waver, L.o_charge, L.o_rally, L.o_bytes = 0x88, 0x94, 0x98, 0x9c, 0x74
    L.u_fields = {hp_left = {0x500, 'float'}, combat = {0x600, 'int32'}}
    O = 2000 + L.u_morale
    local m = MEM[2000]
    m[L.u_morale + L.o_unit] = 2000
    m[L.u_morale + L.o_state], m[L.u_morale + L.o_points], m[L.u_morale + L.o_target] = 3, 41, 33
    m[L.u_morale + L.o_fx_n] = 2
    m[L.u_morale + L.o_fx + 8] = 7000                       -- the first node; the ring ends at the sentinel O + o_fx
    MEM[7000] = {[L.n_next] = 7100, [L.n_id] = 0x22, [L.n_val] = -7}
    MEM[7100] = {[L.n_next] = O + L.o_fx, [L.n_id] = 0x1b, [L.n_val] = -6}
    m[L.u_morale + L.o_timed + 4], m[L.u_morale + L.o_timed + 8] = 1, 7500
    MEM[7500] = {[0] = 0x35, [4] = 0, [8] = 101.5}
    m[L.u_morale + L.o_contacts + 8] = 7200
    MEM[7200] = {[L.n_next] = O + L.o_contacts, [L.c_id] = 9, [L.c_side] = 1}
    m[L.u_morale + L.o_surprise], m[L.u_morale + L.o_waver] = -1, 30
    m[L.u_morale + L.o_bytes + 9] = 2                        -- a flag byte (the routs count)
    m[0x500], m[0x600] = 0.75, 2
"""


@pytest.fixture
def lua():
    runtime = new_runtime()
    runtime.globals().MM = load(runtime, "apps.mem.morale")
    runtime.execute(FAKE)
    runtime.execute(MORALE)
    return runtime


def test_bind_checks_the_object_and_poll_reads_points_effects_timed_and_fields(lua):
    lua.execute("""
        H = assert(MM.attach(mr, CFG))
        OK, I = H.bind('kn', script_unit(20))
        H.unhook()
        D = H.poll()
    """)
    g = lua.globals()
    assert g.OK is True and g.I.st == 3 and g.I.pts == 41 and g.I.n == 2 and g.I.ok == 1
    row = g.D.units.kn
    assert (row.st, row.pts, row.tgt, row.n, row.ok, row.same) == (3, 41, 33, 2, 1, 1)
    assert list(row.fx.values()) == [0x22, -7, 0x1b, -6]
    assert list(row.tm.values()) == [0x35, 0, 101.5]
    assert list(row.ct.values()) == [9, 1] and row.cok == 1
    assert list(row.tim.values()) == [-1, 30, 0, 0] and row.by[10] == 2
    assert row.uf.hp_left == 0.75 and row.uf.combat == 2


def test_a_list_being_rebuilt_reads_not_ok_and_the_walk_is_bounded(lua):
    lua.execute("""
        H = assert(MM.attach(mr, CFG)); H.bind('kn', script_unit(20))
        MEM[2000][L.u_morale + L.o_fx_n] = 3                 -- the count no longer matches the nodes
        D1 = H.poll()
        MEM[2000][L.u_morale + L.o_fx_n] = 2
        MEM[7100][L.n_next] = 7000                            -- a ring that never meets the sentinel
        D2 = H.poll()
        MEM[7100][L.n_next] = 8888                            -- a node pointing to unreadable memory: an error row
        D3 = H.poll()
        E3 = H.errors()
    """)
    g = lua.globals()
    assert g.D1.units.kn.ok == 0
    assert g.D2.units.kn.ok == 0 and len(g.D2.units.kn.fx) == 2 * lua.eval("MM.FX_MAX")
    assert g.D3.units.kn is None and "kn" in g.E3[1]


def test_bind_refuses_an_object_not_pointing_back_and_attach_needs_the_layout(lua):
    lua.execute("""
        MEM[2000][L.u_morale + L.o_unit] = 3000
        H = assert(MM.attach(mr, CFG))
        OK, WHY = H.bind('kn', script_unit(20))
        CFG.layout = {i_unit = 0x10, u_men = 0x180}
        _, ST = MM.attach(mr, CFG)
    """)
    g = lua.globals()
    assert g.OK is False and "point back" in g.WHY
    assert g.ST.ok is False and "morale layout" in g.ST.why

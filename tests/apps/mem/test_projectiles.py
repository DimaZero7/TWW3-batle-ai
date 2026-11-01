"""apps.mem.projectiles: the projectiles in flight from the game's memory, on a fake memreader (no game). The offsets
here are made up (the real ones live outside Git: build/mem/sites.json)."""
import pytest

from tests.lua_runtime import load, new_runtime

FAKE = """
    L = {mgr_active = 0x40, p_shooter = 0x48, p_record = 0x50, p_state = 0x58, p_prev = 0x60, p_pos = 0x70,
         p_vel = 0x80, p_time = 0x90, s_feet = 0x10, r_category = 0x14, b_world = 0x100, w_battle = 0x18,
         w_proj = 0x200, i_unit = 0x10, u_ammo = 0x300}
    MEM, CODE, HOOKS, NEXT = {}, {}, {}, {}
    mr = {base = 4096, version = 1.2, plus_version = '0.15.1'}
    function mr.add(p, n) return p + n end
    function mr.eq(a, b) return a == b end
    function mr.tostring(p) return string.format('%x', p) end
    function mr.tonumber(v) return v end
    function mr.read(p, off, n) return string.sub(CODE[p + off] or '', 1, n) end
    function mr.find_pattern(pat) return PATTERN_AT, PATTERN_COUNT end
    function mr.read_int32(p, off)
        local m = MEM[p]; if not m then error('failed to read memory') end
        return m[off] or 0
    end
    mr.read_float = mr.read_int32
    function mr.read_pointer(p, off)
        local m = MEM[p]; if not m then error('failed to read memory') end
        return m[off or 0] or 0
    end
    function mr.is_null(p) return p == 0 or p == false end
    function mr.read_vector(p, off, field, stride) return MEM[p][off] end
    function mr.read_struct(p, off, layout)
        local m = MEM[p]; if not m then error('failed to read memory') end
        local out = {}
        for k, f in pairs(layout) do
            local v = m[off + f[1]]
            if f[2] == 'pointer' then out[k] = v or false else out[k] = v or 0 end
        end
        return out
    end
    function mr.hook(at, sig, fn) HOOKS[at] = {sig = sig, fn = fn, calls = 0} end
    function mr.hook_next(at, ...) NEXT[#NEXT + 1] = {at, ...} end
    function mr.unhook(at) HOOKS[at] = nil end
    PREFIX = {}
    function mr.patcher(owner)
        return {prefix = function(self, at, sig, name, fn) PREFIX[at] = fn end,
                remove_all = function(self) PREFIX = {} end}
    end
    function mr.hook_info(at) local h = HOOKS[at]; return h and {calls = h.calls, other_thread_calls = 0, attached = true} end
    function proj(p, st, x, t, rec, shooter)
        MEM[p] = {[L.p_state] = st, [L.p_pos] = x, [L.p_pos + 4] = 2, [L.p_pos + 8] = -x, [L.p_vel] = 50,
                  [L.p_vel + 4] = 1, [L.p_vel + 8] = 0, [L.p_time] = t, [L.p_record] = rec, [L.p_shooter] = shooter}
    end
    MGR = 900
    MEM[MGR] = {[L.mgr_active] = {}}
    MEM[700] = {[L.r_category] = 3}                       -- a database record
    MEM[800] = {[L.s_feet] = 10, [L.s_feet + 4] = 0, [L.s_feet + 8] = -80}   -- the shooting man
    SITE = {hi = 0, lo = 256, sig = 'void(pointer)', expect = '48 89 5c', pattern = '48 89 5c ??'}
    CODE[4096 + 256] = string.char(0x48, 0x89, 0x5c, 0x24)
    PATTERN_AT, PATTERN_COUNT = 4096 + 256, 1
    CODE[4096 + 512] = string.char(0x48, 0x89, 0x5c, 0x24, 0x08)
    CODE[4096 + 768] = string.char(0x48, 0x8b, 0x05)
    BATTLE, WORLD = 600, 650
    MEM[4096 + 1024] = {[0] = BATTLE}
    MEM[BATTLE] = {[L.b_world] = WORLD}
    MEM[WORLD] = {[L.w_battle] = BATTLE, [L.w_proj] = MGR}
    MEM[500] = {[L.i_unit] = 510}                               -- the script's unit object
    MEM[510] = {[L.u_ammo] = 77}                                 -- the battle unit
    CFG = {sites = {proj_update = SITE,
                    ammo_left = {hi = 0, lo = 512, sig = 'int32(pointer, pointer)', expect = '48 89 5c 24 08'},
                    battle_ref = {hi = 0, lo = 768, sig = '', expect = '48 8b 05'},
                    battle_global = {hi = 0, lo = 1024, sig = '', expect = ''}},
           layout = L, snapshot_max = 10, hook_max = 2}
"""


@pytest.fixture
def lua():
    runtime = new_runtime()
    runtime.globals().P = load(runtime, "apps.mem.projectiles")
    runtime.execute(FAKE)
    return runtime


def test_address_is_exact_above_the_float_limit(lua):
    # 0x2fedcb8 > 2^24 (a made-up address): split hi * 2^20 + lo (tools/mem_sites.split) adds back exactly
    assert lua.eval("P.address(mr, {hi = 0x2f, lo = 0xedcb8})") == 4096 + 0x2fedcb8


def test_checks_refuse_another_build_or_no_memreader(lua):
    assert lua.eval("P.check(nil, CFG).ok") is False
    assert lua.eval("P.check(mr, CFG).ok") is True
    lua.execute("CODE[4096 + 256] = string.char(0x48, 0x8b, 0x5c, 0x24)")
    st = lua.eval("P.check(mr, CFG)")
    assert st.ok is False and "differ" in st.sites.proj_update.why
    lua.execute("CODE[4096 + 256] = string.char(0x48, 0x89, 0x5c, 0x24); PATTERN_COUNT = 2")
    assert lua.eval("P.check(mr, CFG).ok") is False
    lua.execute("PATTERN_COUNT = 1")
    assert lua.eval("select(1, P.attach(mr, CFG)) ~= nil")


def test_a_projectile_spawns_flies_lands_and_goes(lua):
    lua.execute("""
        H = assert(P.attach(mr, CFG))
        local fn = H.poll
        local list = MEM[MGR][L.mgr_active]
        fn(MGR)                                              -- empty
        proj(1000, 3, 5, 0, 700, 800); list[1] = 1000
        fn(MGR)                                              -- seen: spawn
        MEM[1000][L.p_time] = 0.5; MEM[1000][L.p_pos] = 20
        fn(MGR)                                              -- flies on: no event
        D1 = H.drain()
        MEM[1000][L.p_state] = 5
        fn(MGR)                                              -- landed: a state event
        MEM[1000][L.p_state] = 7
        fn(MGR)                                              -- done
        list[1] = nil
        fn(MGR); fn(MGR); fn(MGR)                            -- removed: gone after three polls without it
        D2 = H.drain()
    """)
    d1, d2 = lua.globals().D1, lua.globals().D2
    assert d1.calls == 3 and len(d1.spawn) == 1
    s = d1.spawn[1]
    # {id, c, st, x, y, z, vx, vy, vz, t, rec, sx, sy, sz}
    assert list(s.values()) == [1, 2, 3, 5, 2, -5, 50, 1, 0, 0, 1, 10, 0, -80]
    assert d1.records[1].cat == 3
    assert len(d1.fly) == 1 and d1.fly[1][1] == 1 and d1.fly[1][3] == 20     # flying now, at x 20
    assert [list(r.values())[:3] for r in d2.state.values()] == [[1, 4, 5], [1, 5, 7]]
    assert [list(r.values()) for r in d2.gone.values()] == [[1, 6]]
    assert len(d2.fly) == 0


def test_a_reused_slot_is_a_new_projectile(lua):
    lua.execute("""
        H = assert(P.attach(mr, CFG))
        local fn = H.poll
        local list = MEM[MGR][L.mgr_active]
        proj(1000, 3, 5, 1.0, 700, 800); list[1] = 1000
        fn(MGR)
        MEM[1000][L.p_state] = 7
        fn(MGR)
        proj(1000, 3, 9, 0, 700, false)                    -- freed and reused between two ticks
        fn(MGR)
        D = H.drain()
    """)
    d = lua.globals().D
    assert len(d.spawn) == 2 and d.spawn[2][1] == 2 and d.spawn[2][12] is False   # no shooter: false
    assert [list(r.values()) for r in d.gone.values()] == [[1, 3]]


def test_an_error_in_the_scan_still_runs_the_game_function(lua):
    lua.execute("""
        H = assert(P.attach(mr, CFG))
        MEM[MGR][L.mgr_active] = {1234}                    -- a pointer to nothing
        H.poll()
        D = H.drain()
    """)
    assert len(lua.globals().D.errors) >= 1


def test_the_chain_must_lead_back_to_the_battle_else_the_unit_route(lua):
    lua.execute("""
        MEM[WORLD][L.w_battle] = 1                          -- the global's battle does not come back
        H = assert(P.attach(mr, CFG))
        proj(1000, 3, 5, 0, 700, 800); MEM[MGR][L.mgr_active] = {1000}
        H.poll()
        D1 = H.drain()
        BATTLE2, WORLD2 = 610, 660
        MEM[510][0x68] = BATTLE2
        MEM[BATTLE2] = {[L.b_world] = WORLD2}
        MEM[WORLD2] = {[L.w_battle] = BATTLE2, [L.w_proj] = MGR}
        PREFIX[4096 + 512]({}, 500, 0)                     -- the ammo hook names a battle unit
        H.poll()
        D2 = H.drain()
    """)
    g = lua.globals()
    assert len(g.D1.spawn) == 0 and g.D1.chain.g_back == "1"
    assert len(g.D2.spawn) == 1 and g.D2.chain.route == "unit+68"
    assert list(g.D2.discover.battle.values()) == [0x68]           # found by the scan of the unit


def test_the_hooks_pass_every_call_on_and_the_ammo_hook_reads_the_unit(lua):
    lua.execute("""
        H = assert(P.attach(mr, CFG))
        HOOKS[4096 + 256].fn(MGR)                           -- the update hook: counts, passes on
        for _ = 1, 3 do PREFIX[4096 + 512]({}, 500, 0) end  -- ammo_left x3 (a prefix); hook_max 2 reads
        D = H.drain()
        ST = H.stats()
        H.detach()
    """)
    g = lua.globals()
    assert len(g.NEXT) == 1                  # the plain hook passed its call on; a prefix never calls the game
    assert [list(r.values()) for r in g.D.ammo.values()] == [["1fe", 77], ["1fe", 77]]
    assert g.ST.ammo_calls == 3 and g.ST.update_callbacks == 1
    assert lua.eval("next(HOOKS)") is None and lua.eval("next(PREFIX)") is None   # both removed


def test_one_missed_poll_is_not_a_new_projectile(lua):
    # the poll races the simulation thread's removals: a projectile missing from one read is still the same one
    lua.execute("""
        H = assert(P.attach(mr, CFG))
        local list = MEM[MGR][L.mgr_active]
        proj(1000, 0, 5, 0, 700, 800); list[1] = 1000
        H.poll()
        list[1] = nil; H.poll()                              -- missed once
        MEM[1000][L.p_state] = 6; list[1] = 1000; H.poll()   -- back, lying now
        D = H.drain()
    """)
    d = lua.globals().D
    assert len(d.spawn) == 1 and len(d.gone) == 0 and list(d.state[1].values())[:3] == [1, 3, 6]


def test_tracked_projectile_leaves_its_last_place_and_spawn_hook_adds_a_field(lua):
    # S.track: every poll reads a flying projectile's place; its gone row carries the last one (where it stopped: the
    # game frees a done projectile within a tick). S.on_spawn's value is the spawn row's 15th field.
    lua.execute("""
        S = P.scanner(mr, L, 0)
        S.track = true
        S.on_spawn = function(p, st) if st == 2 then return {8, 42} end end
        local list = MEM[MGR][L.mgr_active]
        proj(1000, 0, 5, 0, 700, 800); list[1] = 1000
        proj(1100, 2, 7, 0, 700, 800); list[2] = 1100
        S.scan(MGR)
        MEM[1000][L.p_time] = 0.5; MEM[1000][L.p_pos] = 30
        S.scan(MGR)
        list[1] = nil; list[2] = nil
        S.scan(MGR); S.scan(MGR); S.scan(MGR)
        D = S.drain()
    """)
    d = lua.globals().D
    assert d.spawn[1][15] is None and list(d.spawn[2][15].values()) == [8, 42]
    gone = {r[1]: list(r.values()) for r in d.gone.values()}
    assert gone[1] == [1, 3, 30, 2, -5, 50, 1, 0, 0.5]       # the last place read (x 30 at flight time 0.5)
    assert gone[2][:3] == [2, 3, 7]


def test_untracked_gone_rows_stay_short(lua):
    lua.execute("""
        S = P.scanner(mr, L, 0)
        proj(1000, 0, 5, 0, 700, 800); MEM[MGR][L.mgr_active] = {1000}
        S.scan(MGR)
        MEM[MGR][L.mgr_active] = {}
        S.scan(MGR); S.scan(MGR); S.scan(MGR)
        D = S.drain()
    """)
    assert [list(r.values()) for r in lua.globals().D.gone.values()] == [[1, 2]]

"""Target recovery and empty-ammo ATTACK through the real bridge and engine-call spies."""
import pytest

from tests.entries.test_bridge_abilities import SETUP, SYG, FS, events, lua  # noqa: F401
from tools.nn.companion import exchange


def start(lua, tmp_path, kind="attack", ammo=1000, target="enemy_spear_1"):
    lua.globals().SYG, lua.globals().FS = SYG, FS
    lua.execute(SETUP + f"own[2].ammo, own[2].range = {ammo}, 120\n" + """
        enemy[2].pos:set_x(-60)
        enemy[2].melee = true
        STATE = require('entries.nn_arena').main(bm, CONFIG, GLOBALS)
        bm:pump()
    """)
    answer(lua, tmp_path, kind, target)


def answer(lua, tmp_path, kind, target="enemy_spear_1"):
    doc = exchange.read_state(tmp_path / exchange.STATE)
    exchange.write_atomic(tmp_path / exchange.ORDERS, exchange.orders_text(doc["batch"], doc["move"], [
        {"unit": "own_spear_1", "kind": kind, "target": target, "run": True}]))
    lua.execute("bm:tick(100)")


def advance(lua, seconds):
    lua.execute(f"for _ = 1, {seconds * 10} do bm:tick(100) end")


def log(lua):
    return list(lua.eval("bm.orders").values())


@pytest.mark.parametrize("kind", ["attack", "hold"])
def test_targetless_free_fire_reaims_within_three_seconds_and_keeps_firing(lua, tmp_path, kind):
    start(lua, tmp_path, kind)
    assert log(lua) == ["halt"]
    advance(lua, 2)
    assert not any(o.startswith("attack") for o in log(lua))
    advance(lua, 1)
    assert log(lua)[-1] == "attack enemy_spear_1"  # still in melee: bypass fire_freely
    args = lua.eval("own[2].attack_args")
    assert (args.primary, args.run) == (True, kind == "attack")
    assert lua.eval("own[2].melee_mode") is False and lua.eval("own[2].free_fire") is True
    recovery = [r for r in events(tmp_path / "tww3_bai_events.jsonl") if r["event"] == "nn_reaim"]
    assert len(recovery) == 1 and recovery[0]["t"] == 3000
    # Simulate the engine accepting the aim and firing; subsequent decisions must issue nothing.
    lua.execute("own[2].target = enemy[2]; fake.cco['uid_own_spear_1'] = {IsFiringMissiles = true}")
    before = log(lua)
    advance(lua, 12)
    assert log(lua) == before
    # A gap between volleys preserves on-target keep.
    lua.execute("fake.cco['uid_own_spear_1'].IsFiringMissiles = false")
    advance(lua, 2)
    answer(lua, tmp_path, kind)
    assert log(lua) == before
    # The engine subsequently drops its target: re-aim again, without waiting for melee to end.
    lua.execute("own[2].target = nil")
    advance(lua, 3)
    # Existing idle duty may release once while the targetless watchdog counts.
    assert log(lua)[-1] == "attack enemy_spear_1"
    assert [o for o in log(lua)[len(before):] if o.startswith("attack")] == ["attack enemy_spear_1"]
    lua.execute("bm.outcome, bm.winner = true, 2")
    advance(lua, 1)
    assert events(tmp_path / "tww3_bai_events.jsonl")[-1]["nn_reaims"] == 2


def test_recovery_uses_nearest_enemy_when_network_target_is_out_of_range(lua, tmp_path):
    start(lua, tmp_path, target="enemy_lord")
    assert log(lua) == ["attack enemy_lord"]
    advance(lua, 3)
    assert log(lua) == ["attack enemy_lord", "attack enemy_spear_1"]
    assert [r["tg"] for r in events(tmp_path / "tww3_bai_events.jsonl") if r["event"] == "nn_reaim"] == [
        "enemy_spear_1"]


def test_hold_walk_cooldown_does_not_leave_an_idle_shooter_targetless(lua, tmp_path):
    start(lua, tmp_path, "hold")
    advance(lua, 3)
    lua.execute("own[2].target = enemy[2]; own[2].moving = true")
    advance(lua, 2)
    assert log(lua)[-1] == "halt"
    lua.execute("own[2].target = nil")
    advance(lua, 3)
    assert log(lua)[-1] == "attack enemy_spear_1"
    assert len([r for r in events(tmp_path / "tww3_bai_events.jsonl") if r["event"] == "nn_reaim"]) == 2


@pytest.mark.parametrize("was_loaded", [False, True])
@pytest.mark.parametrize("reply", ["attack", "keep"])
def test_empty_shooter_attacks_in_melee_once_and_leaves_ranged_duty(lua, tmp_path, was_loaded, reply):
    start(lua, tmp_path, ammo=1000 if was_loaded else 0)
    if was_loaded:
        assert log(lua) == ["halt"]  # existing free-fire duty, unchanged network ATTACK
        lua.execute("own[2].ammo = 0")
        advance(lua, 1)
    assert log(lua)[-1] == "attack enemy_spear_1"
    args = lua.eval("own[2].attack_args")
    assert (args.primary, args.run) == (False, True)
    assert lua.eval("own[2].melee_mode") is True
    before = log(lua)
    advance(lua, 1)
    answer(lua, tmp_path, reply)
    advance(lua, 12)
    assert log(lua) == before  # no duty release/resume, no duplicate melee order while moving
    # Empty shooter participates in normal melee stall recovery, no longer ranged duty.
    lua.execute("own[2].moving = false")
    advance(lua, 3)
    assert log(lua) == before + ["attack enemy_spear_1"]
    lua.execute("bm.outcome, bm.winner = true, 2")
    advance(lua, 1)
    rows = events(tmp_path / "tww3_bai_events.jsonl")
    empty = [r for r in rows if r["event"] == "nn_empty_melee"]
    assert len(empty) == 2 and empty[0]["tg"] == "enemy_spear_1"
    assert not [r for r in rows if r["event"] == "nn_duty" and r["action"] in ("release", "resume")]
    assert rows[-1]["nn_empty_melees"] == 2 and rows[-1]["nn_stalls"] == 1

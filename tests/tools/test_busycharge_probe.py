"""The busycharge plan (tools/nn/charge_probe.py) and its table (tools/nn/busycharge_probe.py): numpy only."""
import numpy as np

from tools.nn import busycharge_probe as bp
from tools.nn import charge_probe as cp


def test_the_busycharge_plan_charges_busy_clanrats_from_front_flank_rear_and_free_ones():
    plan = cp.battles("busycharge")
    assert len(plan) == 3 and all(len(b) == 6 for b in plan)
    for b in plan:
        swords = sorted(l["kind"] for l in b if l["attacker"] == "swords")
        assert swords == ["busy0", "busy180", "busy90", "free0", "free180"]
        assert sum(l["attacker"] == "general" for l in b) == 1
        for l in b:
            assert l["target"] == "clanrat" and l["mode"] == "attack_run" and l["t_morale"] and l["contact_a_only"]
            assert l["a_go_s"] == cp.BUSYCHARGE["a_go_s"] and l["gap_m"] == 40
            busy = l["kind"].startswith("busy")
            assert bool(l.get("extras")) == busy and bool(l.get("t_attack_extra")) == busy
            if busy:
                (e,) = l["extras"]
                assert e["attack"] and e["b"] == 180 and e["dz"] > 0          # the anvil at the clanrats' front
    assert [l["kind"] for b in plan for l in b if l["attacker"] == "general"] == ["busy180", "free180", "busy90"]


def lane(hp, a_x, flag_t, kind="busy180"):
    """A recorded lane: samples every 0.5 s, the target's health hp, the charger at a_x (z 0), the target at 0."""
    t = np.arange(len(hp)) * 0.5
    spec = dict(cp.layout(cp.battles("busycharge")[0])[0][0], kind=kind, a_side=180)
    s = [(tt, {"x": float(ax), "z": 0.0, "k": 0}, {"x": 0.0, "z": 0.0, "hp": float(h), "mp": 1.0, "r": False}, None)
         for tt, h, ax in zip(t, hp, a_x)]
    return {"run": "r", "spec": spec, "samples": s, "contacts": {1: flag_t}, "men": [],
            "phases": [{"phase": "a_go", "t": 0}]}


def test_the_impact_is_the_first_big_drop_within_reach_before_a_late_melee_flag():
    # the anvil takes 20 HP a half second throughout; the charge lands at 5.0 s (200), the melee flag at 7.5 s
    n = 40
    hp = 9000 - 20 * np.arange(n, dtype=float)
    hp[11:] -= 200
    a_x = np.where(np.arange(n) < 10, 60.0 - 5 * np.arange(n), 12.0)        # within reach (half depths + 3) from 4.5 s
    ln = lane(hp, a_x, 7.5)
    assert bp.impact(ln) == 5.0
    row = bp.measure(ln)
    assert row["contact_s"] == 5.0 and abs(row["flag_lag_s"] - 2.5) < 1e-9
    assert abs(row["pre_rate"] - 40.0) < 1e-6 and abs(row["extra_0_2"] - 200.0) < 1e-6
    # a drop like it while the charger is still far is the anvil's, not the charge's
    far = lane(hp, np.full(n, 80.0), 7.5)
    assert bp.impact(far) == 7.5

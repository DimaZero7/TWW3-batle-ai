"""Drill U10 'the height in my area' (tools/nn/unit/drill_height.py, exam_height.py): the places with a rise in the
area, the enemy from above / below / across, the script's highest point, a short rollout, the mix (a map only). Needs
torch."""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import battle, terrain  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402
from tools.nn.unit import drill_height as DT  # noqa: E402
from tools.nn.unit import exam_height as EH  # noqa: E402
from tools.nn.unit import net as Nt  # noqa: E402
from tools.nn.unit import realmap  # noqa: E402
from tools.nn.unit import train as T  # noqa: E402

P = load()


def ramp(tmp_path, ax=0.2, az=0.06):
    """A 1 m height map h = ax x + az z + 200 over -512..512 (a plain slope), all clear."""
    g = np.arange(-512.0, 513.0, 1.0)
    Zg, Xg = np.meshgrid(g, g, indexing="ij")
    folder = tmp_path / "ramp"
    folder.mkdir(parents=True, exist_ok=True)
    H = (ax * Xg + az * Zg + 200).astype(np.float32)
    np.savez_compressed(folder / "heights.npz", height=H, clear=np.ones(H.shape, np.uint8),
                        ground=np.zeros(H.shape, np.uint8), x0=np.float64(-512), z0=np.float64(-512),
                        step=np.float64(1.0))
    realmap.activate("ramp", exam=True, root=tmp_path)


def test_the_specs_cycle_the_kinds_and_sides():
    sp = DT.exam_specs()
    assert len(sp) == 240 and [(s["kind"], s["frm"]) for s in sp[:6]] == [(k, f) for k in DT.KINDS for f in DT.FROM]


def test_the_places_put_a_rise_in_the_area_and_the_enemy_above_below_or_across(tmp_path):
    ramp(tmp_path)
    try:
        sp = DT.specs(12, torch.Generator().manual_seed(2), fixed=[(k, f) for k in DT.KINDS for f in DT.FROM])
        st, own, area, goal, places = DT.build(sp, P, "cpu")
        assert (places["relief"] >= DT.RELIEF_M).all()
        H = st.N // 2
        ca = terrain.height(st.u, area["x"][:, :1], area["z"][:, :1])[:, 0]
        fe = terrain.height(st.u, st.u["x"][:, H:H + 1], st.u["z"][:, H:H + 1])[:, 0]
        for b, s in enumerate(sp):
            d = float(fe[b] - ca[b])
            dist = math.hypot(float(st.u["x"][b, H] - area["x"][b, 0]), float(st.u["z"][b, H] - area["z"][b, 0]))
            grade = d / dist                                  # the ramp's grade along the enemy's way: ~+-0.21
            if s["frm"] == "up":
                assert grade > 0.16
            elif s["frm"] == "down":
                assert grade < -0.16
            else:
                assert abs(grade) < 0.06
        # the network's view reads the map through the place: our ground differs from the flat field's 0
        assert float(terrain.ground(st.u)[:, 0].abs().min()) > 100
    finally:
        realmap.deactivate()


def test_the_script_high_walks_to_the_area_s_highest_point(tmp_path):
    ramp(tmp_path)
    try:
        sp = DT.specs(4, torch.Generator().manual_seed(5), fixed=[("hold", "side"), ("shoot", "down")])
        setup = T.height_setup({"battles": 4}, P, "cpu", sp=sp)
        st, area = setup["st"], setup["area"]
        gx, gz = EH.high_point(st, area, P)
        h_goal = terrain.height(st.u, gx[:, None], gz[:, None])[:, 0]
        h_mid = terrain.height(st.u, area["x"][:, :1], area["z"][:, :1])[:, 0]
        assert bool((h_goal > h_mid + 1.0).all())
        h0 = terrain.ground(st.u)[:, 0].clone()
        for _ in range(30):
            o = O.merge(setup["script"](st), EH.script_orders(st, "high", (gx, gz)), setup["own"])
            battle.step(st, o, P)
        assert bool((terrain.ground(st.u)[:, 0] > h0 + 0.5).all())
    finally:
        realmap.deactivate()


def test_a_short_rollout_and_the_mix(tmp_path):
    assert T.MIX[-1] == "height" and len(T.MIX) == 12 and "height" in T.MAP_ONLY and "height" in T.SETUPS
    ramp(tmp_path)
    try:
        sp = DT.specs(3, torch.Generator().manual_seed(7))
        cfg = {"battles": 3, "seconds": 6, "cell_m": 8.0, "bounds": None}
        setup = T.height_setup(cfg, P, "cpu", sp=sp)
        torch.manual_seed(0)
        _, rs = T.rollout(Nt.UnitNet(), None, cfg, P, "cpu", T.Duty("cpu"), setup)
        assert rs["bans_broken"] == 0 and rs["nan"] == 0 and rs["terrain"] is not None
    finally:
        realmap.deactivate()
    with pytest.raises(AssertionError):
        DT.build(sp, P, "cpu")                                # a real map only

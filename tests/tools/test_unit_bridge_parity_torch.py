"""The unit network's input one to one between the simulator and the game (CLAUDE.md "Вход сети один в один";
build/diag_g49/bridge_parity.md): a simulator battle and the bridge's state documents of the same battle (what
nn_arena writes: x, z, b, men, hp, mp, ms, flags, a, fat, v, ow, map) give the same raw state through view.raw (the
simulator) and tools/nn/companion/unit_input.py (the game) - every column but the velocity and running, which view.py
still reads from the simulator's state (waits for the user) - and the same views on a map in the game's coordinates.
The simulator's field on a map (terrain.FRAMES, movement.clamp_to_map). Needs torch (the container)."""
import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn.companion import unit_input as UI  # noqa: E402
from tools.nn.sim import battle, movement, scenario, terrain  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402
from tools.nn.sim.state import FATIGUE_LEVELS  # noqa: E402
from tools.nn.train import opponents  # noqa: E402
from tools.nn.unit import train as T  # noqa: E402
from tools.nn.unit import view as V  # noqa: E402
from tools.nn.unit import zmap as Z  # noqa: E402

P = load()
SP = P.sim["formation"]["spacing_m"]
EMP = ["wh_main_emp_cha_general_0", "wh_main_emp_inf_spearmen_0", "wh_dlc04_emp_inf_flagellants_0",
       "wh3_dlc25_emp_inf_nuln_ironsides", "wh_main_emp_cav_demigryph_knights_0"]
SKV = ["wh2_main_skv_cha_warlord_0", "wh2_main_skv_inf_clanrat_spearmen_0", "wh2_dlc12_skv_inf_ratling_gun_0",
       "wh2_main_skv_inf_plague_monk_censer_bearer", "wh2_main_skv_art_warp_lightning_cannon"]
NOT_YET = ("vx", "vz", "f")          # view.py reads them from the simulator's state (bridge_parity.md: waits for the user)


def skirmish(gap=120.0):
    def line(keys, z, b):
        return [{"key": k, "x": -60.0 + 30.0 * i, "z": z, "b": b} for i, k in enumerate(keys)]
    return {"attacker": 1, "sides": {1: {"faction": "wh_main_emp_empire", "units": line(EMP, -gap / 2, 0.0)},
                                     2: {"faction": "wh2_main_skv_skaven", "units": line(SKV, gap / 2, 180.0)}}}


def doc_of_sim(st, b, t_ms, place=None, terrain_path="flat"):
    """The bridge's state document of battle b of a simulator state: every present unit's row in the game's
    coordinates (place (px, pz, deg): the battle laid on the map, terrain.set_map; None: as they are)."""
    u = st.u
    rows = []
    for i in range(st.N):
        if int(u["side"][b, i]) == 0:
            continue
        x, z, bear = float(u["x"][b, i]), float(u["z"][b, i]), float(u["b"][b, i])
        if place is not None:
            px, pz, deg = place
            a = math.radians(deg)
            x, z = x * math.cos(a) + z * math.sin(a) + px, -x * math.sin(a) + z * math.cos(a) + pz
            bear = (bear + deg) % 360
        rows.append({"n": f"u{i}", "side": int(u["side"][b, i]), "key": st.keys[b][i], "x": x, "z": z, "b": bear,
                     "men": float(u["men"][b, i]), "hp": float(u["hp"][b, i]), "mp": float(u["mp"][b, i]),
                     "ms": float(u["ms"][b, i]), "r": bool(u["r"][b, i]), "s": bool(u["s"][b, i]),
                     "w": bool(u["w"][b, i]), "m": bool(u["m"][b, i]), "mv": bool(u["mv"][b, i]),
                     "f": bool(u["f"][b, i]), "fire": bool(u["fire"][b, i]), "a": float(u["a"][b, i]),
                     "fat": FATIGUE_LEVELS[int(u["fat"][b, i])], "v": bool(u["vis"][b, i]),
                     "ow": float(u["width"][b, i])})
    return {"batch": "parity", "move": int(t_ms // 1000), "t": t_ms, "units": rows, "map": terrain_path}


def play(st, seconds, seer, place=None, terrain_path="flat"):
    sim, game = [], []
    for s in range(seconds):
        for _ in range(round(1.0 / P.dt)):
            battle.step(st, opponents.nearest(st), P)
        sim.append(V.raw(st, SP)[:1])
        game.append(seer.raw(doc_of_sim(st, 0, 1000 * (s + 1), place, terrain_path)))
    return torch.stack(sim), torch.stack(game)


def test_the_bridges_raw_state_is_the_simulators_but_the_velocity_on_a_flat_field():
    torch.manual_seed(3)
    st = scenario.build([skirmish()], P)
    sim, game = play(st, 30, UI.UnitSeer(allow_flat=True, params=P))
    same = [k for k in V.RAW if k not in NOT_YET]
    idx = [V.I[k] for k in same]
    d = (sim[..., idx] - game[..., idx]).abs().amax((0, 1, 2))
    assert float(d.max()) < 1e-3, {k: float(v) for k, v in zip(same, d) if v >= 1e-3}
    assert float(sim[..., V.I["m"]].max()) == 1.0                   # the battle went into melee
    # the velocity: the game's shift is zero for units standing or locked in melee, as the simulator's
    mv = sim[..., V.I["mv"]] > 0
    still = ~mv[1:] & ~mv[:-1]                                     # not moving in two snapshots running
    assert float(game[1:, ..., V.I["vx"]][still].abs().max()) == 0.0


def write_map(folder):
    """A 1 m map over -512..512 (h = 0.05 x + 200), its field -400..400 x -300..500 (meta.json frame), terrain path
    'testmap'."""
    g = np.arange(-512.0, 513.0, 1.0)
    Zg, Xg = np.meshgrid(g, g, indexing="ij")
    H = (0.05 * Xg + 200).astype(np.float32)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "heights.npz", height=H, clear=np.ones(H.shape, np.uint8),
                        x0=np.float64(-512), z0=np.float64(-512), step=np.float64(1.0))
    (folder / "meta.json").write_text(json.dumps({"map": "terrain\\battles\\testmap\\",
                                                  "frame": {"min_x": -400, "max_x": 400, "min_z": -300,
                                                            "max_z": 500}}), encoding="utf-8")
    return folder


def test_the_same_views_on_a_map_in_the_games_coordinates(tmp_path):
    folder = write_map(tmp_path / "testmap")
    assert UI.map_folder("terrain/battles/TESTMAP", tmp_path) == folder
    mid = terrain.load_npz(folder / "heights.npz")
    assert terrain.FRAMES[mid] == (-400.0, 400.0, -300.0, 500.0)
    torch.manual_seed(4)
    st = scenario.build([skirmish(gap=100.0)], P)
    place = (150.0, 200.0, 35.0)
    terrain.set_map(st, mid, place=tuple(torch.tensor([v]) for v in place))
    sim, game = play(st, 20, UI.UnitSeer(map_root=tmp_path, params=P), place, "terrain\\battles\\testmap\\")
    assert torch.allclose(sim[..., V.I["gh"]], game[..., V.I["gh"]], atol=1e-3)
    own = (st.u["side"][0] == 1).nonzero()[:, 0]
    R = own.shape[0]
    a = math.radians(place[2])
    ar = {"x": st.u["x"][:1].clone(), "z": st.u["z"][:1].clone(), "b": st.u["b"][:1].clone(),
          "front": torch.full((1, st.N), 80.0), "depth": torch.full((1, st.N), 80.0)}
    ag = dict(ar, x=ar["x"] * math.cos(a) + ar["z"] * math.sin(a) + place[0],
              z=-ar["x"] * math.sin(a) + ar["z"] * math.cos(a) + place[1], b=torch.remainder(ar["b"] + place[2], 360.0))
    card = T.cards(st, "cpu")[:1]
    cfg = {"cell_m": 8.0, "bounds": None}
    di, bi = torch.full((R,), 19, dtype=torch.long), torch.zeros(R, dtype=torch.long)
    vs = T.views(sim, di, bi, own, card, ar, cfg)
    vg = T.views(game, di, bi, own, card, ag, cfg)
    keep = [i for i, k in enumerate(V.ROW) if k not in ("v_right", "v_forward", "f")]
    assert torch.allclose(vs["rows"][..., keep], vg["rows"][..., keep], atol=2e-3)
    assert torch.allclose(vs["mem"][..., keep], vg["mem"][..., keep], atol=2e-3)
    assert torch.allclose(vs["map"], vg["map"], atol=2e-3)
    assert float(vs["map"][:, Z.L["height"]].abs().max()) > 0
    assert torch.equal(vs["cells_ok"], vg["cells_ok"]) and torch.equal(vs["target_ok"], vg["target_ok"])


def test_the_games_velocity_is_the_centres_shift():
    st = scenario.build([skirmish()], P)
    seer = UI.UnitSeer(allow_flat=True, params=P)
    seer.raw(doc_of_sim(st, 0, 1000))
    st.u["x"][0, 1] += 3.0
    st.u["mv"][0, 1] = True
    st.u["x"][0, 2] += 0.6                                         # a standing formation's drift: not moving
    r = seer.raw(doc_of_sim(st, 0, 2000))
    assert float(r[0, 1, V.I["vx"]]) == pytest.approx(3.0) and float(r[0, 1, V.I["f"]]) == 1.0
    assert float(r[0, 2, V.I["vx"]]) == 0.0
    st.u["m"][0, 1] = True
    seer.raw(doc_of_sim(st, 0, 3000))
    st.u["x"][0, 1] += 0.7
    r = seer.raw(doc_of_sim(st, 0, 4000))                           # in melee in both: still
    assert float(r[0, 1, V.I["vx"]]) == 0.0


def test_standing_units_stay_in_the_maps_field_and_routers_leave_past_it():
    g = np.zeros((101, 101), np.float32)
    mid = terrain.register(g, -500.0, -500.0, 10.0)
    terrain.set_frame(mid, -100.0, 100.0, -50.0, 150.0)
    st = scenario.build([skirmish()], P)
    terrain.set_map(st, mid, place=(torch.tensor([20.0]), torch.tensor([10.0]), torch.tensor([90.0])))
    cx, cz, b, front, depth = terrain.frame_local(st.u["terr"][:, 0], [st.u[k][:, 0] for k in terrain.PLACE])
    # the battle's (x, z) lies on the map at (x cos a + z sin a + px, -x sin a + z cos a + pz): the field's centre
    # (0, 50) of the map is the battle's (-40, -20); the map's bearing 0 the battle's -90
    assert float(cx[0]) == pytest.approx(-40.0, abs=1e-4) and float(cz[0]) == pytest.approx(-20.0, abs=1e-4)
    assert float(b[0]) == pytest.approx(270.0) and float(front[0]) == 200.0 and float(depth[0]) == 200.0
    st.u["x"][0, 1], st.u["z"][0, 1] = 500.0, 0.0                   # far beyond the field: held at its edge
    st.u["x"][0, 2], st.u["z"][0, 2] = 900.0, 0.0                   # a router past it but within +-1020 of the map: stays
    st.u["r"][0, 2] = True
    movement.clamp_to_map(st.u, P.map_half)
    xm, zm = terrain.to_map(st.u, st.u["x"][:, 1:2], st.u["z"][:, 1:2])
    assert -100.0 - 1e-3 <= float(xm[0, 0]) <= 100.0 + 1e-3 and -50.0 - 1e-3 <= float(zm[0, 0]) <= 150.0 + 1e-3
    assert not bool(st.u["gone"][0, 2])
    st.u["x"][0, 2] = 1100.0
    movement.clamp_to_map(st.u, P.map_half)
    assert bool(st.u["gone"][0, 2])
    # a flat battle: the square +-bounds of its own frame, as before
    st2 = scenario.build([skirmish()], P)
    st2.u["x"][0, 1] = 1500.0
    movement.clamp_to_map(st2.u, P.map_half)
    assert float(st2.u["x"][0, 1]) == P.map_half

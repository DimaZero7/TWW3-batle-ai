"""The tactician's input one to one between the simulator and the game (CLAUDE.md "Вход сети один в один";
build/diag_g49/bridge_parity.md, section "тактик"): a simulator battle and the bridge's state documents of the same
battle (what the bridge writes: src/entries/nn_arena.lua net_rows + apps/bridge fx, abilities_used) give the same raw
state (tools/nn/tactic/raw.py through tools/nn/companion/tactic_input.py) in every column but the known ones (vx, vz, f:
the unit level's view.raw reads them from the simulator; the enemy's button seconds, which its row never shows), and
the same input - rows, the three windows, the numbers - for the same parcels: on a flat field and on a captured map
(the game in the map's coordinates, the simulator in the battle's turned frame; forest, shallow water, trunks, 'cannot
stand', the field's edge). Needs torch (the container)."""
import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn.companion import tactic_input as TI  # noqa: E402
from tools.nn.model import abilities as MA  # noqa: E402
from tools.nn.sim import battle, scenario, terrain  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402
from tools.nn.sim.state import FATIGUE_LEVELS  # noqa: E402
from tools.nn.tactic import inputs as IN  # noqa: E402
from tools.nn.tactic import parcel as PC  # noqa: E402
from tools.nn.tactic import raw as TR  # noqa: E402
from tools.nn.tactic import rows as RW  # noqa: E402
from tools.nn.tactic import windows as W  # noqa: E402
from tools.nn.tactic import zone as ZN  # noqa: E402
from tools.nn.train import opponents  # noqa: E402
from tools.nn.unit import train as T  # noqa: E402

P = load()
EMP = ["wh_main_emp_cha_general_0", "wh_main_emp_inf_spearmen_0", "wh_dlc04_emp_inf_flagellants_0",
       "wh3_dlc25_emp_inf_nuln_ironsides", "wh_main_emp_cav_demigryph_knights_0"]
SKV = ["wh2_main_skv_cha_warlord_0", "wh2_main_skv_inf_clanrat_spearmen_0", "wh2_dlc12_skv_inf_ratling_gun_0",
       "wh2_main_skv_inf_plague_monk_censer_bearer", "wh2_main_skv_art_warp_lightning_cannon"]
NOT_YET = ("vx", "vz", "f")               # the unit level's view.raw reads them from the simulator (bridge_parity.md)
ROW_NOT_YET = ("v_right", "v_forward", "f")


def skirmish(gap=120.0):
    def line(keys, z, b):
        return [{"key": k, "x": -60.0 + 30.0 * i, "z": z, "b": b} for i, k in enumerate(keys)]
    return {"attacker": 1, "sides": {1: {"faction": "wh_main_emp_empire", "units": line(EMP, -gap / 2, 0.0)},
                                     2: {"faction": "wh2_main_skv_skaven", "units": line(SKV, gap / 2, 180.0)}}}


def to_game(x, z, place):
    if place is None:
        return x, z
    px, pz, deg = place
    a = math.radians(deg)
    return x * math.cos(a) + z * math.sin(a) + px, -x * math.sin(a) + z * math.cos(a) + pz


def doc_of_sim(st, b, t_ms, own=1, place=None, terrain_path="flat"):
    """The bridge's state document of battle b of a simulator state (what the bridge writes): every present unit's row
    in the game's coordinates (place: the battle laid on the map, terrain.set_map)."""
    u = st.u
    names = [f"u{i}" for i in range(st.N)]
    obs, _ = TR.observe_sim(st, P)
    passports = {**MA.load_spells(), **MA.load()}
    abk = sorted(P.abilities)
    rows, used = [], {}
    for i in range(st.N):
        if int(u["side"][b, i]) == 0:
            continue
        x, z = to_game(float(u["x"][b, i]), float(u["z"][b, i]), place)
        bear = (float(u["b"][b, i]) + (place[2] if place else 0.0)) % 360
        fx = []
        for name in TR.PHASE_KEY:
            if bool(obs["fx"][b, i, TR.EI[name]]):
                fx.append(sorted(TR.phases(name))[0])
        rx = int(obs["rx"][b, i])
        for r, k in enumerate(TR.ability_keys()):
            if rx >> r & 1:
                fx += list(passports[k].get("phases") or (k,))[:1]
        last = {}
        k = 0
        while f"ab{k}" in u:
            j = int(u[f"ab{k}"][b, i])
            k += 1
            if j < 0:
                continue
            ak = abk[j]
            p = passports[ak]
            if p.get("passive"):
                continue
            if float(u[f"ab{k - 1}_on"][b, i]) > 0:
                fx += list(p.get("phases") or (ak,))[:1]
            cd = float(u[f"ab{k - 1}_cd"][b, i])
            if cd > 0:
                last[ak] = t_ms - (p["active_s"] + p["recharge_s"] - cd) * 1000
        if last and int(u["side"][b, i]) == own:
            used[names[i]] = last
        rows.append({"n": names[i], "side": int(u["side"][b, i]), "key": st.keys[b][i], "x": x, "z": z, "b": bear,
                     "men": float(u["men"][b, i]), "hp": float(u["hp"][b, i]), "mp": float(u["mp"][b, i]),
                     "ms": float(u["ms"][b, i]), "r": bool(u["r"][b, i]), "s": bool(u["s"][b, i]),
                     "w": bool(u["w"][b, i]), "m": bool(u["m"][b, i]), "mv": bool(u["mv"][b, i]),
                     "f": bool(u["f"][b, i]), "fire": bool(u["fire"][b, i]), "a": float(u["a"][b, i]),
                     "fat": FATIGUE_LEVELS[int(u["fat"][b, i])], "v": bool(u["vis"][b, i]),
                     "ow": float(u["width"][b, i]), "fx": fx, "tr": bool(obs["fx"][b, i, TR.EI["terror"]]),
                     "hid": bool(obs["fx"][b, i, TR.EI["hidden"]])})
    return {"batch": "parity", "move": int(t_ms // 1000), "t": t_ms, "units": rows, "abilities_used": used,
            "map": terrain_path}


def run_both(st, seconds, seers, place=None, terrain_path="flat"):
    """([D, 1, N, RAW] the simulator's, {side: [D, 1, N, RAW]} the game's by each side's bridge: seers {side:
    TacticSeer(own=side)} - a side's own buttons are known to its own bridge only)."""
    sim_seer = TR.Seer(P)
    sim, game = [], {s: [] for s in seers}
    for s in range(seconds):
        for _ in range(round(1.0 / P.dt)):
            battle.step(st, opponents.nearest(st), P)
        sim.append(sim_seer.raw(st)[:1])
        for side, seer in seers.items():
            game[side].append(seer.raw(doc_of_sim(st, 0, 1000 * (s + 1), own=side, place=place,
                                                  terrain_path=terrain_path)))
    return torch.stack(sim), {k: torch.stack(v) for k, v in game.items()}


FRAME = ("x", "z", "b", "terr", "tpx", "tpz", "tpr")    # the battle's frame / the map's: compared through the input


def raw_parity(sim, game, st, frame=True):
    """The raw states agree in every column but NOT_YET (and FRAME when the two are in different frames: frame False);
    the buttons' seconds on the bridge's own units only (an enemy's: the game shows whether it is active, its row
    nothing more)."""
    side = st.u["side"][0]
    for own, g_all in game.items():
        for k in TR.RAW:
            if k in NOT_YET or (not frame and k in FRAME):
                continue
            i = TR.I[k]
            a, g = sim[..., i], g_all[..., i]
            if k.startswith(("a_on", "a_cd")):
                a, g = a[..., side == own], g[..., side == own]
            assert torch.allclose(a, g, atol=2e-3), (own, k, float((a - g).abs().max()))


def parcels(raw, n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = [PC.random(raw, g, group=k) for k in range(n)]
    return [p for p in out if p is not None]


def to_game_parcel(p, place):
    if place is None:
        return p
    x, z = to_game(float(p.cx), float(p.cz), place)
    return PC.Parcel.make(int(p.bi), p.mine[0], p.target[0], x, z, float(p.heading()[0]) + place[2],
                          bool(p.line[0]), int(p.group[0]))


def input_parity(sim, game, st, place=None, bounds=None):
    card = T.cards(st, "cpu")[:1]
    side = st.u["side"][0]
    for d in (5, len(sim) // 2, len(sim) - 1):
        rs = sim[d]
        for p in parcels(rs, seed=d):
            rg = game[int(side[p.mine[0]].max())][d]            # the group's side's own bridge
            q = to_game_parcel(p, place)
            a = IN.build(rs, card, p, ZN.Tiles.of(p), bounds=bounds)
            b = IN.build(rg, card, q, ZN.Tiles.of(q), bounds=bounds)
            assert torch.equal(a["slot"], b["slot"]) and torch.equal(a["who"], b["who"])
            keep = [i for i, n in enumerate(RW.ROW) if n not in ROW_NOT_YET]
            assert torch.allclose(a["rows"][..., keep], b["rows"][..., keep], atol=2e-3), \
                RW.ROW[keep[int((a["rows"][..., keep] - b["rows"][..., keep]).abs().amax((0, 1)).argmax())]]
            for win in ("near", "far", "target"):
                names = W.LAYERS[win]()
                dd = (a[win] - b[win]).abs().amax((0, 2, 3))
                assert float(dd.max()) < 2e-3, (win, names[int(dd.argmax())], float(dd.max()))
                for k in a["cover"]:
                    assert torch.allclose(a["cover"][k], b["cover"][k], atol=2e-3), k
            assert torch.allclose(a["numbers"], b["numbers"], atol=2e-3)
            yield a


def test_the_bridge_gives_the_simulators_input_on_a_flat_field():
    torch.manual_seed(3)
    st = scenario.build([skirmish()], P)
    seers = {k: TI.TacticSeer(allow_flat=True, params=P, own=k) for k in (1, 2)}
    sim, game = run_both(st, 40, seers)
    raw_parity(sim, game, st)
    seen = list(input_parity(sim, game, st, bounds=st.bounds))
    assert len(seen) >= 6
    # the battle went somewhere: melee, effects, the buttons are in the rows
    rows = torch.cat([a["rows"] for a in seen], 1)
    ix = RW.RI
    assert float(rows[..., ix["m"]].max()) == 1.0
    assert float(sim[..., TR.I["e_hold_the_line"]].max()) > 0 and float(sim[..., TR.I["e_lord_near"]].max()) > 0
    assert float(sim[..., TR.I["e_braced"]].max()) > 0 and float(sim[..., TR.I["a_has"]].max()) > 0


def write_map(folder):
    """A 1 m map over -512..512 (h = 0.05 x + 200), forest where z > 380, shallow water where x < 100 and z < 330,
    'cannot stand' 200..230 x 350..380, trunks every 4 m over 120..200 x 400..460, its field -400..400 x -300..500 -
    the companion finds it by its terrain path ('testmap')."""
    g = np.arange(-512.0, 513.0, 1.0)
    Zg, Xg = np.meshgrid(g, g, indexing="ij")
    H = (0.05 * Xg + 200).astype(np.float32)
    G = np.where(Zg > 380, 0, np.where((Xg < 100) & (Zg < 330), 2, 1)).astype(np.uint8)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "heights.npz", height=H, clear=np.ones(H.shape, np.uint8), ground=G,
                        x0=np.float64(-512), z0=np.float64(-512), step=np.float64(1.0))
    blk = ((Xg >= 200) & (Xg <= 230) & (Zg >= 350) & (Zg <= 380)).astype(np.uint8)
    np.savez_compressed(folder / "passable.npz", blocked=blk, x0=np.float64(-512), z0=np.float64(-512),
                        step=np.float64(1.0))
    tx, tz = np.meshgrid(np.arange(120.0, 200.0, 4.0), np.arange(400.0, 460.0, 4.0))
    tx, tz = tx.ravel(), tz.ravel()
    np.savez_compressed(folder / "trees.npz", x=tx, z=tz, y=(0.05 * tx + 200), r=np.full(tx.shape, 0.3),
                        h=np.full(tx.shape, 8.0))
    (folder / "meta.json").write_text(json.dumps({"map": "testmap", "ground_names": ["forest", "grass", "shallow_water"],
                                                  "frame": {"min_x": -400, "max_x": 400, "min_z": -300,
                                                            "max_z": 500}}), encoding="utf-8")
    return folder


def test_the_bridge_gives_the_simulators_input_on_a_map_in_the_games_coordinates(tmp_path):
    folder = write_map(tmp_path / "testmap")
    mid = terrain.load_npz(folder / "heights.npz")
    torch.manual_seed(4)
    st = scenario.build([skirmish(gap=100.0)], P)
    place = (150.0, 380.0, 35.0)            # near the field's top edge (z 500), half in the forest (z > 380)
    terrain.set_map(st, mid, place=tuple(torch.tensor([v]) for v in place))
    seers = {k: TI.TacticSeer(map_root=tmp_path, params=P, own=k) for k in (1, 2)}
    sim, game = run_both(st, 25, seers, place=place, terrain_path="testmap")
    raw_parity(sim, game, st, frame=False)
    seen = list(input_parity(sim, game, st, place=place))
    assert seen
    names = W.near_layers()
    near = torch.cat([a["near"] for a in seen])
    for n in ("forest", "height", "field"):
        assert float(near[:, names.index(n)].abs().max()) > 0, n
    assert float(near[:, names.index("field")].min()) < 0.5       # the field's edge in sight
    assert float(near[:, names.index("forest")].min()) == 0.0
    far = torch.cat([a["far"] for a in seen])
    fl = W.far_layers()
    lit = {n: float(torch.cat([near[:, names.index(n)].flatten(), far[:, fl.index(n)].flatten()]).max())
           for n in ("trunks", "blocked", "shallow")}
    assert all(v > 0 for v in lit.values()), lit

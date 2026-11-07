"""tools/nn/arena_map.py: a battle on a captured map takes the map's zones (size, the gap between their fronts, the
middle as the origin), its units narrowed into the zone's front; the battle file and the config are moved by the origin."""
import json

from tools.nn import arena_map as AM


def test_a_battle_on_a_captured_map(tmp_path):
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "meta.json").write_text(json.dumps({"map": "terrain/battles/m/", "deployment_zones": [
        {"x": 120.0, "z": -190.0, "width": 200.0, "depth": 120.0},
        {"x": -120.0, "z": -190.0, "width": 200.0, "depth": 120.0}]}), encoding="utf-8")
    arena = {"name": "a", "gap_m": 350, "catchment": "c", "sides": {
        "own": {"units": [{"slot": "x", "forward": -70.0, "lateral": 126.0}, {"slot": "y", "forward": -300.0, "lateral": -10.0}]},
        "enemy": {"units": [{"slot": "z", "forward": 0.0, "lateral": 0.0}]}}}
    a = AM.on_map(arena, "m", maps=tmp_path)
    assert a["map"] == "terrain/battles/m/" and a["catchment"] == AM.CATCHMENT
    assert a["deploy_zone"] == {"front": 200.0, "depth": 120.0, "gap": 120.0} and a["gap_m"] == 120.0
    assert a["origin"] == {"x": 0.0, "z": -190.0}
    u = a["sides"]["own"]["units"]
    assert u[0]["lateral"] == 85.0 and u[0]["forward"] == -70.0       # narrowed into 200 / 2 - 15
    assert u[1]["forward"] == -105.0                                    # kept inside the zone's depth
    text = ('<centre x="-120" y="0"/>\n<position x="-60" y="85.5"/>\n'
            '<camera_target_position x="0" y="577" z="0"/>')
    out = AM.shift_xml_text(text, a["origin"])
    assert '<centre x="-120" y="-190"/>' in out and '<position x="-60" y="-104.5"/>' in out
    assert '<camera_target_position x="0" y="577" z="-190"/>' in out
    rc = AM.shift_config({"units": {"own": [{"x": -60.0, "z": 85.0}]}, "deploy_zones": {"own": {"x": -120.0, "z": 0.0}}},
                         a["origin"])
    assert rc["units"]["own"][0]["z"] == -105.0 and rc["deploy_zones"]["own"]["z"] == -190.0


def test_the_games_own_zones_of_a_map_without_its_own():
    """terrain/deployment/east.xml (the map's 0..1024 frame): two areas along x, 416 m across, 176 m deep, fronts 192 m
    apart - the default of on_map when meta.json has them (game_deployment_zones), the capture's kept beside."""
    def area(aid, x0, x1, z0, z1):
        pts = "".join(f"<position x='{x}' y='{z}'/>" for x, z in ((x0, z0), (x1, z0), (x1, z1), (x0, z1)))
        return f"<ALLIANCE id='{aid}'><deployment_area><boundary>{pts}</boundary></deployment_area></ALLIANCE>"
    xml = "<BATTLE_DEPLOYMENT_AREAS>" + area(0, 240, 416, 304, 720) + area(1, 608, 784, 304, 720) + "</BATTLE_DEPLOYMENT_AREAS>"
    zs = AM.game_zones(xml)
    a, b = zs
    assert (a["x"], a["z"], a["width"], a["depth"], a["bearing_deg"]) == (-184.0, 0.0, 416.0, 176.0, 90.0)
    assert (b["x"], b["bearing_deg"]) == (184.0, 270.0)
    meta = {"map": "terrain/battles/m/", "deployment_zones": [{"x": 120.0, "z": -190.0, "width": 200.0, "depth": 120.0},
                                                            {"x": -120.0, "z": -190.0, "width": 200.0, "depth": 120.0}],
            "game_deployment_zones": zs}
    import json
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "m").mkdir()
        (Path(d) / "m" / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        arena = {"name": "a", "gap_m": 350, "sides": {"own": {"units": [{"forward": -69.0, "lateral": 126.0}]},
                                                      "enemy": {"units": [{"forward": 0.0, "lateral": 0.0}]}}}
        g = AM.on_map(arena, "m", maps=d)
        c = AM.on_map(arena, "m", maps=d, zones="capture")
    assert g["deploy_zone"] == {"front": 416.0, "depth": 176.0, "gap": 192.0} and g["origin"] == {"x": 0.0, "z": 0.0}
    assert g["sides"]["own"]["units"][0]["lateral"] == 126.0                 # fits: no narrowing
    assert c["deploy_zone"]["front"] == 200.0

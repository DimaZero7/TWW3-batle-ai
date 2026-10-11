"""Domain randomisation of the simulator's uncertain numbers (tools/nn/train/dr.py, tools/nn/sim/knobs.py,
config/nn/dr.json), the config without torch: every knob is known, every range holds its sim.json centre, every per-unit
centre is a value some unit has. The draws and the simulator: test_nn_dr_torch.py (in the container)."""
import pytest

from tools.nn.sim import knobs
from tools.nn.sim.params import load
from tools.nn.train import dr

P = load()


def test_the_config_names_known_knobs_and_every_range_holds_its_sim_json_centre():
    cfg = dr.load(params=P)
    names = {s.name for s in cfg.scalars}
    assert names and names <= set(knobs.SCALARS)
    for s in cfg.scalars:
        assert s.lo <= s.centre <= s.hi and s.lo < s.hi
        assert s.centre == knobs.centre(P.sim, s.name)
    assert knobs.centre(P.sim, "pursuit_rate") == P.sim["contact"]["pursuit_rate"]
    assert knobs.centre(P.sim, "rally_hazard_scale") == 1.0
    assert "domain randomisation" in cfg.text()


def test_every_per_unit_centre_is_a_value_some_unit_has():
    cfg = dr.load(params=P)
    assert cfg.fields
    values = {}
    for key in P.units:
        st = P.static(key)
        for f in cfg.fields:
            values.setdefault(f.field, set()).add(round(float(st[f.field]), 3))
    for f in cfg.fields:
        for c, lo, hi in f.entries:
            assert lo <= c <= hi and lo < hi
            assert round(c, 3) in values[f.field], (f.field, c)


def test_a_range_without_its_centre_and_an_unknown_knob_are_refused(tmp_path):
    p = tmp_path / "dr.json"
    p.write_text('{"scalars": {"hold_rate": {"lo": 0.6, "hi": 0.7}}}', encoding="utf-8")
    with pytest.raises(ValueError):
        dr.load(p, P)
    p.write_text('{"scalars": {"no_such": {"lo": 0, "hi": 1}}}', encoding="utf-8")
    with pytest.raises(ValueError):
        dr.load(p, P)
    p.write_text('{"scalars": {"hold_rate": {"lo": 0.6, "hi": 0.7, "off": true}}}', encoding="utf-8")
    assert dr.load(p, P).scalars == ()

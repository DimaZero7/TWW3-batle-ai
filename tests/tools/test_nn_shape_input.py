"""The unit's shape in the network's input (observation SHAPE, 17.10.2026): the state's shape / shape_ok become the
token's last columns for own units and seen enemies, zeros and shape_known 0 without them; older checkpoints load with
zero weights for them (encoder.py) and compute what they did."""
import numpy as np
import pytest

from tools.nn.model import observation as ob
from tests.tools.test_nn_spells import caster_battle

F = len(ob.SHAPE_FEATURES) + ob.SHAPE_GRID
S0 = ob.INDEX["shape_known"]                      # the shape's first column (FX_STATS, FIGHT after it)
S1 = S0 + len(ob.SHAPE)


def shape_state(**kw):
    setup, state = caster_battle(**kw)
    v = np.zeros((1, 4, F), np.float32)
    v[0, :, 0] = [0.3, 0.4, 0.5, 0.6]                  # men per m²
    v[0, :, 8] = [0.0, 0.1, 0.2, 0.25]                 # wounded
    v[0, :, 9:] = 1 / 16
    return setup, dict(state, shape=v, shape_ok=np.array([[True, True, True, False]]))


def test_the_shape_columns_are_the_tokens_last():
    assert ob.NAMES[S0:S1] == ob.SHAPE and ob.NAMES[S1:] == ob.FX_STATS + ob.FIGHT + ob.FX_NEW + ob.FX_STATS2 + ob.SIGHT + ob.FX_LAST and len(ob.SHAPE) == 1 + F
    assert ob.SHAPE_FEATURES == tuple(__import__("tools.nn.fighters.shape", fromlist=["x"]).FEATURES)
    assert len(ob.SHAPE_SCALE) == F


def test_own_and_seen_enemies_show_their_shape_scaled():
    setup, state = shape_state()
    obs, _ = ob.observe(state, setup, 1)
    tok = obs.tokens[0]
    assert tok[:, ob.INDEX["shape_known"]].tolist() == [1, 1, 1, 0]       # the clanrats unread: not known
    assert tok[:, ob.INDEX["shape_men_m2"]].tolist() == pytest.approx([0.6, 0.8, 1.0, 0.0])
    assert tok[2, ob.INDEX["shape_wounded"]] == pytest.approx(0.2)        # an enemy's wounds: from its card
    assert tok[0, ob.INDEX["shape_cell0"]] == pytest.approx(0.25)
    hidden, _ = ob.observe(dict(state, vis=np.array([[True, True, False, True]])), setup, 1)
    assert hidden.tokens[0, 2, ob.INDEX["shape_known"]] == 0             # the Warlord not seen: nothing
    assert not hidden.tokens[0, 2, S0:S1].any()
    assert hidden.tokens[0, 0, ob.INDEX["shape_known"]] == 1             # own units always


def test_no_shape_in_the_state_gives_zeros():
    setup, state = caster_battle()
    obs, _ = ob.observe(state, setup, 1)
    assert not obs.tokens[0, :, S0:S1].any()
    cobs, _ = ob.observe(shape_state()[1], setup, 1, full=True)          # the critic sees every unit's
    assert cobs.tokens[0, :3, ob.INDEX["shape_known"]].tolist() == [1, 1, 1]


def test_an_older_checkpoint_loads_with_zero_weights_and_computes_what_it_did():
    torch = pytest.importorskip("torch")
    from tools.nn.model import policy
    from tools.nn.model.config import preset
    cfg = preset("v2", d=32, heads=2, pointer=16, critic_d=32, critic_heads=2, sector_d=16, eyes=False)
    torch.manual_seed(0)
    sd = policy.Actor(cfg).state_dict()
    old = {k: (v[:, :S0].clone() if k == "encoder.unit.0.weight" else v.clone())
           for k, v in sd.items()}
    loaded = policy.Actor(cfg)
    loaded.load_state_dict(old)
    w = loaded.state_dict()["encoder.unit.0.weight"]
    assert torch.all(w[:, ob.INDEX["shape_known"]:] == 0)
    assert torch.equal(w[:, :S0], old["encoder.unit.0.weight"])
    tokens = torch.randn(2, 5, ob.TOKEN)
    ctx = torch.randn(2, ob.CONTEXT)
    with_shape = loaded.encoder(tokens, ctx)
    tokens[..., S0:] = 0
    assert torch.equal(loaded.encoder(tokens, ctx), with_shape)          # whatever the shape: the same output

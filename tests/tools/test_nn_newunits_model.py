"""The network's side of the new units (11.10.2026): the war beasts' caste, the WEAPON columns, the contact effects in
the effect pairs (owned: the unit's weapon lays it; on: the unit is under it), the companion reading them from the
unit card, and older checkpoints loading with the token's new columns at zero weights."""
import numpy as np
import pytest

from tools.nn.model import effects as mfx
from tools.nn.model import observation as ob
from tools.nn.model import passport

RATS, POISON_RATS, CENSER, GUN = ("wh2_dlc16_skv_mon_wolf_rats_0", "wh2_dlc16_skv_mon_wolf_rats_1",
                                  "wh2_main_skv_inf_plague_monk_censer_bearer", "wh2_dlc12_skv_inf_ratling_gun_0")
SWORD = "wh_main_emp_inf_swordsmen"
POISON, CONTAGION, SUPPRESS = ("wh_main_unit_contact_poison", "wh2_main_unit_contact_plague_monk_censer_morale",
                               "wh2_dlc12_unit_contact_suppressive_fire")


def test_war_beasts_have_their_caste_column():
    f = passport.features(passport.load()[RATS])
    at = len(f) - len(passport.FLIGHT) - len(passport.ATTRIBUTES) - (len(passport.SIZES) + 1) - (len(passport.CASTES) + 1)
    caste = f[at:at + len(passport.CASTES) + 1]
    assert caste[passport.CASTES.index("war_beast")] == 1.0 and caste[-1] == 0.0      # not "other" any more


def test_the_weapon_columns():
    w = passport.weapon([GUN, CENSER, POISON_RATS, SWORD, "wh_main_emp_inf_handgunners", ""])
    assert w[:, 0].tolist() == pytest.approx([0.9, 0, 0, 0, 0.05, 0])
    assert w[:, 1].tolist() == [0, 1, 0, 0, 0, 0]
    T = len(ob.AIM) + len(ob.RECEIVED) + len(ob.MAGIC) + len(ob.SHAPE)  # the groups after WEAPON
    assert ob.NAMES[-len(ob.WEAPON) - T:-T] == ob.WEAPON


def test_contact_effects_are_owned_by_who_lays_them_and_on_where_the_bitmask_says():
    keys = mfx.keys()
    e = keys.index(CONTAGION)
    own = mfx.owned([CENSER, SWORD])
    assert own[0, e] == 1 and own[1, e] == 0
    assert mfx.contact()[[keys.index(k) for k in (POISON, CONTAGION, SUPPRESS)]].all()
    state = {"fx_on": np.array([[0, 1 << e]]), "hp": np.ones((1, 2)), "ms": np.full((1, 2), 2.0),
             "m": np.zeros((1, 2)), "mp": np.ones((1, 2))}
    on = mfx.active(np, state, own[None])
    assert on[0, 1, e] and not on[0, 0, e]                 # the swordsmen are under it though they do not own it
    del state["fx_on"]
    assert not mfx.active(np, state, own[None])[..., e].any()   # without a bitmask: not known, off


def test_the_companion_reads_contact_effects_from_the_card():
    from tools.nn.companion import exchange
    keys = mfx.keys()

    class Setup:
        fx_owned = mfx.owned([SWORD, CENSER])[None]
    state = {"hp": np.ones((1, 2)), "ms": np.full((1, 2), 2.0), "m": np.ones((1, 2)), "mp": np.ones((1, 2))}
    doc = {"units": [{"n": "a", "fx": [CONTAGION, "wh_main_unit_passive_frenzy"]}, {"n": "b", "fx": []}]}
    exchange.effects_on(state, doc, ["a", "b"], Setup())
    bits = [int(b) for b in state["fx_on"][0]]
    assert bits[0] >> keys.index(CONTAGION) & 1 and not bits[1] >> keys.index(CONTAGION) & 1


def test_a_token_from_before_the_new_units_maps_to_its_columns():
    E = mfx.SIZE // 2
    width = ob.TOKEN - 2 * (E - 14) - len(ob.WEAPON) - len(ob.AIM) - len(ob.RECEIVED) - len(ob.MAGIC) - len(ob.SHAPE)
    cols = ob.old_columns(width)
    assert cols is not None and len(cols) == width and len(set(cols)) == width
    names = [ob.NAMES[c] for c in cols]
    assert names[-len(ob.ORDER):] == list(ob.ORDER) and ob.NAMES[cols[-len(ob.ORDER) - 2]] == "volley_ready"
    assert names[: width - 7] == list(ob.NAMES[: width - 7])        # the passport and the 14 old pairs stay
    assert ob.old_columns(ob.TOKEN) is None and ob.old_columns(width - 1) is None


def test_an_old_checkpoint_computes_what_it_did():
    torch = pytest.importorskip("torch")
    from tools.nn.model.encoder import TokenEncoder
    E = mfx.SIZE // 2
    width = ob.TOKEN - 2 * (E - 14) - len(ob.WEAPON) - len(ob.AIM) - len(ob.RECEIVED) - len(ob.MAGIC) - len(ob.SHAPE)
    torch.manual_seed(0)
    old = TokenEncoder(16, n_token=width)
    new = TokenEncoder(16)
    new.load_state_dict(old.state_dict())
    t_old = torch.randn(2, 5, width)
    t_new = torch.zeros(2, 5, ob.TOKEN)
    t_new[..., ob.old_columns(width)] = t_old
    t_new[..., -len(ob.WEAPON) - len(ob.AIM) - len(ob.RECEIVED):] = 1.0                 # the new inputs, at zero weights
    ctx = torch.randn(2, ob.CONTEXT)
    assert torch.allclose(old(t_old, ctx), new(t_new, ctx), atol=1e-6)

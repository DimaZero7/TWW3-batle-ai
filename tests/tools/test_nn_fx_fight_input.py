"""The token's FX_STATS and FIGHT (observation.py, 23.10.2026): what the effects on a unit do as numbers (effects.stats_on,
the catalogue's stats summed as RECEIVED), under fire (a living enemy not in melee shoots at it) and the current melee's
seconds (Memory.melee_t0); own units and seen enemies only; older checkpoints load with zero weights for them."""
import numpy as np
import pytest

from tools.nn.model import abilities, effects
from tools.nn.model import observation as ob
from tools.nn.model import sources

SWORD, GUNS = "wh_main_emp_inf_swordsmen", "wh_main_emp_inf_handgunners"
MONKS, RATS = "wh2_main_skv_inf_plague_monks", "wh2_dlc16_skv_mon_wolf_rats_1"
FX = {k: i for i, k in enumerate(effects.keys())}


def battle(**kw):
    """Side 1: swordsmen, handgunners; side 2: plague monks, poison wolf rats. Nobody fights, nothing on."""
    keys = [SWORD, GUNS, MONKS, RATS]
    setup = ob.Setup(keys=[keys], side=np.array([[1, 1, 2, 2]]), bounds=np.array([sources.CROSSROADS], np.float32),
                     factions=[(sources.EMPIRE, sources.SKAVEN)], attacker=np.array([1]))
    _, state = sources.synthetic(batch=1, own=2, enemy=2, seed=3)
    state = dict(state, x=np.array([[0.0, -20.0, 60.0, 80.0]]), z=np.zeros((1, 4)), vis=np.ones((1, 4), bool),
                 hp=np.ones((1, 4)), ms=np.full((1, 4), 2.0), m=np.zeros((1, 4), bool), mp=np.ones((1, 4)),
                 men=np.array([[100.0, 100.0, 100.0, 12.0]]), fire=np.zeros((1, 4), bool),
                 target=np.full((1, 4), -1), fx_on=np.zeros((1, 4), np.int64), t=np.array([10.0]))
    state["w"], state["r"], state["s"] = state["ms"] == 5, state["ms"] >= 6, state["ms"] == 7
    state.update(kw)
    return setup, state


def expected(*effect_keys):
    """The FX_STATS of the effects on, worked out by hand from the catalogue: multipliers multiply, additions add."""
    doc = effects.load()
    scale = {(s, h): k for s, h, k in abilities.STATS}
    mult = {sh: 1.0 for sh in effects.FX_STATS if sh[1] == "mult"}
    add = {sh: 0.0 for sh in effects.FX_STATS if sh[1] == "add"}
    for k in effect_keys:
        for e in doc["effects"][k]["stats"]:
            sh = (e["stat"], e["how"])
            if sh in mult:
                mult[sh] *= e["value"]
            else:
                add[sh] += e["value"] / scale[sh]
    return [mult[sh] - 1 if sh in mult else add[sh] for sh in effects.FX_STATS]


def bits(*names):
    return sum(1 << FX[n] for n in names)


def test_the_columns_are_the_tokens_last():
    n = len(ob.FX_STATS) + len(ob.FIGHT)
    assert ob.NAMES[-n:] == ob.FX_STATS + ob.FIGHT and ob.TAIL[-2:] == (ob.FX_STATS, ob.FIGHT)
    assert len(ob.FX_STATS) == 10 and ob.FIGHT == ("under_fire", "melee_s")


def test_every_stat_the_catalogue_lays_on_a_unit_is_a_column():
    """A new stat of an effect (on self or hit) must be appended to effects.FX_STATS (else it is silently dropped)."""
    doc = effects.load()
    used = {(e["stat"], e["how"]) for k in effects.keys(doc) for e in (doc["effects"].get(k) or {}).get("stats") or ()
            if set(e.get("on") or ()) & {"self", "hit"}}
    assert used <= set(effects.FX_STATS)
    assert {(s, h) for s, h in effects.FX_STATS} <= {(s, h) for s, h, _ in abilities.STATS}


def test_the_effects_on_are_numbers():
    poison = "wh_main_unit_contact_poison"
    frenzy, sin = "wh_main_unit_passive_frenzy", "wh2_main_unit_passive_strength_in_numbers"
    setup, state = battle(fx_on=np.array([[bits(poison), 0, bits(frenzy, sin), 0]]))
    obs, _ = ob.observe(state, setup, 1)
    tok = obs.tokens[0]
    cols = [ob.INDEX[n] for n in ob.FX_STATS]
    assert tok[0, cols].tolist() == pytest.approx(expected(poison), abs=1e-6)        # poisoned: -15 % speed, damage
    assert tok[0, ob.INDEX["fxs_scalar_speed_mult"]] == pytest.approx(-0.15, abs=1e-6)
    assert tok[2, cols].tolist() == pytest.approx(expected(frenzy, sin), abs=1e-6)   # a seen enemy's
    assert tok[2, ob.INDEX["fxs_stat_melee_defence_add"]] == pytest.approx(8 / 50, abs=1e-6)
    assert not tok[1, cols].any() and not tok[3, cols].any()                         # nothing on
    hidden, _ = ob.observe(dict(state, vis=np.array([[True, True, False, True]])), setup, 1)
    assert not hidden.tokens[0, 2, cols].any()                                       # not seen: nothing
    cobs, _ = ob.observe(dict(state, vis=np.array([[True, True, False, True]])), setup, 1, full=True)
    assert cobs.tokens[0, 2, cols].tolist() == pytest.approx(expected(frenzy, sin), abs=1e-6)   # the critic: all


def test_an_effect_not_owned_is_not_on():
    """Frenzy's bit on the swordsmen (they do not own it): effects.active masks it - no numbers."""
    setup, state = battle(fx_on=np.array([[bits("wh_main_unit_passive_frenzy"), 0, 0, 0]]))
    obs, _ = ob.observe(state, setup, 1)
    assert not obs.tokens[0, 0, [ob.INDEX[n] for n in ob.FX_STATS]].any()


def test_under_fire_is_a_standing_enemy_shooting_at_the_unit():
    U = ob.INDEX["under_fire"]
    # the handgunners shoot the monks; the rats shoot the swordsmen
    setup, state = battle(fire=np.array([[False, True, False, True]]), target=np.array([[-1, 2, -1, 0]]))
    obs, _ = ob.observe(state, setup, 1)
    assert obs.tokens[0, :, U].tolist() == [1, 0, 1, 0]
    melee = np.array([[False, False, False, True]])                                   # a shooter in melee: no fire
    obs, _ = ob.observe(dict(state, m=melee), setup, 1)
    assert obs.tokens[0, :, U].tolist() == [0, 0, 1, 0]
    dead = np.array([[100.0, 0.0, 100.0, 12.0]])                                      # a dead shooter: no fire
    obs, _ = ob.observe(dict(state, men=dead), setup, 1)
    assert obs.tokens[0, :, U].tolist() == [1, 0, 0, 0]
    friendly = np.array([[-1, 0, -1, 0]])                                             # a friend as the target: no
    obs, _ = ob.observe(dict(state, target=friendly), setup, 1)
    assert obs.tokens[0, :, U].tolist() == [1, 0, 0, 0]
    obs, _ = ob.observe(dict(state, vis=np.array([[True, True, False, True]])), setup, 1)
    assert obs.tokens[0, :, U].tolist() == [1, 0, 0, 0]                              # a hidden enemy: not shown
    obs, _ = ob.observe({k: v for k, v in state.items() if k != "target"}, setup, 1)
    assert not obs.tokens[0, :, U].any()                                              # no target: not under fire


def test_the_melee_seconds_count_from_the_first_observation_in_melee():
    S = ob.INDEX["melee_s"]
    setup, state = battle()
    mem, got = None, []
    # (t, swordsmen in melee, monks in melee, monks seen)
    for t, a, b, seen in ((0, 0, 0, 1), (1, 1, 1, 1), (11, 1, 1, 1), (16, 1, 1, 0), (17, 1, 1, 1), (46, 1, 1, 1),
                          (47, 0, 1, 1), (48, 1, 1, 1)):
        s = dict(state, t=np.array([float(t)]), m=np.array([[bool(a), False, bool(b), False]]),
                 vis=np.array([[True, True, bool(seen), True]]))
        obs, mem = ob.observe(s, setup, 1, mem)
        got.append((round(float(obs.tokens[0, 0, S]), 4), round(float(obs.tokens[0, 2, S]), 4)))
    assert got == [(0, 0), (0, 0), (round(10 / 30, 4), round(10 / 30, 4)), (0.5, 0), (round(16 / 30, 4), 0),
                   (1, round(29 / 30, 4)), (0, 1), (0, 1)]


def test_memory_merges_keep_the_melee_clock():
    """rollout.MEMORY lists every Memory field after frame in order (merge_memory, branch snapshots build it by
    position)."""
    rollout = pytest.importorskip("tools.nn.train.rollout")
    assert rollout.MEMORY == tuple(f for f in ob.Memory.__dataclass_fields__ if f != "frame")


def test_an_older_checkpoint_loads_with_zero_weights_and_computes_what_it_did():
    torch = pytest.importorskip("torch")
    from tools.nn.model import policy
    from tools.nn.model.config import preset
    from tools.nn.model.encoder import TokenEncoder
    cfg = preset("v2", d=32, heads=2, pointer=16, critic_d=32, critic_heads=2, sector_d=16, eyes=False)
    first = ob.INDEX[ob.FX_STATS[0]]
    torch.manual_seed(0)
    sd = policy.Actor(cfg).state_dict()
    old = {k: (v[:, :first].clone() if k == "encoder.unit.0.weight" else v.clone()) for k, v in sd.items()}
    loaded = policy.Actor(cfg)
    loaded.load_state_dict(old)
    w = loaded.state_dict()["encoder.unit.0.weight"]
    assert torch.all(w[:, first:] == 0) and torch.equal(w[:, :first], old["encoder.unit.0.weight"])
    tokens = torch.randn(2, 5, ob.TOKEN)
    ctx = torch.randn(2, ob.CONTEXT)
    with_new = loaded.encoder(tokens, ctx)
    tokens[..., first:] = 0
    assert torch.equal(loaded.encoder(tokens, ctx), with_new)          # whatever the new columns: the same output
    # the old encoder of the old width, the same weights: the same output (a matrix product of other width: rounding)
    enc_old = TokenEncoder(32, n_token=first)
    enc_old.load_state_dict({k[len("encoder."):]: v for k, v in old.items() if k.startswith("encoder.")})
    torch.testing.assert_close(enc_old(tokens[..., :first], ctx), with_new, rtol=0, atol=1e-6)

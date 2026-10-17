"""Spells in the network (S27, 16.10.2026): a caster's spells are ability slots (tools/nn/model/abilities.py), the side's
winds in the context and the casting / struck columns of the token (observation.py), a spell falls on the unit the
target pointer chooses (heads.py, chain.py), Orders.ability SPELL_AT + k with Orders.cast_target reaches the simulator
(battle.py -> magic.request) and the bridge's line (companion/exchange.py); its price by the spell bricks
(abilities_price: bonuses.py, ability.py)."""
import numpy as np
import pytest

from tools.nn.abilities_price import ability as abl
from tools.nn.abilities_price import bonuses as bz
from tools.nn.abilities_price import validate as va
from tools.nn.model import abilities as mab
from tools.nn.model import observation as ob
from tools.nn.model import passport, sources

WIZARD, ENGINEER = "wh3_dlc25_emp_cha_wizard_metal", "wh2_main_skv_cha_warlock_engineer"
DOOM, LIGHTNING = "wh_main_spell_metal_searing_doom", "wh2_main_spell_ruin_warp_lightning"
GENERAL, WARLORD = "wh_main_emp_cha_general_0", "wh2_main_skv_cha_warlord_0"
CLANRATS = "wh2_main_skv_inf_clanrat_spearmen_0"
S = mab.SPELL_AT


def test_a_casters_spell_sits_in_the_last_active_slots():
    units = passport.load()
    assert WIZARD in units and ENGINEER in units                     # the casters' cards (probe-units.json)
    lay = mab.layout(WIZARD)
    assert lay[S] == DOOM and lay[:S] == [""] * S and lay[S + 1] == ""
    assert mab.layout(ENGINEER)[S] == LIGHTNING
    idx, usable, fx = mab.slots([WIZARD, GENERAL])
    assert mab.rows()[idx[0, S]] == DOOM and usable[0, S] and fx[0, S] == -1
    assert not (idx[1, mab.SPELL_SLOTS[0]:mab.ACTIVE] >= 0).any()          # no caster: no spell slot
    assert mab.sim_fits(idx)
    f = dict(zip(mab.SPELL_STATIC, mab.table()[idx[0, S], mab.STATIC:]))
    assert f["spell"] == 1 and f["mana"] == pytest.approx(6 / mab.WINDS_SCALE) and f["bomb_projectiles"] == 40 / 50
    assert mab.column("range_m", idx)[0, S] == pytest.approx(2.0) and mab.column("mana", idx)[1, 0] == 0
    assert not mab.table()[:len(mab.keys()), mab.STATIC:].any()       # an ability: no spell columns


def caster_battle(**state_kw):
    """Side 1: the Gold Wizard and a General; side 2: the Warlord and clanrats 150 m off."""
    keys = [WIZARD, GENERAL, WARLORD, CLANRATS]
    setup = ob.Setup(keys=[keys], side=np.array([[1, 1, 2, 2]]), bounds=np.array([sources.CROSSROADS], np.float32),
                     factions=[(sources.EMPIRE, sources.SKAVEN)], attacker=np.array([1]))
    _, state = sources.synthetic(batch=1, own=2, enemy=2, seed=3)
    state = dict(state, x=np.array([[0.0, -20.0, 300.0, 150.0]]), z=np.zeros((1, 4)), vis=np.ones((1, 4), bool),
                 hp=np.ones((1, 4)), ms=np.full((1, 4), 2.0), m=np.zeros((1, 4), bool), mp=np.ones((1, 4)),
                 men=np.array([[1.0, 1.0, 1.0, 100.0]]))
    state["w"], state["r"], state["s"] = state["ms"] == 5, state["ms"] >= 6, state["ms"] == 7
    for k in range(3):
        for t in ("on", "cd"):
            state[f"ab{k}_{t}"] = np.zeros((1, 4))
    for k in range(2):
        state[f"sp{k}_on"], state[f"sp{k}_cd"] = np.zeros((1, 4)), np.zeros((1, 4))
    state.update(wm_cur=np.full((1, 4), 12.0), wm_res=np.full((1, 4), 68.0), cast_k=np.full((1, 4), -1),
                 cast_now=np.zeros((1, 4), bool), bm_hit=np.zeros((1, 4), bool))
    state.update(state_kw)
    return setup, state


def test_a_spell_may_be_cast_when_recharged_and_paid_for():
    setup, state = caster_battle()
    obs, _ = ob.observe(state, setup, 1)
    J = mab.STATE_INDEX
    assert obs.abil_ok[0, 0, S] and obs.abil[0, 0, S, J["affordable"]] == 1
    assert obs.ctx[0, ob.CTX["winds_current"]] == pytest.approx(12 / 50)
    assert obs.ctx[0, ob.CTX["winds_reserve"]] == pytest.approx(68 / 50)
    # winds short, recharging, casting: no; the enemy's winds are not seen (side 2's context: its own)
    poor, _ = ob.observe(dict(state, wm_cur=np.array([[3.0, 3.0, 30.0, 30.0]])), setup, 1)
    assert not poor.abil_ok[0, 0, S] and poor.abil[0, 0, S, J["affordable"]] == 0
    assert poor.ctx[0, ob.CTX["winds_current"]] == pytest.approx(3 / 50)
    cd, _ = ob.observe(dict(state, sp0_cd=np.array([[20.0, 0, 0, 0]])), setup, 1)
    assert not cd.abil_ok[0, 0, S] and cd.abil[0, 0, S, J["recharge_left"]] == pytest.approx(20 / 60)
    busy, _ = ob.observe(dict(state, cast_k=np.array([[0, -1, -1, -1]])), setup, 1)
    assert not busy.abil_ok[0, 0, S] and busy.abil[0, 0, S, J["casting"]] == 1
    # nobody to fall on (no enemy seen): no
    blind, _ = ob.observe(dict(state, vis=np.array([[True, True, False, False]])), setup, 1)
    assert not blind.abil_ok[0, 0, S]
    # no range mask: a target 400 m off still lets it be cast (the caster walks in)
    far, _ = ob.observe(dict(state, x=np.array([[0.0, -20.0, 600.0, 400.0]])), setup, 1)
    assert far.abil_ok[0, 0, S]


def test_the_enemy_caster_shows_its_card_and_its_glow_not_its_winds():
    setup, state = caster_battle(cast_now=np.array([[True, False, False, False]]),
                                 bm_hit=np.array([[False, False, False, True]]),
                                 wm_cur=np.array([[12.0, 12.0, 20.0, 20.0]]))
    obs2, _ = ob.observe(state, setup, 2)                       # side 2 looks at the Gold Wizard
    J = mab.STATE_INDEX
    assert obs2.abil[0, 0, S, J["owned"]] == 1 and obs2.abil_idx[0, 0, S] == mab.rows().index(DOOM)
    assert obs2.abil[0, 0, S, J["affordable"]] == 0 and obs2.abil[0, 0, S, J["recharge_left"]] == 0
    assert obs2.tokens[0, 0, ob.INDEX["casting_now"]] == 1 and obs2.tokens[0, 3, ob.INDEX["spell_strike"]] == 1
    assert obs2.ctx[0, ob.CTX["winds_current"]] == pytest.approx(20 / 50)     # its own winds, never the enemy's
    hidden, _ = ob.observe(dict(state, vis=np.array([[False, True, True, True]])), setup, 2)
    assert hidden.tokens[0, 0, ob.INDEX["casting_now"]] == 0   # not seen: not shown


def test_the_spell_bricks_count_the_casts_own_work():
    b = bz.bricks()
    S1 = 21
    on = np.zeros(S1)
    on[4:12] = np.arange(8, 0, -1)                               # launched at 4 s, 8 s active
    gold = np.zeros(S1)
    gold[8:13] = np.cumsum([100, 200, 300, 100, 50])             # landings 8-12 s (the last a second after the effect)
    gold[13:] = gold[12]
    gold[18:] += 500                                             # a later cast's: not this press's
    tr = {"alive": np.ones((S1, 1), bool), "owner": 0, "dt_s": 1.0, "on": on, "sp_gold_e": gold,
          "sp_gold_f": gold * 0.1, "sp_knock": np.zeros(S1), "winds": np.full(S1, 6.0)}
    a = abl.Ability.from_spell(DOOM, mab.load_spells()[DOOM], b)
    c = a.counters(tr, b)
    assert c["spell_damage"] == pytest.approx(750) and c["spell_friendly"] == pytest.approx(75)
    assert c["winds_cost"] == 0                                   # 6 left: the next cast is paid for
    assert a.counters(dict(tr, winds=np.full(S1, 2.0)), b)["winds_cost"] == 6
    assert a.counters(dict(tr, on=np.zeros(S1)), b) == {k: 0.0 for k in c}      # never launched: nothing
    assert all(t.ok for t in va.auto_tests({k: b[k] for k in bz.SPELL_KINDS}))


def test_the_bridge_line_carries_the_spells_target():
    from tools.nn.companion import exchange
    names = ["own_wiz", "own_gen", "enemy_rats"]
    slots = [mab.layout(WIZARD), mab.layout(GENERAL), mab.layout(CLANRATS)]
    ab = np.array([S, -1, -1])
    out = exchange.ability_list(names, np.array([1, 1, 2]), ab, slots, cast=np.array([2, -1, -1]))
    assert out == [{"unit": "own_wiz", "key": DOOM, "target": "enemy_rats"}]
    assert exchange.ability_list(names, np.array([1, 1, 2]), ab, slots, cast=np.array([-1, -1, -1])) == []
    text = exchange.orders_text("b", 1, [], abilities=out)
    assert f"ability own_wiz {DOOM} enemy_rats" in text
    doc = {"t": 10000, "winds": {"cur": 9, "rate": 0.75},
           "units": [{"n": "own_wiz", "side": 1}, {"n": "own_gen", "side": 1}, {"n": "enemy_rats", "side": 2}]}
    assert exchange.winds_of(doc) == pytest.approx((9.0, 67.0))            # the reserve by its flow when not read
    assert exchange.winds_of(dict(doc, winds={"cur": 9, "res": 50})) == (9.0, 50.0)
    doc["abilities_used"] = {"own_wiz": {DOOM: 4000}}
    t = exchange.ability_timers(doc, names, slots)
    assert t["sp0_on"][0, 0] == pytest.approx(2.0) and t["sp0_cd"][0, 0] == pytest.approx(32.0)


# --- torch (skipped in the project's .venv) ---

def test_the_networks_spell_order_reaches_the_simulator():
    torch = pytest.importorskip("torch")
    from tools.nn.abilities_price import lab_spell
    from tools.nn.sim import battle, orders as O, scenario
    params = lab_spell.params_with_casters()
    army = {"attacker": 1, "sides": {
        1: {"faction": "wh_main_emp_empire", "ai": False, "units": [
            {"key": GENERAL, "x": -600.0, "z": 0.0, "general": True}, {"key": WIZARD, "x": 0.0, "z": 0.0}]},
        2: {"faction": "wh2_main_skv_skaven", "ai": False, "units": [
            {"key": WARLORD, "x": 700.0, "z": 0.0, "general": True}, {"key": CLANRATS, "x": 120.0, "z": 0.0}]}}}
    st = scenario.build([army], params, per_side=2)
    o = O.hold(1, 4)
    o.ability[0, 1], o.cast_target[0, 1] = O.SPELL_AT, 3
    O.check(o, 4)
    battle.step(st, o, params)
    assert st.u["cast_k"][0, 1] == 0 and st.u["wm_cur"][0, 1] == pytest.approx(12 - 6 + 0.05, abs=0.06)
    for _ in range(20):
        battle.step(st, O.hold(1, 4), params)
    u = st.u
    assert u["sp0_cd"][0, 1] > 25 and u["bm_gold_e"][0, 1] > 50 and u["bm_gold_f"][0, 1] == 0
    obs = st.observation()
    assert obs["sp0_cd"][0, 1] > 0 and "cast_now" in obs and "bm_hit" in obs


def test_a_spell_falls_on_the_target_pointers_choice():
    torch = pytest.importorskip("torch")
    from tools.nn.model import heads as hd
    B, N = 1, 4
    logits = {"kind": torch.tensor([[[9.0, -9, -9, -9, -9]] * N]), "point": torch.zeros(B, N, 3),
              "target": torch.full((B, N, N), hd.NEG), "run": torch.zeros(B, N),
              "ability": torch.full((B, N, 1 + mab.ACTIVE), hd.NEG)}
    logits["target"][0, :, 3] = 0.0
    logits["target"][0, :, 2] = 0.5
    logits["ability"][..., 0] = 0.0
    logits["ability"][0, 0, 1 + S] = 20.0                        # unit 0 casts its spell
    a = hd.sample(logits, greedy=True, abilities=True)
    assert a.ability[0, 0] == S and a.kind[0, 0] == hd.HOLD and a.target[0, 0] == 2
    assert a.target[0, 1] == -1                                    # no spell, no attack: no target
    ctrl = torch.ones(B, N, dtype=torch.bool)
    lp, _ = hd.log_prob(logits, a, ctrl)
    lt = torch.log_softmax(logits["target"][0, 0], -1)[2]
    la = torch.log_softmax(logits["ability"][0, 0], -1)[1 + S]
    assert lp[0, 0] == pytest.approx(float(torch.log_softmax(logits["kind"][0, 0], -1)[0] + la + lt), abs=1e-5)
    from tools.nn.model.decide import to_orders
    from tools.nn.model.frame import Frame
    obs_t = {"ctrl": ctrl, "pos": torch.zeros(B, N, 2)}
    fr = Frame(torch.zeros(1), torch.zeros(1), torch.ones(1), torch.zeros(1))
    cfg = type("C", (), {"sectors": 0, "grid": 0, "n_dir": 1, "n_dist": 3, "dist_min": 10.0, "dist_max": 40.0})()
    o = to_orders(cfg, a, obs_t, fr, torch.tensor([[-1e3, 1e3, -1e3, 1e3]]))
    assert o.ability[0, 0] == S and o.cast_target[0, 0] == 2 and o.target[0, 0] == -1 and o.spell()[0, 0] == 0


def test_an_older_network_loads_with_zero_weights_for_the_spell_inputs():
    torch = pytest.importorskip("torch")
    from tools.nn.model import pairs, policy
    from tools.nn.model.config import preset
    cfg = preset("v2", d=32, heads=2, pointer=16, critic_d=32, critic_heads=2, sector_d=16, eyes=False)
    torch.manual_seed(0)
    new = policy.Actor(cfg)
    sd = new.state_dict()
    old = {}
    n_sp = len(mab.SPELL_STATIC) + len(mab.SPELL_DYNAMIC)
    for k, v in sd.items():                  # the shapes of a network saved before the spells
        if k == "encoder.ctx.0.weight":
            v = v[:, :ob.CONTEXT - len(ob.WINDS)]
        elif k == "encoder.unit.0.weight":
            v = v[:, :ob.TOKEN - len(ob.MAGIC) - len(ob.SHAPE)]
        elif k.endswith(("abilities.net.0.weight", "ability_extra.net.0.weight", "ability_k.0.weight")):
            v = v[:, :mab.SIZE - n_sp]
        elif k in ("pair_bias.lin.weight", "heads.target_pair.weight"):
            v = v[:, :pairs.OLD_SIZE]
        old[k] = v.clone()
    loaded = policy.Actor(cfg)
    loaded.load_state_dict(old)
    lsd = loaded.state_dict()
    assert torch.all(lsd["encoder.ctx.0.weight"][:, ob.CTX["winds_current"]:ob.CONTEXT] == 0)
    assert torch.all(lsd["encoder.unit.0.weight"][:, ob.INDEX["casting_now"]:] == 0)
    assert torch.all(lsd["heads.ability_k.0.weight"][:, mab.SIZE - n_sp:] == 0)
    assert torch.all(lsd["heads.target_pair.weight"][:, pairs.OLD_SIZE:] == 0)
    for k, v in old.items():
        w = lsd[k]
        assert torch.equal(w[tuple(slice(0, s) for s in v.shape)], v), k


def test_a_cast_is_priced_by_direct_accounting_over_its_window():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from tools.nn.abilities_price import lab_spell
    from tools.nn.model import heads as hd
    from tools.nn.sim import battle, orders as O, scenario
    from tools.nn.train import spell_price
    params = lab_spell.params_with_casters()
    army = {"attacker": 1, "sides": {
        1: {"faction": "wh_main_emp_empire", "ai": False, "units": [
            {"key": GENERAL, "x": -600.0, "z": 0.0, "general": True}, {"key": WIZARD, "x": 0.0, "z": 0.0}]},
        2: {"faction": "wh2_main_skv_skaven", "ai": False, "units": [
            {"key": WARLORD, "x": 700.0, "z": 0.0, "general": True}, {"key": CLANRATS, "x": 120.0, "z": 0.0}]}}}
    st = scenario.build([army], params, per_side=2)
    env = SimpleNamespace(st=st, rows_learn=torch.tensor([0]), B=1, decision_s=1.0)
    N = st.N
    ability = torch.full((1, N), -1)
    ability[0, 1] = S
    idx = torch.full((1, N, mab.SLOTS), -1)
    idx[0, 1, S] = mab.rows().index(DOOM)
    info = {"obs": {"ctrl": torch.ones(1, N, dtype=torch.bool), "abil_idx": idx},
            "action": hd.Action(kind=torch.zeros(1, N, dtype=torch.long), point=torch.zeros(1, N, dtype=torch.long),
                                target=torch.full((1, N), -1), run=torch.zeros(1, N, dtype=torch.bool),
                                ability=ability),
            "logits": {"ability": torch.zeros(1, N, 1 + mab.ACTIVE)}}
    led = spell_price.Ledger(k_winds=10.0, horizon_s=30.0)
    led.add(env, info, 0)
    o = O.hold(1, N)
    o.ability[0, 1], o.cast_target[0, 1] = O.SPELL_AT, 3
    for d in range(25):
        for s in range(2):
            battle.step(st, o if (d == 0 and s == 0) else O.hold(1, N), params)
        led.update(env, 1.0)
    gold_e = float(st.u["bm_gold_e"][0, 1])
    res = led.result(env, 64)
    assert led.stats["spell_priced"] == 1 and led.stats["spell_launched"] == 1
    budget = float(torch.stack([(st.u["cost"] * (st.u["side"] == s)).sum() for s in (1, 2)]).mean())
    assert float(res["share"][0]) * budget == pytest.approx(gold_e - 10.0 * 6.0, rel=1e-4)
    assert gold_e > 50 and float(res["lp"][0]) == pytest.approx(-np.log(1 + mab.ACTIVE), abs=1e-5)
    # a cast whose window is still open at the chunk's end is not priced
    led2 = spell_price.Ledger(k_winds=10.0)
    led2.add(env, info, 0)
    led2.update(env, 1.0)
    assert led2.result(env, 64) is not None and led2.stats["spell_priced"] == 0

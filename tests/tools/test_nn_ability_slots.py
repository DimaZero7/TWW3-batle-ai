"""Every ability of a unit reaches the network (13.10.2026, tools/nn/model/abilities.py): SLOTS slots a unit - the
actives and game-fired ones from slot 0 (the simulator's order and timer slots), the passives from ACTIVE - none
dropped; a passive shows its condition and whether it is on now; a vigour column; the stat bonuses other units'
abilities lay on a unit (the token's RECEIVED); an older network acts as it did (zero weights for the new inputs)."""
import json

import numpy as np
import pytest

from tools import config as project
from tools.nn.model import abilities as mab
from tools.nn.model import effects, passport, sources
from tools.nn.model import observation as ob
from tools.nn.sim import abilities as sab

GENERAL, WARLORD = "wh_main_emp_cha_general_0", "wh2_main_skv_cha_warlord_0"
FLAGELLANTS, MONKS, CLANRATS = ("wh_dlc04_emp_inf_flagellants_0", "wh2_main_skv_inf_plague_monks",
                                "wh2_main_skv_inf_clanrats_0")
FS, HTL, WOUNDED, SCURRY = ("wh_main_character_abilities_foe_seeker", "wh_main_lord_passive_hold_the_line",
                            "wh3_main_unit_passive_single_entity", "wh2_main_unit_passive_scurry_away")
A = mab.ACTIVE


def pool_units():
    """Every unit key of config/nn/pools.json (units, units_v2 and the lord of each faction)."""
    pools = json.loads((project.ROOT / "config" / "nn" / "pools.json").read_text(encoding="utf-8"))
    out = set()
    for f in pools["factions"].values():
        out |= {u["key"] for u in f.get("units", []) + f.get("units_v2", [])}
        if isinstance(f.get("lord"), dict):
            out.add(f["lord"]["key"])
    return sorted(out)


def test_no_ability_of_any_unit_is_lost():
    units, passports = passport.load(), mab.load()
    keys = pool_units()
    assert GENERAL in keys and WARLORD in keys and MONKS in keys
    most = {"actives": 0, "passives": 0}
    for key in sorted(set(keys) | set(units)):                  # our pools, and every unit with a passport
        own = list(units[key].get("abilities") or ())
        assert all(a in passports for a in own), key            # every ability has its passport
        lay = mab.layout(key, units, passports)
        assert sorted(a for a in lay if a) == sorted(own), key   # none dropped, none twice
        act = [a for a in lay[:A] if a]
        pas = [a for a in lay[A:] if a]
        assert all(not passports[a]["passive"] for a in act) and all(passports[a]["passive"] for a in pas)
        assert lay[:len(act)] == act and lay[A:A + len(pas)] == pas    # packed from their first slot
        # the timed ones fit the simulator's timer slots, the network's slot k is the simulator's slot k
        assert len(act) <= sab.SLOTS, key
        assert sab.slot_keys(key, units, passports)[:len(act)] == act
        if key in keys:
            most = {"actives": max(most["actives"], len(act)), "passives": max(most["passives"], len(pas))}
    # the most in our pools (the head chooses among ACTIVE = 15): 3 actives (the Warlord), 3 passives (Plague Monks)
    assert most == {"actives": 3, "passives": 3} and most["actives"] <= A
    assert mab.layout(WARLORD).count("") == mab.SLOTS - 5            # the Warlord's 5, Wounded and Scurry included


def test_the_table_order_is_the_simulators():
    from tools.nn.sim.params import load
    assert sab.keys(load()) == mab.keys()
    assert mab.table().shape == (len(mab.keys()) + 1, mab.STATIC) and not mab.table()[-1].any()


def test_new_columns_are_last_and_vigour_restores():
    assert mab.SIZE == mab.OLD_SIZE + 3 and list(mab.INDEX)[-3:] == ["when_health_below_quarter", "vigour", "on_now"]
    t = dict(zip(mab.STATIC_NAMES, mab.table()[mab.keys().index(FS)]))
    # Foe-Seeker: the database's fatigue_change_ratio -0.01 - the simulator takes 1 % of the maximum fatigue off a
    # second (tools/nn/sim/fatigue.py: fatigue + vigour x max x dt), it restores vigour: + 1
    assert mab.load()[FS]["vigour_per_s"] == pytest.approx(-0.01) and t["vigour"] == pytest.approx(1.0)
    w = dict(zip(mab.STATIC_NAMES, mab.table()[mab.keys().index(WOUNDED)]))
    assert w["passive"] == 1 and w["when_health_below_quarter"] == 1 and w["other_condition"] == 0
    s = dict(zip(mab.STATIC_NAMES, mab.table()[mab.keys().index(SCURRY)]))
    assert s["off_not_wavering"] == 1 and s["vigour"] == 0              # on only while wavering or routing


def test_the_slots_a_network_saw_before():
    idx, _, _ = mab.slots([GENERAL, WARLORD, MONKS, FLAGELLANTS])
    seen = mab.legacy(idx >= 0)
    # the simulator's first 3: the General's 2 actives + Hold the Line; the Warlord's 3 actives; the monks' 3 passives
    assert seen[0].nonzero()[0].tolist() == [0, 1, A] and seen[1].nonzero()[0].tolist() == [0, 1, 2]
    assert seen[2].nonzero()[0].tolist() == [A, A + 1, A + 2] and seen[3].nonzero()[0].tolist() == [0, A]
    names = mab.keys()
    for i, key in enumerate([GENERAL, WARLORD, MONKS, FLAGELLANTS]):
        old = [a for a in sab.slot_keys(key, passport.load(), mab.load()) if a]
        assert sorted(names[j] for j in idx[i][seen[i]]) == sorted(old)


def battle(keys, hp=None, ms=None, sides=(1, 1, 2, 2)):
    setup = ob.Setup(keys=[list(keys)], side=np.array([list(sides)]), bounds=np.array([sources.CROSSROADS], np.float32),
                     factions=[(sources.EMPIRE, sources.SKAVEN)], attacker=np.array([1]))
    _, state = sources.synthetic(batch=1, own=2, enemy=2, seed=3)
    state = dict(state, x=np.array([[-20.0, 0.0, 300.0, 320.0]]), z=np.zeros((1, 4)), vis=np.ones((1, 4), bool),
                 hp=np.array([hp or [1.0] * 4]), ms=np.array([ms or [2.0] * 4]), m=np.zeros((1, 4), bool),
                 mp=np.ones((1, 4)))
    state["w"], state["r"], state["s"] = state["ms"] == 5, state["ms"] >= 6, state["ms"] == 7
    for k in range(sab.SLOTS):
        for t in ("on", "cd"):
            state[f"ab{k}_{t}"] = np.zeros((1, 4))
    return setup, state


def test_a_passive_shows_whether_it_is_on_now():
    setup, state = battle([GENERAL, CLANRATS, WARLORD, CLANRATS], hp=[0.2, 1.0, 1.0, 0.4], ms=[2, 5, 2, 2])
    obs, _ = ob.observe(state, setup, 1)
    on = obs.abil[0, :, :, mab.STATE_INDEX["on_now"]]
    J = mab.STATE_INDEX
    # the General below 25 %: Wounded on; Hold the Line on (no condition); no timer runs (active now: actives only)
    assert on[0, [A, A + 1]].tolist() == [1, 1] and not obs.abil[0, 0, A:, J["active_now"]].any()
    # the wavering clanrats: Scurry Away! on; Strength in Numbers on (health over half)
    assert on[1, [A, A + 1]].tolist() == [1, 1]
    # the enemy clanrats below half: Strength in Numbers off, Scurry Away! off (steady); the Warlord's: off
    assert on[3, [A, A + 1]].tolist() == [0, 0] and on[2, A] == 0
    healthy, _ = ob.observe(dict(state, hp=np.array([[0.9, 1.0, 1.0, 0.4]])), setup, 1)
    assert healthy.abil[0, 0, A + 1, J["on_now"]] == 0                 # Wounded off above 25 %
    hidden, _ = ob.observe(dict(state, vis=np.array([[True, True, False, False]])), setup, 1)
    assert not hidden.abil[0, 2:, :, J["on_now"]].any()                 # an enemy's only while seen
    assert hidden.abil[0, 3, A:A + 2, J["owned"]].tolist() == [1, 1]   # its card: owned


def test_the_ability_head_chooses_among_the_actives_only():
    setup, state = battle([GENERAL, CLANRATS, WARLORD, CLANRATS])
    obs, _ = ob.observe(state, setup, 1)
    assert obs.abil_ok.shape == (1, 4, A) and obs.abil_ok[0, 0].nonzero()[0].tolist() == [0, 1]
    obs2, _ = ob.observe(state, setup, 2)
    assert obs2.abil_ok[0, 2].nonzero()[0].tolist() == [0, 1, 2]
    # a slot without the state's timers is never ready (the simulator keeps the first sab.SLOTS)
    only0 = {k: v for k, v in state.items() if not (k.startswith("ab1") or k.startswith("ab2"))}
    obs3, _ = ob.observe(only0, setup, 2)
    assert obs3.abil_ok[0, 2].nonzero()[0].tolist() == [0]


def test_effects_on_me_from_others():
    setup, state = battle([GENERAL, CLANRATS, WARLORD, CLANRATS])
    a = mab.keys()
    rx = np.zeros((1, 4), np.int64)
    rx[0, 1] = 1 << a.index(HTL)                                       # the General's aura on the clanrats
    rx[0, 3] = 1 << a.index("wh_main_character_abilities_rally")       # the Warlord's Rally on his clanrats
    rx[0, 0] = 1 << a.index("wh_main_character_abilities_rally")       # (an enemy's on the General: its enemies group)
    obs, _ = ob.observe(dict(state, ab_rx=rx), setup, 1)
    I = ob.INDEX
    col = lambda name: obs.tokens[0, :, I[f"rx_{name}"]]
    assert col("stat_melee_defence_add").tolist() == pytest.approx([0, 5 / 50, 0, 0])
    assert col("stat_morale_add").tolist() == pytest.approx([0, 4 / 50, 0, 16 / 50])
    assert col("scalar_speed_mult").tolist() == pytest.approx([0, 0, 0, 0])
    hidden, _ = ob.observe(dict(state, ab_rx=rx, vis=np.array([[True, True, False, False]])), setup, 1)
    assert hidden.tokens[0, 3, I["rx_stat_morale_add"]] == 0           # an enemy's: only while seen
    none, _ = ob.observe(state, setup, 1)
    assert not none.tokens[..., [I[n] for n in ob.RECEIVED]].any()     # no ab_rx in the state: none
    assert set(ob.RECEIVED).isdisjoint(ob.NAMES[i] for i in ob.OWN_ONLY)


def test_the_companion_reads_the_effects_of_others_from_the_card():
    from tools.nn.companion import exchange
    lord = {"n": "own_lord", "side": 1, "key": GENERAL, "fx": [HTL, FS]}
    spear = {"n": "own_spear", "side": 1, "key": "wh_main_emp_inf_spearmen_0", "fx": [HTL]}
    rat = {"n": "enemy_rat", "side": 2, "key": CLANRATS, "fx": [SCURRY]}
    bits = exchange.received({"units": [lord, spear, rat]}, ["own_lord", "own_spear", "enemy_rat"])
    a = mab.keys()
    # the lord's own aura and ability are his slots, not from others; the spearmen's Hold the Line is; the rat's own
    assert bits[0].tolist() == [0, 1 << a.index(HTL), 0]


def test_recorded_slots_of_before_widen():
    from tools.nn.observe.tokens import widen_abil
    setup, state = battle([GENERAL, CLANRATS, WARLORD, MONKS])
    state["ab1_on"] = np.array([[5.0, 0, 3.0, 0]])
    obs, _ = ob.observe(state, setup, 1)
    # the old record: the simulator's 3 slots, state + passport (OLD_SIZE), its legacy view
    idx, owned = obs.abil_idx, obs.abil[..., 0] > 0.5
    seen = mab.legacy(owned)
    old = np.zeros(idx.shape[:-1] + (3, mab.OLD_SIZE), np.float32)
    old_ok = np.zeros(idx.shape[:-1] + (3,), bool)
    full = mab.full(obs.abil, idx)[..., :mab.OLD_SIZE]
    for n in range(4):
        ks = seen[0, n].nonzero()[0]
        old[0, n, :len(ks)] = full[0, n, ks]
        old_ok[0, n, :len(ks)] = obs.abil_ok[0, n, ks[ks < A]].tolist() + [False] * (len(ks) - (ks < A).sum())
    a = widen_abil({"abil": old.astype(np.float16), "abil_ok": old_ok})
    assert a["abil"].shape == obs.abil.shape and a["abil_ok"].shape == obs.abil_ok.shape
    assert np.array_equal(a["abil_idx"], np.where(seen, idx, -1))
    assert np.array_equal(a["abil_ok"], obs.abil_ok)
    keep = seen[..., None]
    want = np.where(keep, obs.abil, 0)
    want[..., A:, mab.STATE_INDEX["on_now"]] = 0                     # a passive's on now was not recorded
    assert np.allclose(a["abil"], want, atol=2e-3)


# --- torch (skipped in the project's .venv) ---

def test_the_simulator_marks_the_abilities_of_others():
    torch = pytest.importorskip("torch")
    from tools.nn.sim import battle as sim_battle
    from tools.nn.sim import orders as O
    from tools.nn.sim import scenario
    from tools.nn.sim.params import load
    params = load()
    spear = "wh_main_emp_inf_spearmen_0"
    units = lambda rows: [{"key": k, "x": x, "z": z, "b": 0, "general": g} for k, x, z, g in rows]
    army = {"attacker": 1, "sides": {
        1: {"faction": "wh_main_emp_empire", "units": units([(GENERAL, 0, 0, True), (spear, 20, 0, False),
                                                             (spear, 300, 0, False)])},
        2: {"faction": "wh2_main_skv_skaven", "units": units([(CLANRATS, 0, 800, False)])}}}
    st = scenario.build([army], params)
    st = sim_battle.step(st, O.hold(st.B, st.N), params)
    rx = st.observation()["ab_rx"][0]
    bit = 1 << mab.keys().index(HTL)
    gen, near, far = (st.keys[0].index(GENERAL), 1, 2)
    assert st.keys[0][near] == spear and st.keys[0][far] == spear
    assert int(rx[near]) & bit                                          # the aura on the spearmen within 35 m
    assert not int(rx[far]) & bit and int(rx[gen]) == 0                 # far: none; the General's own: not "others"
    assert torch.is_tensor(rx)


def _old_view(o):
    """The observation as a network saved before 13.10.2026 read it: no RECEIVED, no on now, its 3 slots only."""
    import torch
    out = dict(o)
    out["tokens"] = torch.cat([o["tokens"][..., :ob.TOKEN - len(ob.RECEIVED)],
                               torch.zeros_like(o["tokens"][..., ob.TOKEN - len(ob.RECEIVED):])], -1)
    seen = mab.legacy(o["abil"][..., 0] > 0.5)
    abil = torch.where(seen[..., None], o["abil"], torch.zeros_like(o["abil"]))
    abil[..., mab.STATE_INDEX["on_now"]] = 0
    out["abil"], out["abil_idx"] = abil, torch.where(seen, o["abil_idx"], torch.full_like(o["abil_idx"], -1))
    return out


@pytest.mark.parametrize("preset", ["small", "v2"])
def test_an_older_network_acts_as_it_did(preset):
    """An actor saved before 13.10.2026 (no ability_extra, ability features of OLD_SIZE, a token without RECEIVED)
    computes on today's inputs exactly what it computes on the inputs it knew (the new ones cut): they enter with zero
    weights. (The real itS25_12 played 8 greedy battles x 240 decisions bit-identically with the old code.)"""
    torch = pytest.importorskip("torch")
    from tools.nn.model import config, policy
    small = dict(d=64, layers=2, heads=4, critic_d=64, critic_layers=2, critic_heads=4)
    cfg = config.preset(preset, **small, **({"sector_d": 32, "sectors": 4, "fine": 4} if preset == "v2" else {}))
    torch.manual_seed(5)
    trained = policy.Actor(cfg).eval()
    with torch.no_grad():                         # a trained ability encoder (its last layer not zero)
        for p in trained.abilities.parameters():
            p.normal_(0, 0.3)
    sd = {k: v for k, v in trained.state_dict().items() if not k.startswith("ability_extra.")}
    for k in ("abilities.net.0.weight", "heads.ability_k.0.weight"):
        sd[k] = sd[k][:, :mab.OLD_SIZE].clone()   # saved before the new columns
    sd["encoder.unit.0.weight"] = sd["encoder.unit.0.weight"][:, :ob.TOKEN - len(ob.RECEIVED)].clone()
    new = policy.Actor(cfg).eval()
    missing, _ = new.load_state_dict(sd)
    assert {k.split(".")[0] for k in missing} == {"ability_extra"}
    setup, state = battle([GENERAL, CLANRATS, WARLORD, MONKS], hp=[0.2, 1.0, 1.0, 0.4], ms=[2, 5, 2, 2])
    state["ab1_on"], state["ab0_cd"] = np.array([[5.0, 0, 3.0, 0]]), np.array([[0.0, 0, 30.0, 0]])
    state["ab_rx"] = np.array([[0, 1 << mab.keys().index(HTL), 0, 0]])
    h_a = h_b = mem = None
    for i in range(2):
        obs, mem = ob.observe(dict(state, t=np.array([30.0 + i])), setup, 1, mem)
        o = policy.to_torch(obs)
        assert o["tokens"][..., ob.TOKEN - len(ob.RECEIVED):].abs().sum() > 0       # the new inputs are there
        assert (o["abil"][..., mab.STATE_INDEX["on_now"]] > 0).any()
        assert (~mab.legacy(o["abil"][..., 0] > 0.5) & (o["abil"][..., 0] > 0.5)).any()   # and slots it never saw
        if preset == "v2":
            o = dict(o, free=o["ctrl"].clone(), commit=torch.zeros(*o["ctrl"].shape, 2))
        with torch.no_grad():
            la, a, h_a = new.act(o, h_a, greedy=True)
            lb, b, h_b = new.act(_old_view(o), h_b, greedy=True)
        for k in la:
            assert torch.equal(la[k], lb[k]), k
        assert torch.equal(h_a, h_b) and torch.equal(a.kind, b.kind) and torch.equal(a.ability, b.ability)
        assert la["ability"].shape[-1] == 1 + A


def test_adams_moments_go_on_with_the_new_parts(tmp_path):
    torch = pytest.importorskip("torch")
    from tools.nn.model import config, critic, policy
    from tools.nn.train import checkpoint, run
    cfg = config.preset("small", d=32, layers=1, heads=2, critic_d=32, critic_layers=1, critic_heads=2)
    torch.manual_seed(1)
    actor, crit = policy.Actor(cfg), critic.Critic(cfg)
    opt = run.optimizer(actor, crit, 1e-3)
    for p in opt.param_groups[0]["params"]:
        p.grad = torch.ones_like(p)
    opt.step()
    # an older actor: no ability_extra, ability inputs of the old width
    old_names = [n for n, _ in actor.named_parameters() if not n.startswith("ability_extra.")]
    saved = opt.state_dict()
    keep = [i for i, (n, _) in enumerate(list(actor.named_parameters()) + [("critic." + n, p) for n, p in
                                                                             crit.named_parameters()])
            if not n.startswith("ability_extra.")]
    st = {j: saved["state"][i] for j, i in enumerate(keep)}
    params = list(actor.named_parameters()) + list(crit.named_parameters())
    for j, i in enumerate(keep):
        n = params[i][0]
        if n in ("abilities.net.0.weight", "heads.ability_k.0.weight"):
            st[j] = {k: (v[:, :mab.OLD_SIZE].clone() if torch.is_tensor(v) and v.dim() == 2 else v)
                     for k, v in st[j].items()}
    old_sd = {"state": st, "param_groups": [dict(saved["param_groups"][0], params=list(range(len(keep))))]}
    sd = {k: v for k, v in actor.state_dict().items() if not k.startswith("ability_extra.")}
    path = tmp_path / "old.pt"
    torch.save({"format": checkpoint.FORMAT, "actor": sd,
                "train": {"optim": old_sd, "state": {"updates": 3}}}, path)
    new_actor, new_crit = policy.Actor(cfg), critic.Critic(cfg)
    new_opt = run.optimizer(new_actor, new_crit, 1e-3)
    fresh = run.fresh_params(new_actor, path)
    assert len(fresh) == 4 and all(any(p is q for q in new_actor.ability_extra.parameters()) for p in fresh)
    assert run.resume(new_opt, path, fresh) == {"updates": 3}
    w = new_opt.state[new_actor.abilities.net[0].weight]["exp_avg"]
    assert w.shape[1] == mab.SIZE and not w[:, mab.OLD_SIZE:].any() and w[:, :mab.OLD_SIZE].any()
    assert new_actor.ability_extra.net[0].weight not in new_opt.state
    assert torch.equal(new_opt.state[new_crit.encoder.unit[0].weight]["exp_avg"],
                       opt.state[crit.encoder.unit[0].weight]["exp_avg"])
    assert old_names


def test_the_stored_slots_come_back_whole():
    torch = pytest.importorskip("torch")
    from tools.nn.train import rollout
    setup, state = battle([GENERAL, CLANRATS, WARLORD, MONKS], hp=[0.2, 1.0, 1.0, 0.4])
    obs, _ = ob.observe(state, setup, 1)
    o = {k: torch.as_tensor(np.asarray(v)) for k, v in (("abil", obs.abil), ("abil_idx", obs.abil_idx))}
    bank_idx = o["abil_idx"].clone()                                  # [M = 1, N, SLOTS]
    used = rollout.used_slots(bank_idx)
    assert used.tolist() == [0, 1, 2, A, A + 1, A + 2]                 # 6 of the 30 slots
    stored = {"abil": o["abil"][..., used, :][None], "abil_row": torch.zeros(1, 1, dtype=torch.long)}
    back = rollout.full_obs(stored, {"idx": bank_idx, "slots": used})
    assert torch.equal(back["abil"][0], o["abil"]) and torch.equal(back["abil_idx"][0], o["abil_idx"])


def test_compiled_and_eager_abilities_agree():
    """The encoders' compiled path (the simulator's: the first SIM_SLOTS actives, the passives by table) gives the
    eager path's sum (torch.compiler.is_compiling patched; the real compile is the training's, on CUDA)."""
    torch = pytest.importorskip("torch")
    from tools.nn.model import encoder
    setup, state = battle([GENERAL, CLANRATS, WARLORD, MONKS], hp=[0.2, 1.0, 1.0, 0.4], ms=[2, 5, 2, 2])
    state["ab1_on"] = np.array([[5.0, 0, 3.0, 0]])
    obs, _ = ob.observe(state, setup, 1)
    a, i = torch.as_tensor(obs.abil), torch.as_tensor(obs.abil_idx)
    torch.manual_seed(0)
    for m in (encoder.AbilityEncoder(16), encoder.AbilityExtra(16)):
        with torch.no_grad():
            for p in m.parameters():
                p.normal_(0, 0.3)
            eager = m(a, i)
            orig = torch.compiler.is_compiling
            torch.compiler.is_compiling = lambda: True
            try:
                comp = m(a, i)
            finally:
                torch.compiler.is_compiling = orig
        assert torch.allclose(eager, comp, atol=1e-5) and eager.abs().sum() > 0

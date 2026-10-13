"""The AIM input (13.10.2026): every missile unit's aim before the first shot and its silence after a new target, the
simulator's own numbers (tools/nn/sim/params.aim_times), on both sides' tokens; older checkpoints (token without
AIM) load with zero weights and act as before; recorded tokens and Adam's moments of the old width widen."""
import numpy as np
import pytest

from tools.nn.model import observation as ob
from tools.nn.model import passport, sources
from tools.nn.sim import params as sim_params

GUN, ARCHERS, HANDGUNS = "wh2_dlc12_skv_inf_ratling_gun_0", "wh2_dlc13_emp_inf_archers_0", "wh_main_emp_inf_handgunners"
SLINGS, SWORD = "wh2_main_skv_inf_skavenslave_slingers_0", "wh_main_emp_inf_swordsmen"
KEYS = [GUN, ARCHERS, HANDGUNS, SLINGS, SWORD]
OLD = ob.TOKEN - len(ob.AIM) - len(ob.RECEIVED)          # the token before AIM (167)


def test_the_aim_columns_by_unit():
    a = passport.aim(KEYS + [""])
    assert a[:, 0].tolist() == pytest.approx([0.45, 0.33, 0.2, 0.43, 0, 0])       # aim_s / 10
    assert a[:, 1].tolist() == pytest.approx([0.45, 0.85, 0, 0.85, 0, 0])         # silence after a new target / 10
    assert ob.NAMES[-len(ob.AIM) - len(ob.RECEIVED):-len(ob.RECEIVED)] == ob.AIM


def test_the_aim_columns_are_the_simulators_numbers_for_every_missile_unit():
    P = sim_params.load()
    ms = P.sim["missile"]
    keys = [k for k, u in passport.load().items() if u.get("missile")]
    a = passport.aim(keys)
    for k, (first, silence) in zip(keys, a):
        st = P.static(k)
        assert first * passport.AIM_S == pytest.approx(st["aim_s"]), k
        assert silence * passport.AIM_S == pytest.approx(
            sim_params.retarget_silence(st["aim_s"], st["rt_s"], st["quiet_s"], ms)), k
    gun = keys.index(GUN)
    others = [r for i, r in enumerate(a) if i != gun]
    assert all(tuple(r) != tuple(a[gun]) for r in others)                         # the gun stands apart


def test_retarget_silence_follows_the_simulators_rules():
    ms = {"aim_reset_on_order": 1, "retarget_s": 3.0}
    assert sim_params.retarget_silence(3.3, -1.0, 8.5, ms) == 8.5                  # arrows: the quiet window
    assert sim_params.retarget_silence(3.8, -1.0, 0.0, ms) == 3.8                  # the rule of all: the aim again
    assert sim_params.retarget_silence(2.0, 0.0, 0.0, ms) == 0.0                   # muskets: nothing
    assert sim_params.retarget_silence(4.5, 4.5, 0.0, ms) == 4.5                   # the Ratling Gun: its spin-up
    assert sim_params.retarget_silence(0, -1.0, 0.0, ms) == 0.0                    # no missile weapon
    assert sim_params.retarget_silence(3.8, -1.0, 0.0, {"retarget_s": 3.0}) == 3.0  # no reset on order: the switch


def test_the_token_carries_aim_on_both_sides():
    setup, state = sources.synthetic(batch=2, own=3, enemy=3, seed=5, keys=KEYS)
    obs, _ = ob.observe(state, setup, 1)
    cols = [ob.INDEX[n] for n in ob.AIM]
    for b, keys in enumerate(setup.keys):
        want = passport.aim(keys)
        assert np.allclose(obs.tokens[b][:, cols], want), keys                     # own (0-2) and enemy (3-5)
    assert not set(cols) & set(ob.OWN_ONLY)


def test_a_token_without_aim_maps_to_its_place():
    assert ob.old_columns(OLD) == list(range(OLD))
    assert ob.old_columns(ob.TOKEN) is None and ob.old_columns(OLD - 1) is None


def test_recorded_tokens_without_aim_widen_with_zeros():
    from tools.nn.observe.tokens import widen
    tok = np.random.default_rng(0).random((3, 4, OLD)).astype(np.float32)
    out = widen(tok)
    assert out.shape == (3, 4, ob.TOKEN) and np.array_equal(out[..., :OLD], tok) and not out[..., OLD:].any()


# --- torch (skipped in the project's .venv) ---

def test_a_network_saved_before_aim_acts_the_same():
    torch = pytest.importorskip("torch")
    from tools.nn.model import config, critic, heads, policy
    cfg = config.preset("small", d=64, layers=2, heads=4, critic_d=64, critic_layers=2, critic_heads=4)
    torch.manual_seed(3)
    old, old_crit = policy.Actor(cfg), critic.Critic(cfg)
    old.encoder.unit[0] = torch.nn.Linear(OLD, cfg.d)
    old_crit.encoder.unit[0] = torch.nn.Linear(OLD, cfg.critic_d)
    old, old_crit = old.eval(), old_crit.eval()
    new, new_crit = policy.Actor(cfg).eval(), critic.Critic(cfg).eval()
    new.load_state_dict(old.state_dict())
    new_crit.load(old_crit.state_dict())
    assert torch.all(new.encoder.unit[0].weight[:, OLD:] == 0)
    assert torch.all(new_crit.encoder.unit[0].weight[:, OLD:] == 0)
    setup, state = sources.synthetic(batch=2, own=4, enemy=4, seed=9, keys=KEYS)
    for k in range(3):
        for t in ("on", "cd"):
            state[f"ab{k}_{t}"] = np.zeros((2, 8))
    keys = ("tokens", "own", "attend", "pos", "ctx")
    m = mc = h_old = h_new = None
    for i in range(3):
        st = dict(state, t=np.full(2, 30.0 + i))
        obs, m = ob.observe(st, setup, 1, m)
        cobs, mc = ob.observe(st, setup, 1, mc, full=True)
        o, c = policy.to_torch(obs), policy.to_torch(cobs)
        assert o["tokens"][..., OLD:].abs().sum() > 0                             # the new inputs are not zero
        cut = lambda d: dict(d, tokens=d["tokens"][..., :OLD])
        with torch.no_grad():
            lo, h_old = old(cut(o), h_old)
            ln, h_new = new(o, h_new)
            vo = old_crit(cut({k: c[k] for k in keys}))
            vn = new_crit({k: c[k] for k in keys})
        for k in lo:
            assert torch.equal(lo[k], ln[k]), k
        assert torch.equal(h_old, h_new) and torch.equal(vo, vn)
        a, b = heads.sample(lo, greedy=True, abilities=True), heads.sample(ln, greedy=True, abilities=True)
        assert torch.equal(a.kind, b.kind) and torch.equal(a.target, b.target) and torch.equal(a.point, b.point)


def test_adams_moments_of_a_token_without_aim_widen(tmp_path):
    torch = pytest.importorskip("torch")
    from tools.nn.model.encoder import TokenEncoder
    from tools.nn.train import checkpoint, run
    old = TokenEncoder(8, n_token=OLD)
    opt = torch.optim.Adam(old.parameters())
    old(torch.randn(2, 3, OLD), torch.randn(2, ob.CONTEXT)).sum().backward()
    opt.step()
    new = TokenEncoder(8)
    new.load_state_dict(old.state_dict())
    new_opt = torch.optim.Adam(new.parameters())
    path = tmp_path / "m.pt"
    torch.save({"format": checkpoint.FORMAT, "train": {"optim": opt.state_dict(), "state": {"updates": 1}}}, path)
    assert run.resume(new_opt, path) == {"updates": 1}
    st = new_opt.state[new.unit[0].weight]
    want = opt.state[old.unit[0].weight]
    for k in ("exp_avg", "exp_avg_sq"):
        assert st[k].shape == (8, ob.TOKEN)
        assert torch.equal(st[k][:, :OLD], want[k]) and not st[k][:, OLD:].any()


def test_training_sees_the_cards_columns():
    """The training setup (scenes.LiveSetup: tensors of its FIELDS) gives the same token as the Setup the companion
    and recorded battles use: WEAPON and AIM included (WEAPON was missing from the fields: zero in training)."""
    torch = pytest.importorskip("torch")
    from tools.nn.train import scenes
    setup, state = sources.synthetic(batch=2, own=3, enemy=3, seed=4, keys=KEYS)
    live = scenes.LiveSetup.of(setup, "cpu")
    obs, _ = ob.observe(state, setup, 1)
    ts = {k: torch.as_tensor(v) if isinstance(v, np.ndarray) else v for k, v in state.items()}
    tobs, _ = ob.observe(ts, live, 1)
    cols = [ob.INDEX[n] for n in ob.WEAPON + ob.AIM]
    assert np.allclose(tobs.tokens[..., cols].numpy(), obs.tokens[..., cols], atol=1e-6)
    assert obs.tokens[..., cols].any()

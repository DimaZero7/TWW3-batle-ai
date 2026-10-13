"""The ability price (tools/nn/abilities_price): bonus bricks, abilities as data, the generic validator. numpy only."""
import json
from pathlib import Path

import numpy as np
import pytest

from tools.nn.abilities_price import ability as abl
from tools.nn.abilities_price import bonuses as bz
from tools.nn.abilities_price import validate as va

ROOT = Path(__file__).resolve().parents[2]
PASSPORTS = json.loads((ROOT / "config/nn/abilities.json").read_text(encoding="utf-8"))["abilities"]
SIM = json.loads((ROOT / "config/nn/sim.json").read_text(encoding="utf-8"))
RULES = bz.rules(json.loads((ROOT / "config/nn/game_rules.json").read_text(encoding="utf-8")), SIM)
BRICKS = bz.bricks(RULES)


def blank(S=40, U=4, active=20):
    """A quiet trajectory: everyone alive, standing, out of melee, fresh, morale at its top; the owner (0) active
    seconds 1..active."""
    S1 = S + 1
    z = np.zeros((S1, U), dtype=bool)
    tr = {"alive": ~z, "r": z.copy(), "m": z.copy(), "mv": z.copy(), "sprint": z.copy(),
          "charge": np.zeros((S1, U)), "mmax": np.full(U, 60.0), "morale": np.full((S1, U), 60.0),
          "fatigue": np.zeros((S1, U)), "x": np.zeros((S1, U)), "z": np.zeros((S1, U)),
          "on": np.clip(active + 1 - np.arange(S1), 0, None).astype(float), "owner": 0, "dt_s": 1.0}
    tr["on"][0] = 0.0
    tr["x"][:, 1:] = np.arange(1, U)[None, :] * 10.0          # unit h 10 h metres east of the owner
    mask = np.zeros((S1, U), dtype=bool)
    mask[1:active + 1, 0] = True
    return tr, mask


# --- layer 1 ---

def test_seconds_bricks_count_only_their_condition_inside_the_mask():
    tr, mask = blank()
    tr["m"][5:15, 0] = True             # 10 s in melee while on
    tr["m"][25:30, 0] = True            # 5 s after it ended: not counted
    assert BRICKS["melee_damage"].count(tr, mask, 1.25) == pytest.approx(25.0 * 10)
    tr["m"][5:10, 1] = True
    m2 = mask.copy()
    m2[1:21, 1] = True
    assert BRICKS["melee_defence"].count(tr, m2, 24.0) == pytest.approx(24.0 * 15)
    tr["mv"][1:8, 0] = True             # 7 s moving, 2 of them in melee (5, 6, 7 - 5..7 overlap)
    assert BRICKS["speed"].count(tr, mask, 1.25) == pytest.approx(25.0 * 4)
    tr["sprint"][2:5, 0] = True
    assert BRICKS["charge_speed"].count(tr, mask, 1.25) == pytest.approx(25.0 * 3)
    tr["alive"][8:, 0] = False          # a dead owner gets nothing
    assert BRICKS["melee_damage"].count(tr, mask, 1.25) == pytest.approx(25.0 * 3)


def test_charge_counts_impacts_while_on():
    tr, mask = blank()
    c = np.zeros(len(tr["on"]))
    for s0 in (3, 12, 30):              # three charges, the third after the ability ended
        c[s0:] = np.clip(1 - (np.arange(len(c) - s0)) / 13.0, 0, 1)
    tr["charge"][:, 0] = c
    assert BRICKS["charge"].count(tr, mask, 1.6) == pytest.approx(60.0 * 2)


def test_morale_counts_the_shortfall_at_most_the_bonus_and_nothing_at_the_top():
    tr, mask = blank(U=3)
    m2 = mask.copy()
    m2[1:21, :] = True
    lift = bz.morale_lift(m2, 16.0, 1.0, RULES)
    tr["morale"] = tr["mmax"][None, :] + lift            # everyone at the top: the lift is the bonus's own
    assert BRICKS["morale"].count(tr, m2, 16.0) == pytest.approx(0.0, abs=1e-9)
    tr["morale"][:, 1] -= 10.0                           # 10 short all along -> 10
    tr["morale"][:, 2] -= np.linspace(0, 40, len(tr["on"]))   # the shortfall grows past the bonus -> 16
    assert BRICKS["morale"].count(tr, m2, 16.0) == pytest.approx(10.0 + 16.0)
    # the lift runs at the morale's pace: max(1, 15 % of the gap) a tick, two ticks a second
    assert lift[1, 0] == pytest.approx(16 * 0.15 + (16 - 2.4) * 0.15)


def test_vigour_counts_the_seconds_a_tier_fresher_while_busy():
    # Foe-Seeker: -1 % of the maximum a second for 25 s; the owner fights 40 s; recorded fatigue 0 while it is on (cut
    # off at 0: rebuilt at the combat rate, 19 a tick x 10), then up 190 a second. Without it: 190 s from the start.
    S = 40
    tr, mask = blank(S=S, active=25)
    tr["m"][1:, 0] = True
    s = np.arange(S + 1)
    tr["fatigue"][:, 0] = np.where(s <= 25, 0.0, 190.0 * (s - 25))
    # tiers apart from s = 15 (2850 >= 2800 without it) to 40
    assert BRICKS["vigour"].count(tr, mask, -0.01) == pytest.approx(26.0)
    assert BRICKS["fatigue_cost"].count(tr, mask, -0.01) == 0.0          # a recovery is no cost
    tr["m"][:, 0] = False
    assert BRICKS["vigour"].count(tr, mask, -0.01) == 0.0                # idle: the tier does nothing


def test_fatigue_cost_counts_the_seconds_a_tier_more_tired():
    S = 40
    tr, mask = blank(S=S, active=25)
    tr["mv"][1:, 0] = True
    s = np.arange(S + 1)
    tr["fatigue"][:, 0] = np.minimum(s, 25) * 300.0 + 2000.0        # the drain alone: 300 a second for 25 s
    n = BRICKS["fatigue_cost"].count(tr, mask, 0.01)
    assert n > 0 and BRICKS["vigour"].count(tr, mask, 0.01) == 0.0


def test_every_brick_declares_its_kind_and_stats():
    for kind, b in BRICKS.items():
        assert b.desc.kind == kind and b.desc.effect in ("buff", "debuff") and b.desc.how in ("mult", "add", "rate")
        assert b.desc.stats


# --- layer 2 ---

def test_our_abilities_are_data_with_every_stat_priced():
    a = abl.abilities(PASSPORTS, BRICKS)
    assert set(a) == {"wh2_main_character_abilities_verminous_valour", "wh_main_character_abilities_deadly_onslaught",
                      "wh_main_character_abilities_foe_seeker", "wh_main_character_abilities_rally",
                      "wh_main_character_abilities_stand_your_ground"}
    parts = {k: {(p.kind, round(p.value, 4), p.groups) for p in v.parts} for k, v in a.items()}
    assert parts["wh_main_character_abilities_deadly_onslaught"] == {("charge", 1.6, ("self",)),
                                                                     ("melee_damage", 1.25, ("self",))}
    assert parts["wh_main_character_abilities_foe_seeker"] == {("charge_speed", 1.25, ("self",)),
                                                               ("speed", 1.25, ("self",)), ("vigour", -0.01, ("self",))}
    assert parts["wh_main_character_abilities_stand_your_ground"] == {("melee_defence", 24.0, ("self", "friends")),
                                                                      ("morale", 16.0, ("self", "friends"))}
    assert parts["wh_main_character_abilities_rally"] == {("morale", 16.0, ("self", "friends"))}
    assert parts["wh2_main_character_abilities_verminous_valour"] == {("speed", 1.25, ("self",)),
                                                                      ("morale", 8.0, ("self",))}
    assert all(not v.unpriced for v in a.values())
    assert a["wh_main_character_abilities_rally"].update_targets
    assert not a["wh_main_character_abilities_stand_your_ground"].update_targets


def test_a_new_stat_is_listed_unpriced_and_a_half_damage_bonus_is_half():
    p = dict(PASSPORTS["wh_main_character_abilities_deadly_onslaught"])
    p["effects"] = [{"stat": "stat_melee_damage_base", "how": "mult", "value": 1.5, "on": ["self"]},
                    {"stat": "stat_armour", "how": "add", "value": -30, "on": ["enemies"]}]
    a = abl.Ability.from_passport("x", p, BRICKS)
    assert [(q.kind, q.value) for q in a.parts] == [("melee_damage", 1.25)]
    assert a.unpriced and "stat_armour" in a.unpriced[0]


def test_masks_follow_the_aura_or_keep_the_cast():
    tr, _ = blank(U=4, active=14)             # units 10, 20, 30 m from the owner
    tr["x"][:, 3] = 50.0                      # unit 3 out of range (35 m) ...
    tr["x"][10:, 3] = 5.0                     # ... comes within it at second 10
    tr["x"][5:, 1] = 100.0                    # unit 1 leaves at second 5
    rally = abl.abilities(PASSPORTS, BRICKS)["wh_main_character_abilities_rally"]
    m = rally.masks(tr)
    assert m["self"][1:15, 0].all() and not m["self"][15:, 0].any() and not m["self"][0].any()
    assert m["friends"][1:5, 1].all() and not m["friends"][5:, 1].any()          # it follows: unit 1 drops out
    assert m["friends"][10:15, 3].all() and not m["friends"][1:10, 3].any()     # unit 3 comes in
    syg = abl.abilities(PASSPORTS, BRICKS)["wh_main_character_abilities_stand_your_ground"]
    tr["on"] = np.clip(19 - np.arange(len(tr["on"])), 0, None).astype(float)
    tr["on"][0] = 0
    m = syg.masks(tr)
    assert m["friends"][1:19, 1].all() and not m["friends"][19:, 1].any()       # laid at the cast: kept 18 s
    assert not m["friends"][:, 3].any()                                          # came later: nothing
    tr["on"][:] = 0                                                              # the press did not fire
    assert not syg.masks(tr)["friends"].any()


def test_price_is_benefit_less_debuffs_and_recharge():
    a = abl.abilities(PASSPORTS, BRICKS)
    rates = {"speed": 0.1, "vigour": 2.0, "charge_speed": 0.0}
    rc = abl.Recharge.fit({"wh_main_character_abilities_foe_seeker": [10.0, 30.0, -5.0, 0.0]}, a)
    assert rc.cost("wh_main_character_abilities_foe_seeker") == pytest.approx(20.0 * 60 / 60)
    p = abl.price(a["wh_main_character_abilities_foe_seeker"], {"speed": 100.0, "vigour": 5.0}, rates, BRICKS, rc)
    assert p["benefit"] == pytest.approx(20.0) and p["price"] == pytest.approx(0.0)
    p = abl.price(a["wh_main_character_abilities_foe_seeker"], {"speed": 100.0, "fatigue_cost": 5.0},
                  dict(rates, fatigue_cost=1.0), BRICKS, rc)
    assert p["debuffs"] == pytest.approx(5.0)


# --- layer 3 ---

def test_the_generic_tests_pass_for_every_brick():
    bad = [c for c in va.auto_tests(BRICKS) if not c.ok]
    assert not bad, bad


class Leaky(bz.MeleeDamage):
    """Counts melee outside the mask too."""
    def count(self, tr, mask, value):
        return float(bz.size("mult", value) * (tr["m"] | ~tr["m"]).sum())


class Capped(bz.Speed):
    """Declared proportional, but capped."""
    def count(self, tr, mask, value):
        return min(50.0, super().count(tr, mask, value))


def test_the_generic_tests_catch_a_broken_brick():
    res = {(c.brick, c.test): c.ok for c in va.auto_tests({"melee_damage": Leaky(), "speed": Capped()})}
    assert not res[("melee_damage", "off")]
    assert not res[("speed", "double")]


def test_spearman_with_ties_and_the_rates_come_back():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 5, 200).astype(float)
    b = a + rng.normal(0, 1, 200)
    scipy = pytest.importorskip("scipy.stats")
    assert va.spearman(a, b) == pytest.approx(scipy.spearmanr(a, b).correlation, abs=1e-9)
    kinds = ["morale", "speed", "fatigue_cost"]
    C = rng.uniform(0, 10, (400, 3)) * (rng.random((400, 3)) < 0.6)
    y = C @ np.array([3.0, 0.5, -2.0]) + rng.normal(0, 0.5, 400)
    r = va.fit_rates(C, y, kinds, BRICKS, boot=50)
    assert r["morale"]["k"] == pytest.approx(3.0, abs=0.1) and r["speed"]["k"] == pytest.approx(0.5, abs=0.1)
    assert r["fatigue_cost"]["k"] == pytest.approx(2.0, abs=0.1)          # a debuff: its rate with a minus
    assert r["morale"]["lo"] < 3.0 < r["morale"]["hi"]
    pred = va.crossfit(C, y, kinds, BRICKS)
    assert np.corrcoef(pred, y)[0, 1] > 0.95


def _dataset(n, noise, seed=0, wrong=False):
    """Presses of Rally and Foe-Seeker on random trajectories, the branch price = the true rates x counters + noise
    (wrong: a price of its own per press, the same in both copies - real, but not what the bricks count)."""
    rng = np.random.default_rng(seed)
    names = ["wh_main_character_abilities_foe_seeker", "wh_main_character_abilities_rally"]
    a = abl.abilities(PASSPORTS, BRICKS)
    true = {"speed": 0.5, "charge_speed": 0.2, "vigour": 3.0, "morale": 2.0}
    cols = {k: [] for k in ("x", "z", "alive", "r", "m", "mv", "sprint", "charge", "morale", "fatigue", "mmax", "on")}
    key, owner, diff = [], [], []
    for p in range(n):
        k = p % 2
        traces = [va.random_trace(rng, S=30, U=6, active=a[names[k]].active_s) for _ in range(2)]
        g = []
        for tr, _ in traces:
            tr["on"] = np.clip(a[names[k]].active_s + 1 - np.arange(31), 0, None)
            tr["on"][0] = 0
            for f in cols:
                cols[f].append(tr[f])
            g.append(sum(true[kd] * c for kd, c in a[names[k]].counters(tr, BRICKS).items()))
        key.append(k)
        owner.append(0)
        own = rng.normal(0, 0.05)
        diff.append(np.full(2, own) if wrong else np.array(g) / 1000.0 + rng.normal(0, noise, 2))
    data = {f: np.stack(v).reshape(n, 2, *np.shape(v[0])) for f, v in cols.items()}
    data.update(key=np.array(key), owner=np.array(owner), diff=np.array(diff), budget=np.full((n, 2), 1000.0),
                t=rng.uniform(0, 600, n), pressed=np.ones(n, dtype=bool), other=np.zeros(n))
    return data, names, a


@pytest.mark.parametrize("noise,wrong,verdict", [(0.002, False, "pass"), (5.0, False, "no signal"),
                                                   (0.0, True, "FAIL")])
def test_the_card_passes_a_true_price_fails_a_wrong_one_and_cannot_judge_noise(noise, wrong, verdict):
    data, names, a = _dataset(120, noise, wrong=wrong)
    kinds, C, _ = va.counters(data, names, a, BRICKS)
    y = data["diff"].mean(1) * data["budget"].mean(1)
    rates = va.fit_rates(C, y, kinds, BRICKS, boot=20)
    pred = va.crossfit(C, y, kinds, BRICKS)
    rows = va.card(data, names, a, BRICKS, kinds, C, rates, pred, PASSPORTS, SIM["abilities"])
    assert all(r["verdict"] == verdict for r in rows.values()), {k: (r["sign"], r["spearman"], r["rel"])
                                                                 for k, r in rows.items()}
    if verdict == "pass":
        assert rates["morale"]["k"] == pytest.approx(2.0, rel=0.1)

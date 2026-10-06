"""The simulator-vs-game forecast scores (tools/nn/simskill.py) on cases where the answer is known."""
import math

import numpy as np
import pytest

from tools.nn import simskill as S


def normal_crps(mu, sd, y):
    """CRPS of N(mu, sd) for y (Gneiting & Raftery 2007, closed form)."""
    z = (y - mu) / sd
    pdf = math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return sd * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))


def family(n, m, rng, spread=1.0, shift=0.0, quantities=3):
    """n battles x `quantities` quantities: the game and m copies drawn from N(mu, 1), mu differing per case; the
    copies' spread scaled by `spread` and moved by `shift`."""
    cases = []
    for b in range(n):
        for q in range(quantities):
            mu = rng.normal(0, 3)
            y = rng.normal(mu, 1)
            x = rng.normal(mu + shift, spread, m)
            cases.append(S.case(b, f"q{q}", y, x))
    return S.score(cases)


def test_calibrated_forecaster_covers_90_percent():
    s = S.summary(family(400, 19, np.random.default_rng(1)), boot=300)
    cov, lo, hi = s["coverage"]
    assert abs(cov - 0.9) < 0.02 and lo < 0.9 < hi
    assert abs(s["below"][0] - 0.05) < 0.015 and abs(s["above"][0] - 0.05) < 0.015
    assert s["crpss"][0] > 0.5          # knows the case's centre, climatology does not
    assert abs(s["spread_median"] - 1) < 0.1


def test_coverage_is_90_percent_for_any_number_of_copies():
    for m in (3, 8):
        s = S.summary(family(800, m, np.random.default_rng(m)), boot=50)
        assert abs(s["coverage"][0] - 0.9) < 0.02


def test_too_narrow_or_biased_copies_cover_less():
    rng = np.random.default_rng(2)
    narrow = S.summary(family(400, 19, rng, spread=0.3), boot=50)
    assert narrow["coverage"][0] < 0.6
    assert narrow["spread_median"] < 0.4                             # the copies spread ~0.3 of their miss
    assert abs(narrow["below"][0] - narrow["above"][0]) < 0.05      # on both sides
    high = S.summary(family(400, 19, rng, shift=2.0), boot=50)
    assert high["coverage"][0] < 0.7 and high["below"][0] > 0.3 > high["above"][0]  # sim too high: game below


def test_wide_copies_cover_more_but_lose_skill():
    rng = np.random.default_rng(3)
    good = S.summary(family(400, 19, rng), boot=50)
    wide = S.summary(family(400, 19, rng, spread=3.0), boot=50)
    assert wide["coverage"][0] > 0.95
    assert wide["crpss"][0] < good["crpss"][0] - 0.1            # the proper score sees through it


def test_fair_crps_is_unbiased_for_small_ensembles():
    rng = np.random.default_rng(4)
    for m in (2, 8):
        fair = [S.crps(rng.normal(0, 1, m), 0.7) for _ in range(20000)]
        assert abs(np.mean(fair) - normal_crps(0, 1, 0.7)) < 0.01
    plain = [S.crps(rng.normal(0, 1, 2), 0.7, fair=False) for _ in range(20000)]
    assert np.mean(plain) > normal_crps(0, 1, 0.7) + 0.05       # the plain ensemble CRPS punishes a small one


def test_binary_crps_is_brier_and_ties_split_the_rank():
    assert S.crps([1, 1, 0, 0], 1, fair=False) == pytest.approx((1 - 0.5) ** 2)
    # every copy equal to the game: the rank is uniform over all m + 1 places, 90 % inside
    inside, below, above = S.pit([1] * 19, 1)
    assert (inside, below, above) == pytest.approx((0.9, 0.05, 0.05))
    # the game above every copy of 19: outside, above
    assert S.pit([0] * 19, 1) == pytest.approx((0.0, 0.0, 1.0))
    # a value strictly inside the range of 19 copies
    assert S.pit(np.arange(19), 9.5) == pytest.approx((1.0, 0.0, 0.0))


def test_bootstrap_draws_whole_battles_and_skips_missing():
    rng = np.random.default_rng(5)
    cases = family(30, 19, rng) + [S.case(99, "q0", None, [1, 2]), S.case(98, "q0", 1.0, [None, 2.0])]
    s = S.summary(cases, boot=100)
    assert s["battles"] == 30 and s["cases"] == 90
    assert s["coverage"][1] <= s["coverage"][0] <= s["coverage"][2]

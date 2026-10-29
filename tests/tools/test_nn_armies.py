"""tools.nn.armies: pools, the budget market, random battles, their deployment and consumers."""
import json
import math

import numpy as np
import pytest

from tools.nn import scenario as nn_scenario
from tools.nn.armies import export
from tools.nn.armies import generate as G
from tools.nn.armies import place as PL
from tools.nn.armies import pools as P

POOLS = P.load()
EMP, SKV = "wh_main_emp_empire", "wh2_main_skv_skaven"
SEEDS = range(300)
# The Skaven's gold relative to the Empire's (config/nn/pools.json budget_factor; 0.8 from 01.10.2026,
# back to 1.0 on 02.10.2026): read from the config, not fixed here.
SKV_SHARE = POOLS[SKV].budget_factor / max(POOLS[EMP].budget_factor, POOLS[SKV].budget_factor)
EMP_SHARE = POOLS[EMP].budget_factor / max(POOLS[EMP].budget_factor, POOLS[SKV].budget_factor)


@pytest.fixture(scope="module")
def battles():
    return [G.battle(s) for s in SEEDS]


def units_of(side):
    return [u for u in side["units"] if not u.get("general")]


class TestPools:
    def test_every_pool_unit_has_a_passport_cost_and_a_family(self):
        passports = json.loads(P.PASSPORTS.read_text(encoding="utf-8"))["units"]
        for pool in POOLS.values():
            for u in (pool.lord,) + pool.units:
                assert u.key in passports and u.cost == passports[u.key]["multiplayer_cost"] > 0
            assert all(u.family for u in pool.units) and pool.templates

    def test_cap_counts_the_lord_rounds_down_and_allows_one(self):
        pool = POOLS[EMP]
        assert pool.cap("inf_ranged", 0) == 1 and pool.cap("inf_ranged", 1) == 1
        assert pool.cap("inf_ranged", 4) == 3 and pool.cap("inf_ranged", 19) == 12
        assert pool.cap("no_such_category", 5) is None

    def test_depth_of_a_formation(self):
        assert P.depth(120, 30) == 9.0 and P.depth(180, 30) == 13.5 and P.depth(1, 5) == 0.0

    def test_family_shares_sum_by_root_and_split_evenly(self):
        parents = {"a_high": "a", "a": "a_trash", "b_low": "b"}
        shares = P.family_shares({"a_high": 30, "a": 10, "b_low": 20, "artillery": 40},
                                 ["a_trash", "a_trash", "b"], parents)
        assert shares == pytest.approx((1 / 3, 1 / 3, 1 / 3))
        assert P.family_shares({"naval": 5}, ["a_trash", "b"], parents) is None
        for pool in POOLS.values():
            for t in pool.templates:
                assert t.weight > 0 and sum(t.shares) == pytest.approx(1.0)

    def test_group_asks_go_to_the_subtree_then_up_the_parents(self):
        # a: a_high, a_low under it; b alone. The template asks a (no unit of its own: its subtree), a_high (no pool
        # unit: up to a), b, and artillery (no unit anywhere: dropped).
        parents = {"a_high": "a", "a_low": "a", "a": "root"}
        asks = P.group_asks({"a": 30, "a_high": 10, "b": 20, "artillery": 40}, [["a_low"], ["a_low"], ["b"]], parents)
        assert asks == ((30.0, ((0, 1), (0, 1))), (10.0, ((0, 1), (0, 1))), (20.0, ((2,),)))
        t = P.Template("x", 1.0, (), asks)
        assert list(t.weights(np.ones(3, bool))) == [20, 20, 20]
        assert list(t.weights(np.array([False, True, False]))) == [0, 40, 0]     # the group keeps its ratio

    def test_the_game_templates_reach_the_high_quality_groups(self):
        """Under the group rule the stormvermin take the templates' high-quality ratios (spears 15 / swords 10 of
        WH_Skaven_land), the Empire's greatswords the swords' high-quality ratio; the skavenslaves sit at the root of
        the melee tree, which no Skaven template asks."""
        assert json.loads(P.POOLS.read_text(encoding="utf-8"))["template_rule"] == "group"
        skv = {t.name: dict(zip([u.slot for u in POOLS[SKV].units], t.shares)) for t in POOLS[SKV].templates}
        land = skv["WH_Skaven_land"]
        assert land["stormvermin"] > land["stormvermin_shield"] > land["clanrat"] > 0
        assert all(t["slave"] == 0 for t in skv.values())
        emp = {t.name: dict(zip([u.slot for u in POOLS[EMP].units], t.shares)) for t in POOLS[EMP].templates}
        assert emp["WH_Empire_land"]["greatsword"] == max(emp["WH_Empire_land"].values())

    def test_the_family_rule_is_kept_as_an_option(self, tmp_path):
        doc = json.loads(P.POOLS.read_text(encoding="utf-8"))
        doc["template_rule"] = "family"
        path = tmp_path / "pools.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        for pool in P.load(path).values():
            fam = [u.family for u in pool.units]
            for t in pool.templates:       # even within a family
                for f in set(fam):
                    vals = {round(s, 9) for s, g in zip(t.shares, fam) if g == f}
                    assert len(vals) == 1

    def test_the_mix_is_three_quarters_templates(self):
        assert P.mix() == {"template": 0.75, "random": 0.25}

    def test_budget_factors_are_the_configs(self):
        doc = json.loads(P.POOLS.read_text(encoding="utf-8"))["factions"]
        for faction, pool in POOLS.items():
            assert pool.budget_factor == float(doc[faction].get("budget_factor", 1.0)) > 0


class TestMarket:
    @pytest.mark.parametrize("faction", [EMP, SKV])
    def test_every_spendable_budget_is_spent_inside_the_window(self, faction):
        m = G.Market(POOLS[faction])
        rng = np.random.default_rng(0)
        pool = POOLS[faction]
        for budget in np.linspace(m.totals.min(), m.totals.max(), 60):
            if not m.can_spend(budget):
                continue
            for shares in list(pool.templates) + [t.shares for t in pool.templates] + [None]:
                bought = m.template_army(rng, budget, shares) if shares else m.random_army(rng, budget)
                cost = pool.lord.cost + sum(pool.units[i].cost for i in bought)
                assert budget * (1 - G.TOLERANCE) - 1e-6 <= cost <= budget + 1e-6
                assert len(bought) <= G.MAX_UNITS

    def test_template_armies_keep_the_caps(self):
        m, pool = G.Market(POOLS[EMP]), POOLS[EMP]
        rng = np.random.default_rng(1)
        for budget in np.linspace(1000, 6900, 40):
            if m.can_spend(budget):
                archers = tuple(float(u.category == "inf_ranged") for u in pool.units)
                bought = m.template_army(rng, budget, archers)      # a template of archers only
                n_missile = sum(pool.units[i].category == "inf_ranged" for i in bought)
                assert n_missile <= pool.cap("inf_ranged", len(bought))

    def test_a_random_army_has_the_drawn_number_of_units(self):
        m = G.Market(POOLS[SKV])
        counts = m.counts(3000)
        assert min(counts) <= 8 and max(counts) >= 15      # few clanrats .. many slaves
        rng = np.random.default_rng(2)
        sizes = {len(m.random_army(rng, 3000)) for _ in range(200)}
        assert sizes == set(counts)


class TestBattles:
    def test_each_side_spends_its_budget_within_five_percent(self, battles):
        for a in battles:
            for side in a["sides"].values():
                assert side["budget"] * 0.95 - 1 <= side["cost"] <= side["budget"] + 1
                assert side["budget"] <= a["budget"] + 1
            costs = [a["sides"][s]["cost"] for s in ("own", "enemy")]
            if len({s["faction"] for s in a["sides"].values()}) == 1:     # a mirror: equal budgets
                assert a["sides"]["own"]["budget"] == a["sides"]["enemy"]["budget"] == a["budget"]
                assert abs(costs[0] - costs[1]) <= 0.05 * max(costs) + 1e-9

    def test_budget_ratio_per_faction_pair(self, battles):
        """Skaven spend their budget_factor of the Empire's budget; mirrors spend equal budgets."""
        r = SKV_SHARE / EMP_SHARE
        expect = {(EMP, EMP): 1.0, (SKV, SKV): 1.0, (EMP, SKV): r, (SKV, EMP): 1 / r}
        ratios = {}
        for a in battles:
            own, enemy = a["sides"]["own"], a["sides"]["enemy"]
            pair = (own["faction"], enemy["faction"])
            assert enemy["budget"] / own["budget"] == pytest.approx(expect[pair], abs=0.002)
            ratios.setdefault(pair, []).append(enemy["cost"] / own["cost"])
            if EMP in pair and EMP_SHARE == 1.0:
                assert a["sides"]["own" if own["faction"] == EMP else "enemy"]["budget"] == a["budget"]
        for pair, r in ratios.items():
            lo, hi = expect[pair] * 0.95, expect[pair] / 0.95
            assert all(lo - 1e-3 <= x <= hi + 1e-3 for x in r), pair
            assert np.mean(r) == pytest.approx(expect[pair], rel=0.03), pair

    def test_the_skaven_factor_scales_only_the_skaven_side(self):
        rng = np.random.default_rng(5)
        for _ in range(30):
            a = G.generate(rng, sides=(EMP, SKV))
            emp, skv = a["sides"]["own"], a["sides"]["enemy"]
            assert abs(emp["budget"] - EMP_SHARE * a["budget"]) <= 1
            assert abs(skv["budget"] - SKV_SHARE * a["budget"]) <= 1

    def test_a_lord_and_at_most_nineteen_units_from_the_pool(self, battles):
        for a in battles:
            for side in a["sides"].values():
                pool = POOLS[side["faction"]]
                generals = [u for u in side["units"] if u.get("general")]
                assert len(generals) == 1 and generals[0]["key"] == pool.lord.key
                assert 1 <= len(side["units"]) <= 20
                keys = {u.key for u in pool.units}
                assert all(u["key"] in keys for u in units_of(side))
                cost = pool.lord.cost + sum(next(p.cost for p in pool.units if p.key == u["key"])
                                            for u in units_of(side))
                assert cost == side["cost"]

    def test_the_same_seed_gives_the_same_battle(self):
        assert G.battle(7) == G.battle(7)
        assert G.battle(7) != G.battle(8)

    def test_train_and_eval_seeds_are_disjoint(self):
        assert G.TRAIN_SEEDS.stop <= G.EVAL_SEEDS.start
        assert G.split(0) == "train" and G.split(G.EVAL_SEEDS.start) == "eval"
        with pytest.raises(ValueError):
            G.split(-1)

    def test_mirrors_cross_matchups_templates_and_random_armies_all_occur(self, battles):
        pairs = {(a["sides"]["own"]["faction"], a["sides"]["enemy"]["faction"]) for a in battles}
        assert pairs == {(EMP, EMP), (EMP, SKV), (SKV, EMP), (SKV, SKV)}
        kinds = [s["army"] == "random" for a in battles for s in a["sides"].values()]
        assert 0.15 < np.mean(kinds) < 0.35

    def test_template_armies_keep_the_caps(self, battles):
        for a in battles:
            for side in a["sides"].values():
                if side["army"] != "random":
                    pool = POOLS[side["faction"]]
                    missile = {u.key for u in pool.units if u.category == "inf_ranged"}
                    n = len(units_of(side))
                    assert sum(u["key"] in missile for u in units_of(side)) <= pool.cap("inf_ranged", n)

    def test_shapes_vary_many_cheap_against_few_elite(self, battles):
        ratio = [max(len(units_of(s)) for s in a["sides"].values())
                 / max(1, min(len(units_of(s)) for s in a["sides"].values())) for a in battles]
        # Cheap Skaven against fewer Empire units: ~7% of battles over 1.5 at 0.8 of the gold, more at 1.0.
        assert np.mean(np.asarray(ratio) >= 1.5) > 0.03 and max(ratio) >= 2

    def test_a_budget_range_a_unit_limit_and_one_faction(self):
        rng = np.random.default_rng(3)
        for _ in range(50):
            a = G.generate(rng, factions=[SKV], budget_range=(1000, 1500), max_units=5)
            assert all(s["faction"] == SKV for s in a["sides"].values())
            assert 1000 <= a["budget"] <= 1500
            assert all(len(s["units"]) <= 6 for s in a["sides"].values())
        with pytest.raises(AssertionError):
            G.generate(rng, factions=[EMP], budget_range=(100, 200))

    def test_the_budget_never_exceeds_what_both_can_field(self):
        gen = G.default()
        lo, hi = gen.budget_bounds((SKV,))
        # the cheapest Skaven unit: plain skavenslaves (125); the Skaven's own cap (pools.json faction budget_max,
        # 02.10.2026): the Warlord and 19 clanrat spearmen, their most before the Night Runners
        assert lo == POOLS[SKV].lord.cost + 125 and hi == POOLS[SKV].budget_max == POOLS[SKV].lord.cost + 19 * 325
        # The Empire's most: 19 units, at most 12 of them archers (the cap), the rest its dearest melee.
        emp = POOLS[EMP]
        melee = max(u.cost for u in emp.units if u.category != "inf_ranged")
        archer = max(u.cost for u in emp.units if u.category == "inf_ranged")
        emp_hi = emp.lord.cost + 12 * max(melee, archer) + 7 * melee
        # ... capped at pools.json budget_max (02.10.2026: the budgets stay where they were before the
        # greatswords, 7725)
        cap = P.budget_max()
        assert gen.budget_bounds((EMP,)) == (emp.lord.cost + min(u.cost for u in emp.units),
                                             emp_hi if cap is None else min(emp_hi, cap))
        assert cap is None or G.Generator(pools=POOLS).budget_bounds((EMP,))[1] == emp_hi
        # Against the Empire each side gets its share of B: B up to what both can field.
        lo, hi = gen.budget_bounds((EMP, SKV))
        skv_hi = POOLS[SKV].lord.cost + 19 * 325
        assert hi == pytest.approx(min(emp_hi / EMP_SHARE, skv_hi / SKV_SHARE))
        assert lo == pytest.approx(max((emp.lord.cost + 300) / EMP_SHARE, (POOLS[SKV].lord.cost + 150) / SKV_SHARE))
        assert gen.shares((EMP, SKV)) == {EMP: EMP_SHARE, SKV: SKV_SHARE} and gen.shares((SKV, SKV)) == {SKV: 1.0}


class TestPlace:
    @staticmethod
    def check(units, pools_units, deployment_m=300.0):
        depth = {u.key: u.depth for u in pools_units}
        boxes = [PL.footprint(u, depth.get(u["key"], 0.0)) for u in units]
        for f0, f1, l0, l1 in boxes:
            assert f0 >= PL.ZONE_FRONT_M - deployment_m and f1 <= PL.ZONE_FRONT_M
            assert l0 >= -deployment_m / 2 and l1 <= deployment_m / 2
        for i, a in enumerate(boxes):
            for b in boxes[i + 1:]:
                assert a[1] <= b[0] or b[1] <= a[0] or a[3] <= b[2] or b[3] <= a[2], (a, b)

    @pytest.mark.parametrize("faction", [EMP, SKV])
    def test_one_to_twenty_units_fit_the_zone_without_overlaps(self, faction):
        pool = POOLS[faction]
        rng = np.random.default_rng(4)
        for n in range(0, 20):
            units = [pool.units[i] for i in rng.integers(len(pool.units), size=n)]
            placed = PL.place(pool.lord, units)
            assert len(placed) == n + 1 and placed[0]["general"]
            assert len({u["slot"] for u in placed}) == n + 1
            self.check(placed, (pool.lord,) + pool.units)

    def test_melee_in_front_missile_behind_the_lord_last(self):
        pool = POOLS[SKV]
        units = [pool.units[0]] * 9 + [pool.units[2]] * 7
        placed = PL.place(pool.lord, units)
        melee = [u["forward"] for u in placed if u["slot"].startswith("clanrat")]
        missile = [u["forward"] for u in placed if u["slot"].startswith("sling")]
        assert max(melee) == 0.0 and min(melee) > max(missile)
        assert placed[0]["forward"] < min(missile)

    def test_generated_battles_fit(self, battles):
        for a in battles[:100]:
            for side in a["sides"].values():
                pool = POOLS[side["faction"]]
                self.check(side["units"], (pool.lord,) + pool.units, a["deployment_m"])


class TestConsumers:
    def test_the_game_battle_file_takes_a_generated_battle(self, battles):
        for a in battles[:20]:
            xml = export.scenario_xml(a)
            n = sum(len(s["units"]) for s in a["sides"].values())
            assert xml.count("<unit ") == n and xml.count("<general>") == 2
            config = nn_scenario.run_config(a)
            assert len(config["units"]["own"]) == len(a["sides"]["own"]["units"])

    def test_an_arenas_file_loads_as_named_arenas(self, battles, tmp_path):
        path = export.write_arenas(battles[:5], tmp_path / "arenas.json")
        for a in battles[:5]:
            loaded = nn_scenario.load_arena(a["name"], arenas_path=path)
            assert nn_scenario.placements(loaded) == nn_scenario.placements(a)

    def test_the_simulator_takes_generated_battles(self, battles):
        pytest.importorskip("torch")
        from tools.nn.sim import scenario as sim_scenario
        armies = export.to_sim(battles[:4], "attack")
        st = sim_scenario.build(armies, per_side=G.MAX_UNITS + 1)
        for b, a in enumerate(battles[:4]):
            assert int((st.u["side"][b] == 1).sum()) == len(a["sides"]["own"]["units"])
            assert int((st.u["side"][b] == 2).sum()) == len(a["sides"]["enemy"]["units"])
        assert math.isfinite(float(st.u["x"].abs().max()))


class TestRichBattles:
    def test_by_default_no_rich_battle_and_no_extra_draw(self):
        assert P.budget_rare() == (0.0, None)
        assert G.Generator().rare == (0.0, None)

    def test_a_rich_share_draws_above_the_usual_top(self, tmp_path):
        doc = json.loads(P.POOLS.read_text(encoding="utf-8"))
        doc["budget_rare"] = {"share": 0.5, "max": 12400}
        path = tmp_path / "pools.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        assert P.budget_rare(path) == (0.5, 12400.0)
        gen = G.Generator()
        gen.rare = P.budget_rare(path)
        lo, hi = gen.budget_bounds((EMP, SKV))
        rng = np.random.default_rng(3)
        budgets = [gen.generate(rng, sides=(EMP, SKV))["budget"] for _ in range(60)]
        rich = [b for b in budgets if b > hi + 1]
        assert 10 <= len(rich) <= 50 and max(budgets) <= 12400 + 1


# The old scheme's armies of a few train and evaluation seeds (the budget, then per side the template, the cost and the
# unit keys), hashed before the v2 scheme joined the generator (11.10.2026): v1 must give them bit for bit. A deliberate
# change of the old scheme (pools.json, the templates) changes it; then the evaluation's battles change too.
V1_SEEDS = (0, 1, 2, 3, 1_000_000_000, 1_000_900_795)
V1_HASH = "0d3626c6a39c54a47b542a111de4190cd9db1f5ce6d69c3af3f723b649969f22"


def composition(arena):
    return [arena["budget"]] + [[arena["sides"][s]["army"], arena["sides"][s]["cost"],
                                 [u["key"] for u in arena["sides"][s]["units"]]] for s in ("own", "enemy")]


@pytest.fixture(scope="module")
def battles_v2():
    return [G.battle(s, armies="v2") for s in range(1500)]


class TestArmiesV2:
    def test_the_old_scheme_is_the_default_and_unchanged(self):
        import hashlib
        arenas = [G.battle(s) for s in V1_SEEDS]
        assert hashlib.sha256(json.dumps([composition(a) for a in arenas]).encode()).hexdigest() == V1_HASH
        assert all(G.battle(s, armies="v1") == a for s, a in zip(V1_SEEDS, arenas))
        assert "armies" not in arenas[0] and "ceiling" not in arenas[0]

    def test_the_same_seed_the_same_v2_battle_another_than_v1(self):
        a = G.battle(17, armies="v2")
        assert a == G.battle(17, armies="v2") and a["armies"] == "v2"
        assert composition(a) != composition(G.battle(17))
        with pytest.raises(AssertionError):
            G.battle(17, armies="v4")

    def test_the_count_bands_by_their_shares(self):
        gen, rng = G.default(), np.random.default_rng(5)
        n = np.asarray([gen.count(rng) for _ in range(10_000)])
        assert n.min() == 1 and n.max() == 19
        for (first, last, share) in P.armies_v2()["counts"]:
            assert abs(np.mean((n >= first) & (n <= last)) - share) < 0.02
        assert abs(np.mean(n == 15) - 0.08) < 0.015 and abs(np.mean(n == 19) - 0.08) < 0.015   # even inside a band
        assert max(gen.count(rng, 6) for _ in range(200)) == 6                                   # max_units clips

    def test_equal_gold_lord_and_one_to_nineteen_units(self, battles_v2):
        for a in battles_v2:
            costs = [a["sides"][s]["cost"] for s in ("own", "enemy")]
            assert max(costs) <= min(costs) / (1 - G.TOLERANCE) + 1e-6
            for s in ("own", "enemy"):
                side = a["sides"][s]
                assert sum(u.get("general", False) for u in side["units"]) == 1
                assert 1 <= len(units_of(side)) <= G.MAX_UNITS
            assert 2000 <= a["ceiling"] <= 15000

    def test_big_armies_and_budgets_above_the_old_cap(self, battles_v2):
        n = np.asarray([len(units_of(a["sides"][s])) for a in battles_v2 for s in ("own", "enemy")])
        b = np.asarray([a["budget"] for a in battles_v2])
        assert np.mean(n >= 15) > 0.30 and np.mean(n <= 4) < 0.17 and np.mean((n >= 10) & (n <= 14)) > 0.25
        assert np.mean(b >= 5000) > 0.40 and np.mean(b > 7725) > 0.05 and b.max() <= 15000 * 1.06

    def test_only_the_v2_scheme_fields_the_new_units(self, battles_v2):
        new = {"wh_main_emp_inf_halberdiers", "wh2_dlc12_skv_inf_ratling_gun_0", "wh2_main_skv_inf_plague_monks",
               "wh2_main_skv_inf_plague_monk_censer_bearer", "wh2_dlc16_skv_mon_wolf_rats_0",
               "wh2_dlc16_skv_mon_wolf_rats_1"}
        v1 = {u.key for p in POOLS.values() for u in p.units}
        v2 = {u.key for p in P.load(scheme="v2").values() for u in p.units}
        assert not (v1 & new) and v2 == v1 | new
        seen = {u["key"] for a in battles_v2 for s in ("own", "enemy") for u in units_of(a["sides"][s])}
        assert new <= seen
        old = {u["key"] for s in range(300) for side in G.battle(s)["sides"].values() for u in units_of(side)}
        assert not (old & new)

    def test_template_armies_keep_the_caps(self, battles_v2):
        pools_v2 = P.load(scheme="v2")
        for a in battles_v2:
            for s in ("own", "enemy"):
                side = a["sides"][s]
                if side["army"] != "random":
                    pool, units = pools_v2[side["faction"]], units_of(side)
                    ranged = {u.key for u in pool.units if u.category == "inf_ranged"}
                    assert sum(u["key"] in ranged for u in units) <= pool.cap("inf_ranged", len(units))

    @staticmethod
    def _gen(counts, ceiling):
        gen = G.Generator()
        gen.v2 = {"counts": counts, "ceiling": ceiling}
        return gen

    def test_the_ceiling_bounds_the_price(self):
        gen, rng = self._gen(((3, 3, 1.0),), (2000.0, 2000.0)), np.random.default_rng(6)
        for _ in range(40):
            a = gen.generate(rng, sides=(EMP, SKV), armies="v2")
            assert all(len(units_of(a["sides"][s])) == 3 for s in ("own", "enemy"))
            costs = [a["sides"][s]["cost"] for s in ("own", "enemy")]
            assert min(costs) <= 2000 and max(costs) <= 2000 / (1 - G.TOLERANCE)

    def test_the_count_wins_over_a_low_ceiling(self):
        # 19 Empire units cost at least 600 + 19 x 300 = 6300: the price is the cheapest army, not the ceiling
        gen, rng = self._gen(((19, 19, 1.0),), (2000.0, 2000.0)), np.random.default_rng(7)
        for _ in range(10):
            a = gen.generate(rng, sides=(EMP, EMP), armies="v2")
            assert all(len(units_of(a["sides"][s])) == 19 for s in ("own", "enemy"))
            assert all(a["sides"][s]["cost"] <= 6300 / (1 - G.TOLERANCE) for s in ("own", "enemy"))

    def test_the_bigger_army_shrinks_when_its_cheapest_units_are_too_dear(self):
        # one Empire unit costs at most 600 + 850; 19 Empire units at least 6300: the second side takes what fits
        gen, rng = self._gen(((1, 1, 0.5), (19, 19, 0.5)), (15000.0, 15000.0)), np.random.default_rng(8)
        sizes = set()
        for _ in range(40):
            a = gen.generate(rng, sides=(EMP, EMP), armies="v2")
            costs = [a["sides"][s]["cost"] for s in ("own", "enemy")]
            assert max(costs) <= min(costs) / (1 - G.TOLERANCE) + 1e-6
            sizes.add(tuple(sorted(len(units_of(a["sides"][s])) for s in ("own", "enemy"))))
        assert (19, 19) in sizes and any(s[0] == 1 and s[1] <= 3 for s in sizes) and not any(s == (1, 19) for s in sizes)

    def test_the_smaller_army_is_not_always_elite(self):
        # 5 Skaven units against 5: the price spreads between the cheapest (5 slaves) and the dearest (5 stormvermin)
        gen, rng = self._gen(((5, 5, 1.0),), (15000.0, 15000.0)), np.random.default_rng(9)
        costs = [gen.generate(rng, sides=(SKV, SKV), armies="v2")["sides"]["own"]["cost"] for _ in range(60)]
        cheapest, dearest = 525 + 5 * 125, 525 + 5 * 950
        assert min(costs) < cheapest + 0.25 * (dearest - cheapest) and max(costs) > dearest - 0.25 * (dearest - cheapest)


# The v2 scheme's whole battles (placement included) of the exam's 6 seeds (the game's nn-arena exam and its proxy
# tools/nn/train/exam_sim.py) and a few train seeds, hashed before the v3 scheme joined (23.10.2026): v2 must give them
# bit for bit, or the exam plays other battles.
V2_SEEDS = (1000800014, 1000800163, 1000800092, 1000800042, 1000800127, 1000800133, 0, 1, 2, 17)
V2_HASH = "c1486a702fd1231e24cfd1bea1067da3e00794963f43a4793520d809c6968ca1"


@pytest.fixture(scope="module")
def battles_v3():
    return [G.battle(s, armies="v3") for s in range(1500)]


class TestArmiesV3:
    def test_v2_and_the_exam_seeds_unchanged(self):
        import hashlib
        from tools.nn.train import exam_sim
        assert set(exam_sim.SEEDS) <= set(V2_SEEDS) and exam_sim.ARMIES == "v2"
        arenas = [G.battle(s, armies="v2") for s in V2_SEEDS]
        assert hashlib.sha256(json.dumps(arenas, sort_keys=True).encode()).hexdigest() == V2_HASH

    def test_the_numbers(self):
        v3 = P.armies_v2(scheme="v3")
        assert v3 == {"counts": ((1, 19, 1.0),), "ceiling": (2000.0, 20000.0), "full": 0.05}
        assert P.armies_v2()["full"] == 0.0                        # v2: no full draw
        b = G.battle(17, armies="v3")
        assert b == G.battle(17, armies="v3") and b["armies"] == "v3"
        assert composition(b) != composition(G.battle(17, armies="v2"))

    def test_the_counts_evenly(self):
        gen, rng = G.default(), np.random.default_rng(5)
        n = np.asarray([gen.count(rng, scheme="v3") for _ in range(19_000)])
        assert n.min() == 1 and n.max() == 19
        assert all(abs(np.mean(n == k) - 1 / 19) < 0.01 for k in range(1, 20))

    def test_sides_by_bands_of_five_and_full_battles(self, battles_v3):
        n = np.asarray([[len(units_of(a["sides"][s])) for s in ("own", "enemy")] for a in battles_v3])
        flat = n.ravel()
        assert flat.min() >= 1 and flat.max() == G.MAX_UNITS
        # the lord with them 2-5 / 6-10 / 11-15 / 16-20: each band about a quarter (equal gold shrinks the bigger
        # side of a mismatched pair, so single numbers are not exactly even)
        for lo, hi in ((1, 4), (5, 9), (10, 14), (15, 19)):
            assert abs(np.mean((flat >= lo) & (flat <= hi)) - 0.25) < 0.04
        assert np.mean((n[:, 0] == 19) & (n[:, 1] == 19)) >= 0.045     # 20 against 20 with the lords
        for a in battles_v3:
            costs = [a["sides"][s]["cost"] for s in ("own", "enemy")]
            assert max(costs) <= min(costs) / (1 - G.TOLERANCE) + 1e-6
            assert 2000 <= a["ceiling"] <= 20000

    def test_the_ceiling_up_to_20000(self, battles_v3):
        c = np.asarray([a["ceiling"] for a in battles_v3])
        cost = np.asarray([a["sides"][s]["cost"] for a in battles_v3 for s in ("own", "enemy")])
        assert c.max() > 19000 and abs(np.mean(c > 15000) - 5 / 18) < 0.05
        assert cost.max() > 13000 and cost.max() <= 20000 / (1 - G.TOLERANCE)

    def test_the_cavalry_only_in_v3(self, battles_v3):
        # the Empire's six mounted units (29.10.2026, cavalry step 3): in the v3 pools only (v1 / v2 bit for bit, the test
        # above), each in its generator group (the database's cdir_military_generator_unit_qualities), which every Empire
        # template asks; their width the game's default formation (men / rank_depth files x the template's h)
        cav = {"wh_main_emp_cav_empire_knights": ("medium", 45.0), "wh_main_emp_cav_reiksguard": ("medium", 45.0),
               "wh3_dlc25_emp_cav_knights_of_the_black_rose": ("medium", 45.0),
               "wh_dlc04_emp_cav_knights_blazing_sun_0": ("heavy", 45.0),
               "wh_main_emp_cav_demigryph_knights_0": ("heavy", 51.2), "wh_main_emp_cav_demigryph_knights_1": ("heavy", 51.2)}
        for scheme in ("v1", "v2"):
            assert not set(cav) & {u.key for u in P.load(scheme=scheme)[EMP].units}
        pool = P.load(scheme="v3")[EMP]
        units = {u.key: u for u in pool.units}
        assert set(cav) <= set(units)
        doc = json.loads(P.TEMPLATES.read_text(encoding="utf-8"))["factions"][EMP]["units"]
        for key, (weight, width) in cav.items():
            assert doc[key][0]["group"] == f"wh3_default_land_melee_cavalry_main_mounted_{weight}"
            assert units[key].width == width and units[key].category == "cavalry" and not units[key].missile
        idx = [i for i, u in enumerate(pool.units) if u.key in cav]
        assert pool.templates and all(sum(t.shares[i] for i in idx) > 0.09 for t in pool.templates)
        sides = [a["sides"][s] for a in battles_v3 for s in ("own", "enemy")
                 if any("_emp_" in u["key"] for u in a["sides"][s]["units"])]
        with_cav = np.mean([any(u["key"] in cav for u in side["units"]) for side in sides])
        assert 0.3 < with_cav < 0.7                                  # 0.49 on 600 seeds (build/diag_cav/step3)

    def test_the_artillery_only_in_v3(self, battles_v3):
        # artillery step 1 (29.10.2026): the Empire's four guns and the Skaven's two in the v3 pools only (v1 / v2 bit for
        # bit, the test above), in the generator's artillery_long_range group (cdir_military_generator_unit_qualities);
        # their width one file a gun (4 x the template's h)
        art = {"wh_main_emp_art_mortar": (EMP, 24.0), "wh_main_emp_art_great_cannon": (EMP, 24.0),
               "wh_main_emp_art_helblaster_volley_gun": (EMP, 24.0), "wh_main_emp_art_helstorm_rocket_battery": (EMP, 24.0),
               "wh2_main_skv_art_plagueclaw_catapult": (SKV, 40.0), "wh2_main_skv_art_warp_lightning_cannon": (SKV, 40.0)}
        for scheme in ("v1", "v2"):
            for fac in (EMP, SKV):
                assert not set(art) & {u.key for u in P.load(scheme=scheme)[fac].units}
        v3 = P.load(scheme="v3")
        for key, (fac, width) in art.items():
            units = {u.key: u for u in v3[fac].units}
            doc = json.loads(P.TEMPLATES.read_text(encoding="utf-8"))["factions"][fac]["units"]
            assert doc[key][0]["group"] == "wh3_default_land_artillery_long_range"
            assert units[key].width == width and units[key].category == "artillery" and units[key].missile
        sides = [a["sides"][s] for a in battles_v3 for s in ("own", "enemy")]
        with_art = np.mean([any(u["key"] in art for u in side["units"]) for side in sides])
        assert 0.2 < with_art < 0.5                                  # 0.32 a side on 600 seeds (half the battles)

    def test_full_battles_both_sides_full(self):
        gen = G.Generator()
        gen.v3 = dict(gen.v3, full=1.0)
        rng = np.random.default_rng(11)
        for sides in ((EMP, SKV), (SKV, SKV), (EMP, EMP)):
            for _ in range(5):
                a = gen.generate(rng, sides=sides, armies="v3")
                assert all(len(units_of(a["sides"][s])) == 19 for s in ("own", "enemy"))

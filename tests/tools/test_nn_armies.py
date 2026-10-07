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

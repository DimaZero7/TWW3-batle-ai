"""Evaluation of a network in the simulator: whole battles against fixed opponents, no training
(docs/en/training/training.md), and a few battles written down like the game's recordings.

    python -m tools.nn.train.evaluate --checkpoint build/nn-train/latest.pt      # in the container

play(): per opponent, `per_scene` battles of every scene (the learner's side alternating), all
in one batch until every battle ends. The numbers are the simulator's own (no randomisation),
the start places moved by up to 2 m, the learner's orders sampled (greedy=False) as in training.
Results by the learner's role (attack, defend).

Fair metrics (docs/en/training/training.md "Network evaluation: fair metrics"; tools/nn/train/skill.py):
generated battles are played in swapped pairs (every battle twice on the same armies, the network on side 1, then on
side 2); a script opponent's battles against itself on the same armies give the matchup's natural
edge (baselines(): cached in build/nn-train/baselines per opponent, seed set and simulator version);
one rating with a faction term is fitted over all battles.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from tools.nn.model import factions as fac
from tools.nn.sim import abilities as sim_abilities
from tools.nn.sim import check, scenario
from tools.nn.sim import orders as O
from tools.nn.sim import state as S
from tools.nn.train import behaviour, checkpoint, league, matchups, randomise, reward, rollout, scenes, skill
from tools.nn.train import opponents as scripts

SPREAD = randomise.Spread(common=0.0, side=0.0, jitter_m=2.0)
OPPONENTS = ("nearest", "hold_shoot", "hold", "ai_like", "past")
REPLAYS = checkpoint.DIR / "replays"
BASELINES = checkpoint.DIR / "baselines"
ROOT = Path(__file__).resolve().parents[3]
# What a script-vs-script battle depends on: the simulator, the armies, the scripts, the numbers.
VERSION_FILES = ("tools/nn/sim", "tools/nn/armies", "tools/nn/train/opponents.py", "tools/nn/train/scenes.py",
                 "tools/nn/train/reward.py", "tools/nn/train/randomise.py", "tools/nn/scenario.py", "tools/nn/units.py",
                 "tools/nn/abilities.py", "tools/nn/model", "config/nn")
SEP = (",", ":")          # compact, as the game writes events.jsonl (gamedata looks for '"event":"result"')


def faction_name(key):
    return fac.load()["factions"].get(key, {}).get("name", key or "?")


def label(scene, side):
    """What the learner plays in a scene from a side: 'Empire v Skaven, attack'."""
    army = scenes.army(scene)
    own, enemy = army["sides"][side]["faction"], army["sides"][3 - side]["faction"]
    attacks = army["attacker"] == side
    return f"{faction_name(own)} v {faction_name(enemy)}, {'attack' if attacks else 'defend'}"


def combined(opponents, per_scene, n_scenes, scene_attacker=None, attack_only=()):
    lays = [league.layout(per_scene * n_scenes, n_scenes, opponent=o, scene_attacker=scene_attacker,
                          attack_only=attack_only) for o in opponents]
    return league.Layout(*(np.concatenate([getattr(x, k) for x in lays]) for k in ("scene", "learner", "opponent")))


@torch.no_grad()
def play(actor, opponents=OPPONENTS, per_scene=512, past=None, device="cpu", limit_s=3600.0, greedy=False,
         seed=1, scene_list=scenes.SCENES, compile=None, hold_defend=False, generated=None, max_units=19, small=None,
         together=False, paired=True, baseline=False):
    """-> {"by_opponent": {name: {games, wins, win_rate, seconds, hp_own_lost, hp_enemy_lost, timeouts,
    gold_destroyed, gold_lost (mean gold a battle, reward.gold_sides at the end), gold_ratio (their sums'
    ratio), gold_trade (mean (destroyed - lost) / budget),
    contact_share, first_contact_s, fired_share, orders_per_minute, switches_per_minute, kinds, behaviour,
    roles: {attack|defend: {games, wins, win_rate, seconds, hp_own_lost, hp_enemy_lost, timeouts, behaviour}},
    factions: {EMP|SKV (ours): {attack|defend: {games, wins, win_rate}}}, matchups: {EMP-SKV (ours first): {games,
    wins, win_rate, gold_destroyed, gold_lost, gold_ratio, gold_trade}} (tools/nn/train/matchups.py)}},
    "by_scene": {name: {label: {games, wins, win_rate}}}, "orders_per_minute", "kinds"}. One batch per
    opponent (together: one batch for all, as fast as one; the randomised start places differ). past: the
    actor behind "past" (e.g. the untrained one). `hold` is met only as the defender unless hold_defend: as
    the attacker it never moves, and the battle only waits out the limit. generated: that many battles of
    random armies (tools/nn/armies, EVAL_SEEDS, up to max_units units a side) per opponent instead of the
    scenes. behaviour: tools/nn/train/behaviour.py Tracker.

    Fair metrics (tools/nn/train/skill.py). paired (generated battles): the `generated` battles an
    opponent are pair_count(generated) battles played twice, the learner on side 1, then on side 2; the
    scenes come in such pairs anyway (each scene from both sides). Per opponent: "pairs" (skill.pairs: won
    both / split / lost both, pair_score; not for an attack-only opponent), "pair_gold" (skill.pair_gold:
    the gold balance of the pairs, both pairs of hands on the same armies), "margin" (the mean signed
    margin; per matchup in "matchups"); with baseline (generated, paired, no `small`): "baseline" =
    skill.advantage over the script playing itself on the same battles (baselines(), cached). "skill":
    skill.fit over all battles (the rating with a faction term). "battles": {name: per-battle lists
    (won, own, enemy, attacks, side, pair, seed, margin, trade, destroyed, lost, budget)}."""
    out = {"by_opponent": {}, "by_scene": {}, "limit_s": limit_s, "greedy": greedy, "per_scene": per_scene,
           "battles": {}}
    only = () if hold_defend else league.ATTACK_ONLY
    groups = [tuple(opponents)] if together else [(n,) for n in opponents]
    for names in groups:
        res = _play_many(actor, names, per_scene, past, device, limit_s, greedy, seed, scene_list, compile, only,
                         generated, max_units, small, paired)
        for name, (mine, rows, per) in res.items():
            out["by_opponent"][name] = mine
            out["by_scene"][name] = rows
            out["battles"][name] = per
    if baseline and generated and paired and small is None:
        base = baselines([n for n in opponents if n in scripts.SCRIPTS], pair_count(generated), max_units, limit_s,
                         device, compile)
        for name, b in base.items():
            per = out["battles"][name]
            won_s, trade_s = script_view(b, per["seed"], per["side"])
            out["by_opponent"][name]["baseline"] = dict(
                skill.advantage(per["won"], won_s, per["trade"], trade_s, per["own"], per["enemy"]), cache=b["cache"])
    names = list(out["battles"])
    every = (lambda k: [x for n in names for x in out["battles"][n][k]])
    out["skill"] = skill.fit(every("won"), every("own"), every("enemy"), every("attacks"),
                             [n for n in names for _ in out["battles"][n]["won"]])
    ops = list(out["by_opponent"].values())
    out["orders_per_minute"] = float(np.mean([o["orders_per_minute"] for o in ops]))
    out["kinds"] = {k: float(np.mean([o["kinds"][k] for o in ops])) for k in ops[0]["kinds"]}
    return out


def _summary(sel, won, t, hp_own, hp_enemy, limit_s, lord_own=None, lord_enemy=None, gold=None):
    """gold: (own gold lost, enemy gold destroyed, budget) per battle (reward.gold_sides, at the end)."""
    n = int(sel.sum())
    if not n:
        return {"games": 0}
    out = {"games": n, "wins": int(won[sel].sum()), "win_rate": float(won[sel].mean()),
           "seconds": float(t[sel].mean()), "hp_own_lost": float(hp_own[sel].mean()),
           "hp_enemy_lost": float(hp_enemy[sel].mean()), "timeouts": float((t[sel] >= limit_s - 1e-6).mean())}
    if gold is not None:
        own, enemy, bud = (g[sel] for g in gold)
        out.update({"gold_destroyed": float(enemy.mean()), "gold_lost": float(own.mean()),
                    "gold_ratio": float(enemy.sum() / max(1e-9, own.sum())),
                    "gold_trade": float(((enemy - own) / bud).mean())})
    if lord_own is not None:
        out["lord_dead_own"] = float(lord_own[sel].mean())
        out["lord_dead_enemy"] = float(lord_enemy[sel].mean())
    return out


def generated_layout(source, opponent, attack_only=league.ATTACK_ONLY, rows=None):
    """Battle b = bank battle b; the learner's side alternates every two battles (the bank alternates
    who attacks), so both roles and both sides come up; an attack_only opponent only defends.
    rows: the bank rows of this opponent's battles (default: all)."""
    rows = np.arange(source.bank.M) if rows is None else np.asarray(rows)
    n = len(rows)
    side = 1 + (np.arange(n) // 2) % 2
    if opponent in attack_only:
        side = source.bank.attacker.cpu().numpy()[rows]
    return league.Layout(np.zeros(n, dtype=int), side, np.full(n, league.CODE[opponent]))


def pair_count(generated):
    """Pairs a paired evaluation of `generated` battles an opponent plays: generated // 2, even (two pairs
    make a block of paired_order), at least 2."""
    return max(2, generated // 4 * 2)


def eval_seeds(n):
    """The first n seeds of the generator's EVAL_SEEDS."""
    from tools.nn.armies import generate
    return list(range(generate.EVAL_SEEDS.start, generate.EVAL_SEEDS.start + n))


def paired_order(n_pairs, n_opponents):
    """The batch of a paired evaluation: (opponent index, pair, learner side, seed index) per battle, [M]
    each, M = 2 x n_pairs x n_opponents. Blocks of four battles [p, q, p, q] (two pairs: seeds 2k and
    2k + 1), the learner on side 1 in the first two, on side 2 in the last two; the blocks cycle through
    the opponents. A battle's position sets its armies' role (scenes.Generated: side 1 attacks at even
    positions) and size (`small`: the first ones), so both battles of a pair stay the same battle."""
    assert n_pairs % 2 == 0, n_pairs
    k = np.arange(n_pairs // 2 * n_opponents)
    opp, block = k % n_opponents, k // n_opponents
    i = np.arange(4)
    pair = (2 * block[:, None] + i % 2).ravel()
    side = np.tile(1 + i // 2, len(k))
    return np.repeat(opp, 4), pair, side, pair.copy()


def padded(seeds, small):
    """seeds padded at the end (bank rows the batch never plays), so that scenes.Generated's small share
    ends at a block of four: no pair has one small and one full battle."""
    seeds = list(seeds)
    if small:
        while int(round(small[0] * len(seeds))) % 4:
            seeds.append(seeds[0])
    return seeds


def _play_many(actor, names, per_scene, past, device, limit_s, greedy, seed, scene_list, compile, attack_only,
               generated=None, max_units=19, small=None, paired=True):
    """{name: (result, rows by scene, per-battle lists)} of the opponents `names`, played in one batch."""
    params = rollout.params_with_limit(limit_s)
    if generated and paired:
        n_pairs = pair_count(generated)
        opp_i, pair, side_b, seed_i = paired_order(n_pairs, len(names))
        seeds_b = np.array(eval_seeds(n_pairs))[seed_i]
        source = scenes.Generated(padded(seeds_b.tolist(), small), max_units, params, device, sequential=True,
                                  small=small)
        only = np.isin(np.array(names)[opp_i], list(attack_only))
        side_b = np.where(only, source.bank.attacker.cpu().numpy()[:len(side_b)], side_b)
        pair = np.where(only, -1, pair)                                # an attack-only opponent: no pairs
        lay = league.Layout(np.zeros(len(opp_i), dtype=int), side_b,
                            np.array([league.CODE[n] for n in names])[opp_i])
    elif generated:
        seeds = eval_seeds(generated)
        source = scenes.Generated(seeds * len(names), max_units, params, device, sequential=True, small=small)
        lays = [generated_layout(source, n, attack_only, range(i * generated, (i + 1) * generated))
                for i, n in enumerate(names)]
        lay = league.Layout(*(np.concatenate([getattr(x, k) for x in lays]) for k in ("scene", "learner", "opponent")))
        seeds_b = np.array(seeds * len(names))
        pair = np.full(lay.B, -1)
    else:
        source = None
        lay = combined(names, per_scene, len(scene_list), scenes.attackers(scene_list), attack_only)
        # league.layout: battles j and j + n_scenes of an opponent play one scene from the two sides
        n_sc = len(scene_list)
        j = np.concatenate([np.arange((lay.opponent == league.CODE[n]).sum()) for n in names])
        pair = np.where(np.isin(lay.opponent, [league.CODE[n] for n in attack_only]), -1,
                        j % n_sc + n_sc * (j // (2 * n_sc)))
        seeds_b = lay.scene
    env = rollout.Battles(lay, scene_list, device=device, params=params, spread=SPREAD,
                          seed=seed, auto_reset=False, compile=compile, source=source)
    if "past" in names:
        env.set_past(past)
    B = env.B
    learner_side = torch.as_tensor(lay.learner, device=env.device)
    mine = env.st.u["side"] == learner_side[:, None]
    contact = torch.full((B,), -1.0, device=env.device)
    fired = torch.zeros(B, dtype=torch.bool, device=env.device)
    watch = behaviour.Tracker(env.st, env.params, mine, wrap=lambda f: rollout.fast(f, env.compiled))
    for _ in range(int(limit_s / env.params.dt) + 2):
        if bool(env.st.done.all()):
            break
        live = ~env.st.done
        env.step(actor, None, greedy)
        watch.update(env.st, live)
        u = env.st.u
        touch = (u["m"] & mine).any(1)
        contact = torch.where((contact < 0) & touch, env.st.t, contact)
        fired = fired | (u["fire"] & mine).any(1)
    health = reward.health(env.st).cpu().numpy()
    winner = env.st.winner.cpu().numpy()
    t = env.st.t.cpu().numpy()
    side = lay.learner
    won = winner == side
    hp_own = 1 - health[np.arange(B), side - 1]
    hp_enemy = 1 - health[np.arange(B), 2 - side]
    attacks_all = env.st.attacker.cpu().numpy() == side
    c_all = contact.cpu().numpy()
    fired_all = fired.cpu().numpy()
    dead = env.st.lord_dead_s.cpu().numpy() >= 0                      # [B, side]
    lords = (dead[np.arange(B), side - 1], dead[np.arange(B), 2 - side])
    gold_b = reward.gold_sides(env.st.u, env.weights.rout_share).cpu().numpy()   # [B, side]
    gold = (gold_b[np.arange(B), side - 1], gold_b[np.arange(B), 2 - side], reward.budget(env.st.u).cpu().numpy())
    margin = skill.margin(winner, side, gold_b, start_cost(env.st.u))
    trade = (gold[1] - gold[0]) / np.maximum(gold[2], 1e-9)
    fac_b = env.setup.factions                                       # [(side 1, side 2)] per battle
    own_f = np.array([fac_b[b][side[b] - 1] for b in range(B)], dtype=object)
    enemy_f = np.array([fac_b[b][2 - side[b]] for b in range(B)], dtype=object)
    kinds_b = env.kind_battle.cpu().numpy()
    orders_b = env.order_battle.cpu().numpy()                         # changes, switches, unit-steps
    minutes = params.dt / 60
    out = {}
    for name in names:
        this = lay.opponent == league.CODE[name]
        attacks = attacks_all & this
        c = c_all[this]
        k = kinds_b[this].sum(0)
        ch, sw, steps = orders_b[this].sum(0)
        result = _summary(this, won, t, hp_own, hp_enemy, limit_s, *lords, gold=gold)
        result.update({"contact_share": float((c >= 0).mean()),
                       "first_contact_s": float(np.median(c[c >= 0])) if (c >= 0).any() else None,
                       "fired_share": float(fired_all[this].mean()),
                       "orders_per_minute": float(ch / max(1e-9, steps * minutes)),
                       "switches_per_minute": float(sw / max(1e-9, steps * minutes)),
                       "kinds": {kd: float(k[i] / max(1, k.sum())) for i, kd in enumerate(O.KINDS)},
                       "behaviour": watch.summary(this),
                       "roles": {"attack": _summary(attacks, won, t, hp_own, hp_enemy, limit_s, *lords, gold=gold),
                                 "defend": _summary(this & ~attacks, won, t, hp_own, hp_enemy, limit_s, *lords,
                                                    gold=gold)}})
        result.update(matchups.group(won[this], own_f[this], enemy_f[this], attacks_all[this],
                                     gold=tuple(g[this] for g in gold), margin=margin[this]))
        result["margin"] = float(margin[this].mean())
        if (pair[this] >= 0).any():
            result["pairs"] = skill.pairs(won[this], pair[this])
            result["pair_gold"] = skill.pair_gold(pair[this], side[this], gold[1][this], gold[0][this],
                                                  gold[2][this], margin[this])
        per = {"won": won[this].tolist(), "own": own_f[this].tolist(), "enemy": enemy_f[this].tolist(),
               "attacks": attacks_all[this].tolist(), "side": side[this].tolist(), "pair": pair[this].tolist(),
               "seed": np.asarray(seeds_b)[this].tolist(), "margin": np.round(margin[this], 4).tolist(),
               "trade": np.round(trade[this], 4).tolist(), "destroyed": np.round(gold[1][this], 2).tolist(),
               "lost": np.round(gold[0][this], 2).tolist(), "budget": np.round(gold[2][this], 2).tolist()}
        for role, sel in (("attack", attacks), ("defend", this & ~attacks)):
            if sel.any():
                result["roles"][role]["behaviour"] = watch.summary(sel)
        rows = {}
        if generated:
            for b in np.nonzero(this)[0]:
                key = f"{faction_name(own_f[b])} v {faction_name(enemy_f[b])}, {'attack' if attacks_all[b] else 'defend'}"
                r = rows.setdefault(key, {"games": 0, "wins": 0})
                r["games"] += 1
                r["wins"] += int(won[b])
        else:
            for i, sc in enumerate(scene_list):
                for s_ in (1, 2):
                    k_ = this & (lay.scene == i) & (side == s_)
                    if not k_.any():
                        continue
                    r = rows.setdefault(label(sc, s_), {"games": 0, "wins": 0})
                    r["games"] += int(k_.sum())
                    r["wins"] += int(won[k_].sum())
        for r in rows.values():
            r["win_rate"] = r["wins"] / max(1, r["games"])
        out[name] = (result, rows, per)
    return out


def start_cost(u):
    """[B, 2] numpy: each side's starting gold (the cost of its units)."""
    return torch.stack([(u["cost"] * (u["side"] == s)).sum(1) for s in (1, 2)], 1).cpu().numpy()


# --- the script baseline: the opponent script against itself on the same battles --------------------

def sim_version():
    """A hash of what a script-vs-script battle depends on: VERSION_FILES (line ends ignored) and the
    generator's budget factors as loaded (an override in memory changes the armies too)."""
    from tools.nn.armies import generate
    h = hashlib.sha256()
    h.update(json.dumps(sorted((f, p.budget_factor) for f, p in generate.default().pools.items())).encode())
    for rel in VERSION_FILES:
        p = ROOT / rel
        for f in (sorted(p.rglob("*")) if p.is_dir() else [p]):
            if f.is_file() and f.suffix in (".py", ".json") and "__pycache__" not in f.parts:
                h.update(f.relative_to(ROOT).as_posix().encode())
                h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:12]


def baseline_path(name, max_units, limit_s, version=None):
    return BASELINES / f"{name}_{max_units}u_{limit_s:g}s_{version or sim_version()}.json"


@torch.no_grad()
def script_battles(names, n_pairs, max_units=19, limit_s=3600.0, device="cpu", compile=None, seed=1):
    """{name: {seed, winner, lost [n, 2], start [n, 2], budget, factions [n, 2], attacker}}: the first
    n_pairs EVAL seeds, each played once by the script `name` on both sides, with the armies and roles
    of a paired evaluation (scenes.Generated: side 1 attacks at even positions, as in paired_order)."""
    params = rollout.params_with_limit(limit_s)
    seeds = eval_seeds(n_pairs)
    source = scenes.Generated(seeds * len(names), max_units, params, device, sequential=True)
    code = np.repeat([league.CODE[n] for n in names], n_pairs)
    lay = league.Layout(np.zeros(len(code), dtype=int), np.ones(len(code), dtype=int), code)
    env = rollout.Battles(lay, (), device=device, params=params, spread=SPREAD, seed=seed, auto_reset=False,
                          compile=compile, source=source)
    ctrl = torch.as_tensor(np.stack([code, code], 1), device=env.device)
    sim_abilities.set_rule(env.st.u, torch.ones_like(ctrl, dtype=torch.bool))     # both sides by the game-AI rule
    plays = tuple((league.CODE[n], scripts.SCRIPTS[n]) for n in names)
    for _ in range(int(limit_s / env.params.dt) + 2):
        if bool(env.st.done.all()):
            break
        env.advance(env.st, env._assemble(env.st, ctrl, plays, ()), env.params, env.params.dt)
    winner = env.st.winner.cpu().numpy()
    lost = reward.gold_sides(env.st.u, env.weights.rout_share).cpu().numpy()
    start = start_cost(env.st.u)
    budget = reward.budget(env.st.u).cpu().numpy()
    attacker = env.st.attacker.cpu().numpy()
    out = {}
    for i, n in enumerate(names):
        r = slice(i * n_pairs, (i + 1) * n_pairs)
        out[n] = {"seed": seeds, "winner": winner[r].tolist(), "lost": np.round(lost[r], 2).tolist(),
                  "start": np.round(start[r], 2).tolist(), "budget": np.round(budget[r], 2).tolist(),
                  "factions": [list(f) for f in env.setup.factions[r]], "attacker": attacker[r].tolist()}
    return out


def baselines(names, n_pairs, max_units=19, limit_s=3600.0, device="cpu", compile=None):
    """{name: script_battles()[name] + {"cache": its file name}} of the script opponents `names`: read
    from build/nn-train/baselines (a file per opponent, units, limit and simulator version; one of more
    seeds serves a prefix), the missing ones played in one batch and written there."""
    version = sim_version()
    out, missing = {}, []
    for n in names:
        path = baseline_path(n, max_units, limit_s, version)
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            doc = None
        if doc and doc["seed"][:n_pairs] == eval_seeds(n_pairs):
            out[n] = dict({k: v[:n_pairs] for k, v in doc.items()}, cache=path.name)
        else:
            missing.append(n)
    if missing:
        BASELINES.mkdir(parents=True, exist_ok=True)
        for n, doc in script_battles(missing, n_pairs, max_units, limit_s, device, compile).items():
            path = baseline_path(n, max_units, limit_s, version)
            path.write_text(json.dumps(doc, separators=SEP), encoding="utf-8", newline="\n")
            out[n] = dict(doc, cache=path.name)
    return out


def script_view(base, seeds, sides):
    """(won [n] bool, trade [n]) of the script's own battle of each seed, from the side the network
    played there (trade: (gold destroyed - gold lost) / budget)."""
    at = {s: i for i, s in enumerate(base["seed"])}
    idx = np.array([at[s] for s in seeds], dtype=int)
    side = np.asarray(sides, dtype=int)
    lost = np.asarray(base["lost"], dtype=float)[idx]
    n = np.arange(len(idx))
    trade = (lost[n, 2 - side] - lost[n, side - 1]) / np.maximum(np.asarray(base["budget"], dtype=float)[idx], 1e-9)
    return np.asarray(base["winner"])[idx] == side, trade


def names_of(army, H):
    """(script names, keys, sides, slots) of the army's units in slot order."""
    slot = scenario.slots(army, H)
    units = [(slot[u["name"]], u, s) for s in (1, 2) for u in army["sides"][s]["units"]]
    units.sort(key=lambda x: x[0])
    return [u["name"] for _, u, _ in units], [u["key"] for _, u, _ in units], [s for _, _, s in units], \
        [sl for sl, _, _ in units]


def write_run(path, b, army, arena, side, opponent, source):
    """A simulated battle (gamedata.Battle) -> a run folder like the game's: manifest.json and
    events.jsonl (nn_sample every second, result), readable by tools.nn.gamedata.load."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    tags = {1: "own", 2: "enemy"}
    units = {tags[s]: [{"script_name": n, "key": k, "slot": n.split("_", 1)[-1]}
                       for n, k, sd in zip(b.names, b.keys, b.side) if sd == s] for s in (1, 2)}
    cfg = {"arena": arena, "own_ai": "attack" if army["attacker"] == 1 else "defend",
           "enemy_role": "attack" if army["attacker"] == 2 else "defend",
           "factions": {"own": army["sides"][1]["faction"], "enemy": army["sides"][2]["faction"]},
           "units": units, "source": "simulator", "network_side": side, "opponent": opponent, "checkpoint": source}
    (path / "manifest.json").write_text(json.dumps({"config": cfg}, indent=1), encoding="utf-8", newline="\n")
    lines = []
    for ti in range(len(b.t)):
        rows = []
        for i, n in enumerate(b.names):
            u = {"n": n, "side": int(b.side[i])}
            for k in check.FLOATS:
                u[k] = round(float(b.f[k][ti, i]), 3)
            for k in check.BOOLS:
                u[k] = bool(b.f[k][ti, i])
            tg = int(b.target[ti, i])
            u["t"] = b.names[tg] if tg >= 0 else ""
            u["fat"] = S.FATIGUE_LEVELS[int(b.f["fat"][ti, i])]
            rows.append(u)
        lines.append(json.dumps({"event": "nn_sample", "t": int(round(b.t[ti] * 1000)), "units": rows}, separators=SEP))
    lines.append(json.dumps({"event": "result", "status": "completed", "winner": int(b.winner)}, separators=SEP))
    (path / "events.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


@torch.no_grad()
def record(actor, out=REPLAYS, opponents=("nearest", "hold_shoot", "hold", "ai_like", "self"), device="cpu", limit_s=3600.0, source="", seed=2,
           scene_list=scenes.SCENES, compile=None):
    """Every scene from both sides against each opponent, written down per second; -> run folders."""
    lay = combined(opponents, 2, len(scene_list), scenes.attackers(scene_list), league.ATTACK_ONLY)
    env = rollout.Battles(lay, scene_list, device=device, params=rollout.params_with_limit(limit_s), spread=SPREAD,
                          seed=seed, auto_reset=False, compile=compile)
    rec = check.Recorder(env.st)
    for _ in range(int(limit_s / env.params.dt) + 2):
        if bool(env.st.done.all()):
            break
        env.step(actor, None, False)
        rec(env.st)
    winners = env.st.winner.cpu().numpy()
    H = env.N // 2
    paths = []
    for bi in range(env.B):
        sc = scene_list[int(lay.scene[bi])]
        army = scenes.army(sc)
        names, keys, sides, slot_of = names_of(army, H)
        b = rec.battle(bi, tuple(names), tuple(keys), np.array(sides), slot_of, winners[bi], sc[1], sc[0])
        opp = league.OPPONENTS[int(lay.opponent[bi]) - 1]
        tag = f"{sc[0]}-{sc[1]}-net{int(lay.learner[bi])}-{opp}"
        paths.append(write_run(Path(out) / tag, b, army, sc[0], int(lay.learner[bi]), opp, source))
    return paths


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(checkpoint.LATEST))
    ap.add_argument("--against", default=str(checkpoint.RANDOM), help="the network behind 'past'")
    ap.add_argument("--per-scene", type=int, default=512)
    ap.add_argument("--limit", type=float, default=3600.0)
    ap.add_argument("--hold-defend", action="store_true", help="also meet `hold` as the attacker")
    ap.add_argument("--generated", type=int, default=0, help="battles of random armies (EVAL_SEEDS) per opponent")
    ap.add_argument("--max-units", type=int, default=19)
    ap.add_argument("--opponents", default=",".join(OPPONENTS), help="comma-separated opponents")
    ap.add_argument("--no-pairs", action="store_true", help="generated battles once each, not in swapped pairs")
    ap.add_argument("--no-baseline", action="store_true", help="no script-vs-script baseline (generated battles)")
    ap.add_argument("--together", action="store_true", help="all opponents in one batch")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--out", help="write the results as json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    actor = checkpoint.load_policy(args.checkpoint, args.device)
    past = checkpoint.load_policy(args.against, args.device)
    res = play(actor, opponents=tuple(args.opponents.split(",")), per_scene=args.per_scene, past=past, device=args.device, limit_s=args.limit, greedy=args.greedy,
               hold_defend=args.hold_defend, generated=args.generated, max_units=args.max_units, together=args.together,
               paired=not args.no_pairs, baseline=not args.no_baseline)
    from tools.nn.train.run import show
    show(Path(args.checkpoint).name, res)
    print("\n".join(skill.text(skill.summary(res))), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()

"""Evaluation of a network in the simulator: whole battles against fixed opponents, no training
(docs/en/training/training.md), and a few battles written down like the game's recordings.

    python -m tools.nn.train.evaluate --checkpoint build/nn-train/latest.pt      # in the container

play(): per opponent, `per_scene` battles of every scene (the learner's side alternating), all
in one batch until every battle ends. The numbers are the simulator's own (no randomisation),
the start places moved by up to 2 m, the learner's orders sampled (greedy=False) as in training.
Results by the learner's role (attack, defend). The network decides as often as in training and in the
game (tools/nn/train/cadence.py; --decide-s, --order-latency); the scripts every simulator step.

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
from types import SimpleNamespace

import numpy as np
import torch

from tools.nn.model import factions as fac
from tools.nn.sim import abilities as sim_abilities
from tools.nn.sim import check, scenario
from tools.nn.sim import orders as O
from tools.nn.sim import state as S
from tools.nn.train import behaviour, checkpoint, league, matchups, randomise, reward, rollout, scenes, skill
from tools.nn.train import cadence as cad
from tools.nn.train import opponents as scripts
from tools.nn.train import version as sim_ver
from tools.nn.train.drills import transfer as drill_transfer

SPREAD = randomise.Spread(common=0.0, side=0.0, jitter_m=2.0)
OPPONENTS = ("nearest", "hold_shoot", "hold", "ai_like", "past")
REPLAYS = checkpoint.DIR / "replays"
BASELINES = checkpoint.DIR / "baselines"
ROOT = Path(__file__).resolve().parents[3]
# What a script-vs-script battle depends on: the simulator, the armies, the scripts, the numbers
# (tools/nn/train/version.py; the same hash on the host without torch).
VERSION_FILES = sim_ver.VERSION_FILES
# Pairs of script battles played on a baseline cache miss to try an older version's file first
# (baselines(); test5 --baseline-canary N; 0: always play the whole baseline). When those pairs come
# out identical in every field (winner, gold lost, start, budget, factions, attacker) the older file
# is adopted under the new version: a code change that cannot touch a script battle (the network, a
# reward term) no longer costs the ~20-30 minutes of playing 256 pairs x 3 scripts again.
CANARY = 0
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
         together=False, paired=True, baseline=False, compact=True, cuda_graph=True, cadence=None):
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
    (won, own, enemy, attacks, side, pair, seed, margin, trade, destroyed, lost, budget)}.

    compact: once most battles have ended, step only the running ones (BUCKETS; rollout.Battles.narrow).
    The simulator is deterministic and an ended battle frozen, so only the sampled orders' random draws
    differ from compact=False (the batch's shape sets them); the results stay the same in distribution.
    cuda_graph (CUDA, compact): the shrunk batch's decisions replayed as one CUDA graph (Graphed).
    cadence: how often the network decides and how late its orders land (tools/nn/train/cadence.py;
    default the game's, as in training).

    "transfer" (tools/nn/train/drills/transfer.py): per drill with a transfer detector, the share of the
    unit-seconds in the drill's situation where the units apply its skill, the network's and (per
    opponent) the opponent script's; "ai_like": the ai_like script's units, the reference."""
    out = {"by_opponent": {}, "by_scene": {}, "limit_s": limit_s, "greedy": greedy, "per_scene": per_scene,
           "battles": {}, "transfer": {}}
    only = () if hold_defend else league.ATTACK_ONLY
    groups = [tuple(opponents)] if together else [(n,) for n in opponents]
    for names in groups:
        res = _play_many(actor, names, per_scene, past, device, limit_s, greedy, seed, scene_list, compile, only,
                         generated, max_units, small, paired, compact, cuda_graph, cadence)
        for name, (mine, rows, per) in ((k, v) for k, v in res.items() if k != _TRANSFER):
            out["by_opponent"][name] = mine
            out["by_scene"][name] = rows
            out["battles"][name] = per
        out["transfer"] = merge_transfer(out["transfer"], res.get(_TRANSFER, {}))
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


_TRANSFER = "_transfer"     # _play_many's key of its transfer report (not an opponent)


def merge_transfer(acc, new):
    """The transfer reports of several batches (together=False) as one: per drill the opponents' blocks
    side by side; "network" and "ai_like" from the batch that has them (the network's over all opponents
    only when one batch played them all)."""
    for name, x in new.items():
        if name not in acc:
            acc[name] = x
            continue
        acc[name]["by_opponent"].update(x["by_opponent"])
        if "ai_like" in x:
            acc[name]["ai_like"] = x["ai_like"]
        acc[name]["network"] = None                  # (a mean over batches would need the sums)
    return acc


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


# Evaluation steps only the battles still running: once few are left, the batch shrinks to the
# smallest of these sizes that holds them (rollout.Battles.narrow; padded with ended battles, which
# stay frozen). Fixed sizes: torch.compile makes a graph per batch size (~100 s each the first time,
# then kept in the torch cache); a graph for any size (dynamic=True) would not build: the simulator's
# step then needs a C++ compiler the container lacks.
BUCKETS = (64,)
CHECK_EVERY = 8           # steps between the checks (each reads the GPU's answer back)


def bucket(n_live, B):
    """The batch size for n_live running battles in a batch of B: the smallest BUCKETS size below B
    that holds them, else B."""
    fit = [s for s in BUCKETS if n_live <= s < B]
    return min(fit) if fit else B


def _put(full, orig, st):
    """The running batch's State st back into the whole batch's full at the places orig (in place)."""
    if st is full:                               # not shrunk yet: the running batch is the whole one
        return
    for k, v in st.u.items():
        full.u[k][orig] = v
    for k in ("t", "attacker", "done", "winner", "lord_dead_s"):
        getattr(full, k)[orig] = getattr(st, k)


def _ended(env, size):
    """Running battles first, then as many ended ones as fill size (rollout.Battles.narrow's keep)."""
    done = env.st.done
    n = int((~done).sum())
    return torch.cat([(~done).nonzero().squeeze(1), done.nonzero().squeeze(1)[:size - n]])


def _per_battle(obj, B):
    """{attribute: tensor [B, ...] or dict of them} of obj (e.g. behaviour.Tracker: its sums, memories)."""
    def rows(x):
        return torch.is_tensor(x) and x.dim() > 0 and x.shape[0] == B
    return {k: v for k, v in vars(obj).items()
            if rows(v) or (isinstance(v, dict) and v and all(rows(x) for x in v.values()))}


def _scatter(full, orig, cur):
    if isinstance(full, dict):
        for k, v in cur.items():
            full[k][orig] = v
    else:
        full[orig] = cur


class _Full:
    """The whole evaluation batch while rollout.Battles.narrow shrinks the one that steps: what the
    results read (the state, the per-battle counters, the per-battle tensors of the watchers: the
    behaviour.Tracker, the drills' transfer Tracker) at full size, taken at the first shrink; a shrink
    writes the running batch back first, finish() at the end."""

    def __init__(self, env, watch):
        self.watches = tuple(watch) if isinstance(watch, (tuple, list)) else (watch,)
        self.orig = torch.arange(env.B, device=env.device)          # running battle -> its place in the batch
        self.full = None

    def _save(self, env, contact, fired):
        ws = [_per_battle(w, env.B) for w in self.watches]
        if self.full is None:                    # the first shrink: the running batch is the whole one
            self.full = dict(st=env.st, setup=env.setup, kind_battle=env.kind_battle,
                             order_battle=env.order_battle, contact=contact, fired=fired, watch=ws)
            return
        f, o = self.full, self.orig
        _put(f["st"], o, env.st)
        for k, v in (("kind_battle", env.kind_battle), ("order_battle", env.order_battle), ("contact", contact),
                     ("fired", fired)):
            f[k][o] = v
        for i, w in enumerate(ws):
            for k, v in w.items():
                _scatter(f["watch"][i][k], o, v)

    def shrink(self, env, contact, fired, mine):
        """-> (any battle running, contact, fired, mine of the batch that steps on)."""
        n = int((~env.st.done).sum())
        size = bucket(n, env.B)
        if n == 0 or size >= env.B:
            return n > 0, contact, fired, mine
        self._save(env, contact, fired)
        ws = [_per_battle(w, env.B) for w in self.watches]
        keep = env.narrow(_ended(env, size))
        self.orig = self.orig[keep]
        for watch, w in zip(self.watches, ws):
            for k, v in w.items():
                setattr(watch, k, {kk: x[keep] for kk, x in v.items()} if isinstance(v, dict) else v[keep])
        return True, contact[keep], fired[keep], mine[keep]

    def finish(self, env, contact, fired):
        """-> (the whole batch: st, setup, weights, kind_battle, order_battle; contact; fired), at full size.
        The Tracker is the whole batch's again."""
        if self.full is None:                    # never shrunk
            return env, contact, fired
        self._save(env, contact, fired)
        f = self.full
        for watch, w in zip(self.watches, f["watch"]):
            for k, v in w.items():
                setattr(watch, k, v)
        whole = SimpleNamespace(st=f["st"], setup=f["setup"], weights=env.weights, kind_battle=f["kind_battle"],
                                order_battle=f["order_battle"])
        return whole, f["contact"], f["fired"]


# What Graphed walks for the tensors a decision reads and writes (the bank and the setup are only read).
_WALK = (S.State, rollout.Battles, behaviour.Tracker, drill_transfer.Tracker, SimpleNamespace, rollout.ob.Memory,
         rollout.Frame)
_SKIP = {"source", "bank", "setup", "params", "weights", "layout", "scripts", "past_actor", "gen"}


def _leaves(root):
    """{path: tensor} of everything reachable from root through _WALK objects, dicts, lists and tuples."""
    out = {}

    def walk(x, path):
        if torch.is_tensor(x):
            out[path] = x
        elif isinstance(x, dict):
            for k, v in x.items():
                walk(v, path + (("item", k),))
        elif isinstance(x, (list, tuple)):
            for k, v in enumerate(x):
                walk(v, path + (("idx", k),))
        elif isinstance(x, _WALK):
            for k, v in vars(x).items():
                if k not in _SKIP:
                    walk(v, path + (("attr", k),))
    walk(root, ())
    return out


def _set(root, path, value):
    """Puts value at path (from _leaves) under root; a tuple on the way is rebuilt."""
    def rec(x, p):
        if not p:
            return value
        kind, k = p[0]
        if kind == "attr":
            setattr(x, k, rec(getattr(x, k), p[1:]))
        elif kind == "item" or isinstance(x, list):
            x[k] = rec(x[k], p[1:])
        else:
            return tuple(rec(v, p[1:]) if j == k else v for j, v in enumerate(x))
        return x
    rec(root, path)


class Graphed:
    """fn (one decision of a batch: the state reached from root, changed in place or replaced) as one
    CUDA graph: a decision of a small batch is ~10 ms of launching hundreds of small kernels while the
    GPU idles; the graph's replay launches them at once. fn first runs `warm` times for real
    (compiling, cuBLAS); the capture records fn, then copies every tensor fn replaced back into the one
    it replaced, so each replay leaves the state where the next one reads it. The random draws come
    from the graph's own offsets of the generator (the same distribution). If the capture fails, ok is
    False and the caller steps as before."""

    def __init__(self, fn, root, B, warm=3):
        self.B, self.warm, self.ok, self.graph = B, warm, False, None
        self._own(root)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warm):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        self._own(root)
        before = _leaves(root)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                fn()
                after = _leaves(root)
                if set(after) != set(before):
                    raise RuntimeError("the state changed its layout")
                # a new tensor in the memory of an input (the input itself, a view) is copied first
                ptrs = {x.untyped_storage().data_ptr() for x in before.values()}
                moved = [(before[p], after[p].clone() if after[p].untyped_storage().data_ptr() in ptrs else after[p])
                         for p in before if after[p] is not before[p]]
                for dst, src in moved:
                    dst.copy_(src)
            self.graph, self.ok = graph, True
        except RuntimeError as e:
            print(f"evaluate: no CUDA graph ({str(e)[:200]}); stepping as before", flush=True)
        for p, x in before.items():                      # the state is the captured inputs again
            _set(root, p, x)

    @staticmethod
    def _own(root):
        """No two paths share memory (the copies back would overwrite each other): each gets its own."""
        seen = set()
        for path, x in _leaves(root).items():
            ptr = x.untyped_storage().data_ptr()
            if ptr in seen and x.numel():
                _set(root, path, x.clone())
            seen.add(ptr)

    def replay(self):
        self.graph.replay()


def _play_many(actor, names, per_scene, past, device, limit_s, greedy, seed, scene_list, compile, attack_only,
               generated=None, max_units=19, small=None, paired=True, compact=True, cuda_graph=True, cadence=None):
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
                          seed=seed, auto_reset=False, compile=compile, source=source, cadence=cadence)
    if "past" in names:
        env.set_past(past)
    B = env.B
    learner_side = torch.as_tensor(lay.learner, device=env.device)
    mine = env.st.u["side"] == learner_side[:, None]
    contact = torch.full((B,), -1.0, device=env.device)
    fired = torch.zeros(B, dtype=torch.bool, device=env.device)
    watch = behaviour.Tracker(env.st, env.params, mine, wrap=lambda f: rollout.fast(f, env.compiled))
    xfer = drill_transfer.Tracker(env.st, env.params, mine, wrap=lambda f: rollout.fast(f, env.compiled))
    full = _Full(env, (watch, xfer)) if compact else None
    acc = SimpleNamespace(contact=contact, fired=fired, mine=mine)

    def counts(live):
        """The evaluation's own per-battle counts, after every simulator step."""
        watch.update(env.st, live)
        xfer.update(env.st, live)
        u = env.st.u
        touch = (u["m"] & acc.mine).any(1)
        acc.contact = torch.where((acc.contact < 0) & touch, env.st.t, acc.contact)
        acc.fired = acc.fired | (u["fire"] & acc.mine).any(1)

    def one():
        """One decision of every battle (its simulator steps), and the counts."""
        env.step(actor, None, greedy, each=counts)

    graph = None
    n_steps, i = int(limit_s / env.decision_s) + 2, 0
    while i < n_steps:
        if compact:
            live_any, acc.contact, acc.fired, acc.mine = full.shrink(env, acc.contact, acc.fired, acc.mine)
            if not live_any:
                break
            if (cuda_graph and env.device.type == "cuda" and env.B in BUCKETS and n_steps - i > 4 * CHECK_EVERY
                    and (graph is None or graph.B != env.B)):
                graph = None                     # (a graph of another size: its memory goes)
                graph = Graphed(one, SimpleNamespace(env=env, watch=watch, xfer=xfer, acc=acc), env.B)
                i += graph.warm
        elif bool(env.st.done.all()):
            break
        # between checks: a battle that ended stays frozen (the simulator, the counters skip it)
        for _ in range(min(CHECK_EVERY if compact else 1, n_steps - i)):
            if graph is not None and graph.ok:
                graph.replay()
            else:
                one()
            i += 1
    graph = None
    contact, fired = acc.contact, acc.fired
    if compact:
        env, contact, fired = full.finish(env, contact, fired)
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
    out[_TRANSFER] = drill_transfer.report(xfer, {n: lay.opponent == league.CODE[n] for n in names})
    return out


_SCRIPT_REF = {}


def drill_version():
    """sim_version() and the drills' code (tools/nn/train/drills): what a drill's script battles depend on."""
    h = hashlib.sha256(sim_version().encode())
    for f in sorted((ROOT / "tools/nn/train/drills").glob("*.py")):
        h.update(f.name.encode())
        h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:12]


def drill_scripts(name, n, device="cpu"):
    """{"naive", "skilled"}: the drill's two check scripts' win rate and gold trade on the evaluation's
    battles (DRILL_EVAL_SEEDS, our side alternating; SPREAD), cached per process and on disk
    (BASELINES/drill_<name>_<n>_broad<share>[_embed<share>]_<drill_version>.json: the broad and embedded
    frames' long battles make them ~15 min of a test5 start). With more than one frame (drills.FRAMES) in
    the battles, per frame too: {frame: {games, win_rate, gold_trade}}."""
    from tools.nn.train import drills as D
    from tools.nn.train.drills import verify
    embed = D.EMBED if D.load([name])[name].embedded is not None else 0.0
    key = (name, n, str(device), D.BROAD, embed)
    path = BASELINES / f"drill_{name}_{n}_broad{D.BROAD:g}{f'_embed{embed:g}' if embed else ''}_{drill_version()}.json"
    if key not in _SCRIPT_REF:
        try:
            _SCRIPT_REF[key] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    if key not in _SCRIPT_REF:
        drill = D.load([name])[name]
        seeds = range(D.DRILL_EVAL_SEEDS.start, D.DRILL_EVAL_SEEDS.start + n)
        out = {}
        from tools.nn.train.drills import metrics as drill_metrics
        for which in ("naive", "skilled"):
            box = {}

            def extra(st, box=box):
                if "tr" not in box:
                    box["tr"] = drill_metrics.Tracker(st, box["ours"], drill.roles)
                box["tr"].update(st, ~box.get("done", st.done))
                box["done"] = st.done.clone()
                box["st"] = st
            res, descs = verify.play(drill, getattr(drill, which), device=device, spread=SPREAD, seeds=seeds,
                                     extra=extra, ours_box=box)
            out[which] = {"win_rate": round(float(res["won"].mean()), 3), "gold_trade": round(float(res["trade"].mean()), 3),
                          "play": box["tr"].summary(None, box["st"]) if "tr" in box else None}
            kinds = res["frame"]
            out[which]["frames"] = {k: int((kinds == k).sum()) for k in D.FRAMES if (kinds == k).any()}
            if len(out[which]["frames"]) > 1:
                for tag in out[which]["frames"]:
                    sel = kinds == tag
                    out[which][tag] = {"games": int(sel.sum()), "win_rate": round(float(res["won"][sel].mean()), 3),
                                       "gold_trade": round(float(res["trade"][sel].mean()), 3)}
        _SCRIPT_REF[key] = out
        BASELINES.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, separators=SEP), encoding="utf-8", newline="\n")
    return _SCRIPT_REF[key]


@torch.no_grad()
def play_drills(actor, n=128, device="cpu", names=None, greedy=False, seed=1, compile=None, scripts=True, cadence=None):
    """The drills (tools/nn/train/drills; default the verified ones, drills.READY): n battles of each on
    DRILL_EVAL_SEEDS (our side alternating, SPREAD) against the drill's enemy script -> {drill: {games, wins,
    win_rate, gold_trade (mean (enemy gold destroyed - own lost) / budget), gold_destroyed, gold_lost, seconds,
    timeouts, scripts: {naive, skilled: {win_rate, gold_trade}} (scripts: the drill's check scripts on the same
    battles), frames: {frame: battles} (drills.FRAMES: clean, broad, embedded; drills.BROAD, drills.EMBED), with
    more than one frame in the battles {frame: {games, win_rate, gold_trade}} of each}}; {} without drills. cadence: the network's (tools/nn/train/cadence.py; default the game's)."""
    from tools.nn.train import drills as D
    from tools.nn.train.drills import metrics as drill_metrics
    from tools.nn.train.drills import source as drill_source
    names = [x for x in D.NAMES if x in (names if names is not None else D.READY)]
    if not names or n <= 0:
        return {}
    params = rollout.params_with_limit(None)
    src, lay = drill_source.evaluation(names, n, params, device)
    env = rollout.Battles(lay, device=device, params=params, spread=SPREAD, seed=seed, auto_reset=False,
                          compile=compile, source=src, cadence=cadence)
    steps = int(params.limit_s / env.decision_s) + 2
    # what our units do in each drill's battles (drills/metrics.py; the drill's roles: correct / bad targets)
    loaded = D.load(names)
    side_t = torch.as_tensor(lay.learner, device=env.device)
    opp_t = torch.as_tensor(lay.opponent, device=env.device)
    trackers = {n: drill_metrics.Tracker(env.st, torch.where(opp_t == league.CODE[D.opponent(n)], side_t,
                                                             torch.zeros_like(side_t)), loaded[n].roles,
                                                limit_s=params.limit_s)
                for n in names}
    for i in range(steps):
        if i % CHECK_EVERY == 0 and bool(env.st.done.all()):
            break
        def track(live):
            for tr in trackers.values():
                tr.update(env.st, live)
        env.step(actor, None, greedy, each=track)
    side = lay.learner
    B = env.B
    won = env.st.winner.cpu().numpy() == side
    gold_b = reward.gold_sides(env.st.u, env.weights.rout_share).cpu().numpy()
    own, enemy = gold_b[np.arange(B), side - 1], gold_b[np.arange(B), 2 - side]
    bud = reward.budget(env.st.u).cpu().numpy()
    t = env.st.t.cpu().numpy()
    out = {}
    for name in names:
        this = lay.opponent == league.CODE[D.opponent(name)]
        out[name] = {"games": int(this.sum()), "wins": int(won[this].sum()), "win_rate": float(won[this].mean()),
                     "gold_trade": float(((enemy - own) / np.maximum(bud, 1e-9))[this].mean()),
                     "gold_destroyed": float(enemy[this].mean()), "gold_lost": float(own[this].mean()),
                     "seconds": float(t[this].mean()), "timeouts": float((t[this] >= params.limit_s - 1e-6).mean())}
        out[name]["play"] = trackers[name].summary(this, env.st)
        kinds = src.frame[:B]
        out[name]["frames"] = {k: int((this & (kinds == k)).sum()) for k in D.FRAMES if (this & (kinds == k)).any()}
        if len(out[name]["frames"]) > 1:
            trade = (enemy - own) / np.maximum(bud, 1e-9)
            for tag in out[name]["frames"]:
                sel = this & (kinds == tag)
                out[name][tag] = {"games": int(sel.sum()), "win_rate": float(won[sel].mean()),
                                  "gold_trade": float(trade[sel].mean())}
        u = env.st.u
        up = (u["side"] > 0) & (u["men"] > 0) & ~u["gone"] & ~u["r"]
        ours_b = u["side"] == torch.as_tensor(side, device=env.device)[:, None]
        sel = torch.as_tensor(this, device=env.device)
        out[name]["standing_end"] = {"ours": float((up & ours_b).sum(1)[sel].float().mean()),
                                     "enemy": float((up & ~ours_b).sum(1)[sel].float().mean())}
        if scripts:
            out[name]["scripts"] = drill_scripts(name, n, device)
    return out


def start_cost(u):
    """[B, 2] numpy: each side's starting gold (the cost of its units)."""
    return torch.stack([(u["cost"] * (u["side"] == s)).sum(1) for s in (1, 2)], 1).cpu().numpy()


# --- the script baseline: the opponent script against itself on the same battles --------------------

def sim_version():
    """A hash of what a script-vs-script battle depends on: VERSION_FILES (line ends ignored) and the
    generator's budget factors as loaded (tools/nn/train/version.py, shared with the host-side tools)."""
    return sim_ver.sim_version()


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
    full, orig, factions = env.st, torch.arange(env.B, device=env.device), env.setup.factions
    n_steps, i = int(limit_s / env.params.dt) + 2, 0
    while i < n_steps:                                   # as play(compact=True): the running battles only
        n = int((~env.st.done).sum())
        if n == 0:
            break
        size = bucket(n, env.B)
        if size < env.B:
            _put(full, orig, env.st)
            keep = env.narrow(_ended(env, size))
            orig, ctrl = orig[keep], ctrl[keep]
        for _ in range(min(CHECK_EVERY, n_steps - i)):
            env.advance(env.st, env._assemble(env.st, ctrl, plays, ()), env.params, env.params.dt)
            env.st.u["lost_worst"] = env._track(env.st.u, env.weights.rout_share)   # as Battles: a loss counts once
            i += 1
    _put(full, orig, env.st)
    env.st = full
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
                  "factions": [list(f) for f in factions[r]], "attacker": attacker[r].tolist()}
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
    if missing and CANARY and 0 < CANARY < n_pairs:
        # the canary: the first CANARY pairs now, compared with the older versions' files (the newest
        # first); identical in every field -> that file is the baseline of this version too
        seeds = eval_seeds(n_pairs)
        old = {n: sim_ver.older_baselines(BASELINES, n, max_units, limit_s, version, seeds) for n in missing}
        names = [n for n in missing if old[n]]
        if names:
            print(f"baselines {', '.join(names)} missing for simulator version {version}: playing a canary of "
                  f"{CANARY} pairs against the older files", flush=True)
            BASELINES.mkdir(parents=True, exist_ok=True)
            for n, canary in script_battles(names, CANARY, max_units, limit_s, device, compile).items():
                for p, doc in old[n]:
                    if sim_ver.same_prefix(doc, canary, CANARY):
                        path = baseline_path(n, max_units, limit_s, version)
                        path.write_text(json.dumps(doc, separators=SEP), encoding="utf-8", newline="\n")
                        out[n] = dict({k: v[:n_pairs] for k, v in doc.items()}, cache=path.name)
                        missing.remove(n)
                        print(f"baseline {n}: adopted {p.name} (the {CANARY} canary pairs identical)", flush=True)
                        break
                else:
                    print(f"baseline {n}: no older file matches the canary; playing all {n_pairs} pairs", flush=True)
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
           scene_list=scenes.SCENES, compile=None, cadence=None):
    """Every scene from both sides against each opponent, written down per second; -> run folders.
    cadence: the network's (tools/nn/train/cadence.py; default the game's)."""
    lay = combined(opponents, 2, len(scene_list), scenes.attackers(scene_list), league.ATTACK_ONLY)
    env = rollout.Battles(lay, scene_list, device=device, params=rollout.params_with_limit(limit_s), spread=SPREAD,
                          seed=seed, auto_reset=False, compile=compile, cadence=cadence)
    rec = check.Recorder(env.st)
    for _ in range(int(limit_s / env.decision_s) + 2):
        if bool(env.st.done.all()):
            break
        env.step(actor, None, False, each=lambda live: rec(env.st))
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
    cad.add_args(ap)
    args = ap.parse_args()
    actor = checkpoint.load_policy(args.checkpoint, args.device)
    past = checkpoint.load_policy(args.against, args.device)
    res = play(actor, opponents=tuple(args.opponents.split(",")), per_scene=args.per_scene, past=past, device=args.device, limit_s=args.limit, greedy=args.greedy,
               hold_defend=args.hold_defend, generated=args.generated, max_units=args.max_units, together=args.together,
               paired=not args.no_pairs, baseline=not args.no_baseline, cadence=cad.of_args(args))
    from tools.nn.train.run import show
    show(Path(args.checkpoint).name, res)
    print("\n".join(skill.text(skill.summary(res))), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()

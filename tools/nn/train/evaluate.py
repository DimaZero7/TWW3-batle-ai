"""Evaluation of a network in the simulator: whole battles against fixed opponents, no training
(docs/en/training/training.md), and a few battles written down like the game's recordings.

    python -m tools.nn.train.evaluate --checkpoint build/nn-train/latest.pt      # in the container

play(): per opponent, `per_scene` battles of every scene (the learner's side alternating), all
in one batch until every battle ends. The numbers are the simulator's own (no randomisation),
the start places moved by up to 2 m, the learner's orders sampled (greedy=False) as in training.
Results by the learner's role (attack, defend).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from tools.nn.model import factions as fac
from tools.nn.sim import check, scenario
from tools.nn.sim import state as S
from tools.nn.train import checkpoint, league, randomise, reward, rollout, scenes

SPREAD = randomise.Spread(common=0.0, side=0.0, jitter_m=2.0)
OPPONENTS = ("nearest", "hold_shoot", "hold", "ai_like", "past")
REPLAYS = checkpoint.DIR / "replays"
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
         seed=1, scene_list=scenes.SCENES, compile=None, hold_defend=False, generated=None, max_units=19, small=None):
    """-> {"by_opponent": {name: {games, wins, win_rate, seconds, hp_own_lost, hp_enemy_lost, timeouts,
    contact_share, first_contact_s, fired_share, orders_per_minute, kinds, roles: {attack|defend: {games,
    wins, win_rate, seconds, hp_own_lost, hp_enemy_lost, timeouts}}}}, "by_scene": {name: {label: {games,
    wins, win_rate}}}, "orders_per_minute", "kinds"}. One batch per opponent. past: the actor behind "past"
    (e.g. the untrained one). `hold` is met only as the defender unless hold_defend: as the attacker it
    never moves, and the battle only waits out the limit. generated: that many battles of random armies
    (tools/nn/armies, EVAL_SEEDS, up to max_units units a side) per opponent instead of the scenes."""
    out = {"by_opponent": {}, "by_scene": {}, "limit_s": limit_s, "greedy": greedy, "per_scene": per_scene}
    for name in opponents:
        only = () if hold_defend else league.ATTACK_ONLY
        mine, rows = _play_one(actor, name, per_scene, past, device, limit_s, greedy, seed, scene_list, compile, only,
                               generated, max_units, small)
        out["by_opponent"][name] = mine
        out["by_scene"][name] = rows
    ops = list(out["by_opponent"].values())
    out["orders_per_minute"] = float(np.mean([o["orders_per_minute"] for o in ops]))
    out["kinds"] = {k: float(np.mean([o["kinds"][k] for o in ops])) for k in ops[0]["kinds"]}
    return out


def _summary(sel, won, t, hp_own, hp_enemy, limit_s, lord_own=None, lord_enemy=None):
    n = int(sel.sum())
    if not n:
        return {"games": 0}
    out = {"games": n, "wins": int(won[sel].sum()), "win_rate": float(won[sel].mean()),
           "seconds": float(t[sel].mean()), "hp_own_lost": float(hp_own[sel].mean()),
           "hp_enemy_lost": float(hp_enemy[sel].mean()), "timeouts": float((t[sel] >= limit_s - 1e-6).mean())}
    if lord_own is not None:
        out["lord_dead_own"] = float(lord_own[sel].mean())
        out["lord_dead_enemy"] = float(lord_enemy[sel].mean())
    return out


def generated_layout(source, opponent, attack_only=league.ATTACK_ONLY):
    """Battle b = bank battle b; the learner's side alternates every two battles (the bank alternates
    who attacks), so both roles and both sides come up; an attack_only opponent only defends."""
    n = source.bank.M
    side = 1 + (np.arange(n) // 2) % 2
    if opponent in attack_only:
        side = source.bank.attacker.cpu().numpy()
    return league.Layout(np.zeros(n, dtype=int), side, np.full(n, league.CODE[opponent]))


def _play_one(actor, name, per_scene, past, device, limit_s, greedy, seed, scene_list, compile, attack_only,
              generated=None, max_units=19, small=None):
    params = rollout.params_with_limit(limit_s)
    if generated:
        from tools.nn.armies import generate
        source = scenes.Generated(range(generate.EVAL_SEEDS.start, generate.EVAL_SEEDS.start + generated),
                                  max_units, params, device, sequential=True, small=small)
        lay = generated_layout(source, name, attack_only)
    else:
        source = None
        lay = combined((name,), per_scene, len(scene_list), scenes.attackers(scene_list), attack_only)
    env = rollout.Battles(lay, scene_list, device=device, params=params, spread=SPREAD,
                          seed=seed, auto_reset=False, compile=compile, source=source)
    if name == "past":
        env.set_past(past)
    B = env.B
    learner_side = torch.as_tensor(lay.learner, device=env.device)
    mine = env.st.u["side"] == learner_side[:, None]
    contact = torch.full((B,), -1.0, device=env.device)
    fired = torch.zeros(B, dtype=torch.bool, device=env.device)
    for _ in range(int(limit_s / env.params.dt) + 2):
        if bool(env.st.done.all()):
            break
        env.step(actor, None, greedy)
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
    attacks = env.st.attacker.cpu().numpy() == side
    c = contact.cpu().numpy()
    dead = env.st.lord_dead_s.cpu().numpy() >= 0                      # [B, side]
    lords = (dead[np.arange(B), side - 1], dead[np.arange(B), 2 - side])
    every = np.ones(B, dtype=bool)
    switches = env.lords()["switches_per_minute"]
    result = _summary(every, won, t, hp_own, hp_enemy, limit_s, *lords)
    result.update({"contact_share": float((c >= 0).mean()),
                   "first_contact_s": float(np.median(c[c >= 0])) if (c >= 0).any() else None,
                   "fired_share": float(fired.cpu().numpy().mean()),
                   "orders_per_minute": env.orders_per_minute(), "switches_per_minute": switches, "kinds": env.kinds(),
                   "roles": {"attack": _summary(attacks, won, t, hp_own, hp_enemy, limit_s, *lords),
                             "defend": _summary(~attacks, won, t, hp_own, hp_enemy, limit_s, *lords)}})
    rows = {}
    if generated:
        fac = env.setup.factions
        for b in range(B):
            own, enemy = fac[b][side[b] - 1], fac[b][2 - side[b]]
            key = f"{faction_name(own)} v {faction_name(enemy)}, {'attack' if attacks[b] else 'defend'}"
            r = rows.setdefault(key, {"games": 0, "wins": 0})
            r["games"] += 1
            r["wins"] += int(won[b])
        for r in rows.values():
            r["win_rate"] = r["wins"] / max(1, r["games"])
        return result, rows
    for i, sc in enumerate(scene_list):
        for s in (1, 2):
            k = (lay.scene == i) & (side == s)
            if not k.any():
                continue
            r = rows.setdefault(label(sc, s), {"games": 0, "wins": 0})
            r["games"] += int(k.sum())
            r["wins"] += int(won[k].sum())
    for r in rows.values():
        r["win_rate"] = r["wins"] / max(1, r["games"])
    return result, rows


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
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--out", help="write the results as json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    actor = checkpoint.load_policy(args.checkpoint, args.device)
    past = checkpoint.load_policy(args.against, args.device)
    res = play(actor, opponents=tuple(args.opponents.split(",")), per_scene=args.per_scene, past=past, device=args.device, limit_s=args.limit, greedy=args.greedy,
               hold_defend=args.hold_defend, generated=args.generated, max_units=args.max_units)
    from tools.nn.train.run import show
    show(Path(args.checkpoint).name, res)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()

"""The battles of an in-game gate played again in the simulator from the same recorded starts
(docs/en/training/workflow.md "Game-vs-sim gap card"; tools/ops/gapcard.py prints the card).

Each gate battle's first recorded sample (tools/nn/sim/scenario.py from_recording: places, bearings,
men, widths, factions; side 1 = our network's army in the game, side 2 = the game AI's) starts
`copies` simulated battles: side 1 played by the gate's checkpoint (live, sampled unless the gate was
greedy), side 2 by the ai_like script, at the game's cadence (a decision a second, the orders ~0.36 s
late: tools/nn/train/cadence.py GAME), the copies differing by the start's 2 m jitter
(evaluate.SPREAD) and the sampled orders. Every copy is recorded once a simulated second as the game
records (tools/nn/sim/check.py Recorder -> gamedata.Battle in the recording's unit order), with the
times of our lord's ability uses (the cooldown marks, behaviour.ability_marks), so the same code
(tools/nn/battle_metrics.py) measures the game and the simulator.

The closed loop moved here from the gap analyses' replay scripts (the network against a script from a
recorded start); their probes (perturbed numbers, recorded-order replay, observation tweaks) stay there.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from tools import config as project
from tools.nn import gamedata
from tools.nn.sim import check, scenario
from tools.nn.sim import orders as O
from tools.nn.train import behaviour, checkpoint, evaluate, league, rollout, scenes
from tools.nn.train import cadence as cad

RUNS = project.BUILD / "nn-arena" / "runs"
OPPONENT = "ai_like"


class Recorder(check.Recorder):
    """check.Recorder, also the orders in force (order_kind, order_run) at each recorded second."""

    def take(self, st):
        super().take(st)
        if not hasattr(self, "orders"):
            self.orders = []
        self.orders.append({k: st.u[k].detach().cpu().numpy().copy() for k in ("order_kind", "order_run")})

    def orders_of(self, b, slot_of):
        """{"move": [T, N], "run": [T, N]} of battle b in the recording's unit order (as battle())."""
        end = self.done_at[b] or len(self.rows)
        idx = np.array(slot_of)
        kind = np.stack([r["order_kind"][b, idx] for r in self.orders[:end + 1]])
        run = np.stack([r["order_run"][b, idx] for r in self.orders[:end + 1]]).astype(bool)
        return {"move": (kind == O.MOVE) | (kind == O.WITHDRAW), "run": run}


class Bank:
    """The source of rollout.Battles: the given armies, battle b = army b (no random pick)."""

    def __init__(self, armies, params, device, H, names):
        self.bank = scenes.Bank(armies, params, device, H, names=names)

    def pick(self, want=None):
        return torch.arange(self.bank.M, device=self.bank.state.device)


def gate_battles(gate_dir, runs=RUNS, only=None):
    """[(summary row, run folder)] of a gate folder's battles with a recording (summary.json, else
    battles.json); only: battle numbers to keep."""
    gate_dir = Path(gate_dir)
    doc = {}
    for name in ("summary.json", "battles.json"):
        p = gate_dir / name
        if p.is_file():
            doc = json.loads(p.read_text(encoding="utf-8-sig"))
            break
    out = []
    for row in doc.get("battles", []):
        run = row.get("run")
        if not run or (only and row["battle"] not in only):
            continue
        d = Path(run) if Path(run).is_absolute() else Path(runs) / Path(run).name
        if (d / "events.jsonl").is_file():
            out.append((row, d))
    return doc, out


def start_of(run_dir):
    """(gamedata.Battle, army description) of a game recording: the simulator's start."""
    g = gamedata.load(run_dir)
    army = scenario.from_recording(g, run_dir)
    fac = check.factions_of(run_dir, g.arena)
    army["sides"][1]["faction"] = fac.get("own")
    army["sides"][2]["faction"] = fac.get("enemy")
    return g, army


@torch.no_grad()
def play(starts, ckpt, copies=8, seed=1, greedy=False, device="cpu", limit_s=3600.0, cadence=None, log=print):
    """starts: [(gamedata.Battle, army)] -> per start [{"battle": gamedata.Battle, "winner": 1 | 2 | 0,
    "t": end s, "abilities": [s of our lord's ability uses], "orders": {"move", "run"} [T, N] in force}] of
    its copies."""
    params = rollout.params_with_limit(limit_s)
    H = max(len(a["sides"][s]["units"]) for _, a in starts for s in (1, 2))
    armies, game_of, names = [], [], []
    for k, (_, army) in enumerate(starts):
        for c in range(copies):
            armies.append(army)
            game_of.append(k)
            names.append(f"g{k}c{c}")
    B = len(armies)
    code = league.CODE[OPPONENT]
    lay = league.Layout(np.zeros(B, dtype=int), np.ones(B, dtype=int), np.full(B, code))
    env = rollout.Battles(lay, device=device, params=params, spread=evaluate.SPREAD, seed=seed, auto_reset=False,
                          compile=False, source=Bank(armies, params, device, H, names), cadence=cadence or cad.GAME)
    actor = checkpoint.load_policy(ckpt, device)
    torch.manual_seed(seed)                          # the sampled orders (the global generator) repeat
    rec = Recorder(env.st)
    u = env.st.u
    lord1 = u["lord"] & (u["side"] == 1)
    marks = behaviour.ability_marks(u)
    uses = [[] for _ in range(B)]
    n_dec = int(params.limit_s / env.cadence.decide_s) + 2
    t0 = time.time()
    for i in range(n_dec):
        if bool(env.st.done.all()):
            break
        live = ~env.st.done.clone()
        env.step(actor, None, greedy)                  # one decision: cadence.steps simulator steps
        new = behaviour.ability_marks(env.st.u)
        used = ((new > marks + 1e-3) & lord1[..., None]).sum((1, 2)) * live
        for b in torch.nonzero(used).flatten().tolist():
            uses[b] += [float(env.st.t[b])] * int(used[b])
        marks = new
        rec(env.st)
        if i % 200 == 0:
            log(f"gapsim: decision {i}, {int((~env.st.done).sum())} of {B} battles running, {time.time() - t0:.0f} s")
    winner = env.st.winner.cpu().numpy()
    t_end = env.st.t.cpu().numpy()
    out = [[] for _ in starts]
    for b in range(B):
        k = game_of[b]
        g, army = starts[k]
        m = scenario.slots(army, H)
        slot_of = [m[n] for n in g.names]
        bt = rec.battle(b, g.names, g.keys, g.side, slot_of, winner[b], g.own_ai, g.arena)
        out[k].append({"battle": bt, "winner": int(winner[b]), "t": float(t_end[b]), "abilities": uses[b],
                       "orders": rec.orders_of(b, slot_of)})
    return out

"""The companion: waits for the game's state file, runs the network, writes the orders file.

    bash tools/nn/dock.sh ...   (see docs/en/launch/watch.md: the watch launcher starts it in the
    snake-ai-trainer container with the game folder mounted at /game)
    python -m tools.nn.companion --game /game [--checkpoint build/nn-train/random.pt] [--greedy]
        [--enemy-script ai_like]   (also the enemy side: the script answers the enemy's bridge, script.py)

One line per decision: move number, battle time, how long the network took, the orders.
A new battle (a new batch in the state) starts the network's memory afresh (and a v2 network's commitment: its
units held by their commitment keep, an event in the game's state - melee, a threat, the target gone, a rout or rally,
the lord's death - frees them; tools/nn/model/commit.py).
"""
import argparse
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

from tools.nn.companion import exchange
from tools.nn.companion import policy as companion_policy
from tools.nn.model import decide
from tools.nn.model import observation as ob

SIDE = 1   # our side in the arena (src/entries/nn_arena.lua)


class Brain:
    """The network with its memory for the current battle."""

    def __init__(self, actor, greedy=False, temperature=1.0):
        self.actor, self.greedy, self.temperature = actor, greedy, temperature
        self.battle, self.memory, self.h = None, None, None
        self.given, self.points = {}, {}     # the orders in force, the order points (exchange.order_points)
        self.moved = None                    # the last positions and time (exchange.running_by_speed)
        self.commit = {}                     # v2: the commitment's state (tools/nn/model/commit.py), per battle
        self.before = None                   # (batch, move, the orders in force before that move's answer)

    def orders_before(self, batch, move):
        """The network's orders in force as the simulator's opponent sees them at decision `move`: before
        this move's answer (both sides decide on the same state). {} for another battle."""
        if self.before is not None and self.before[:2] == (batch, move):
            return self.before[2]
        if self.battle is None or self.battle.batch != batch:
            return {}
        return dict(self.given)

    def decide(self, doc):
        """-> (orders list, think ms, abilities to use [{unit, key}]) for one state document."""
        t0 = time.perf_counter()
        if self.battle is None or self.battle.batch != doc["batch"]:
            self.battle, self.memory, self.h = exchange.battle(doc), None, None
            self.given, self.points, self.moved, self.commit = {}, {}, None, {}
        b = self.battle
        self.before = (doc["batch"], doc["move"], dict(self.given))
        state = exchange.arrays(doc, b.names, b.slots)
        self.moved = exchange.running_by_speed(state, b.walk, self.moved)
        exchange.engaged_targets(state, b.side, shape=b.shape)
        self.points = exchange.order_points(state, b.names, b.side, self.given, self.points)
        obs, self.memory = ob.observe(state, b.setup, SIDE, self.memory)
        orders, self.h, _, _ = decide.act(self.actor, obs, b.setup, self.h, self.greedy, self.temperature,
                                          commit_state=self.commit, t=state["t"])
        cols = [getattr(orders, k)[0].cpu().numpy() for k in ("kind", "x", "z", "target", "run")]
        out = exchange.orders_list(b.names, b.side, *cols)
        ctrl = np.asarray(obs.ctrl[0])
        ctrl_names = {n for n, c in zip(b.names, ctrl) if c}
        for o in out:
            if o["unit"] not in ctrl_names:
                o["out"] = True     # dead, routing or shattered: the network's HOLD is no order (summary)
        exchange.remember_orders(self.given, out, ctrl_names)
        uses = exchange.ability_list(b.names, b.side, orders.ability[0].cpu().numpy(), b.slots)
        return out, (time.perf_counter() - t0) * 1000, uses


def describe(orders, limit=4):
    parts = []
    for o in orders[:limit]:
        name = o["unit"].removeprefix("own_").removeprefix("enemy_")
        if o["kind"] in ("move", "withdraw"):
            parts.append(f"{name} {o['kind']}({o['x']:.0f},{o['z']:.0f}){'!' if o['run'] else ''}")
        elif o["kind"] == "attack":
            parts.append(f"{name} attack {o['target'].removeprefix('enemy_')}")
        else:
            parts.append(f"{name} {o['kind']}")
    more = f" +{len(orders) - limit}" if len(orders) > limit else ""
    return ", ".join(parts) + more


BUSY_S = 5.0     # after a state, look for the next one every poll_s; later (loading, menus) every idle_poll_s


class Seat:
    """One side the companion answers: its state and orders files, who decides, its log and line prefix."""

    def __init__(self, state, orders, decide, log=None, label=""):
        self.state, self.orders, self.decide, self.log, self.label = Path(state), Path(orders), decide, log, label
        first = exchange.read_state(self.state)
        self.last = (first["batch"], first["move"]) if first else None   # a file of an earlier run: not answered
        self.done = False


def run(game, brain, log=None, poll_s=0.005, idle_poll_s=0.1, exit_on_done=False, idle_exit_s=0, out=print,
        enemy=None, enemy_log=None):
    """The loop. Returns when the battle is done (exit_on_done) or nothing new came for idle_exit_s.
    enemy: a script.ScriptSide that answers the enemy's bridge too (exchange.ENEMY_STATE / ENEMY_ORDERS),
    after our side's answer to the same move, seeing our orders in force before it (Brain.orders_before)."""
    seats = [Seat(Path(game) / exchange.STATE, Path(game) / exchange.ORDERS, brain.decide, log)]
    if enemy is not None:
        seats.append(Seat(Path(game) / exchange.ENEMY_STATE, Path(game) / exchange.ENEMY_ORDERS,
                          lambda doc: enemy.decide(doc, brain.orders_before(doc["batch"], doc["move"])),
                          enemy_log, "enemy "))
    last_new, busy_until = time.monotonic(), 0.0
    while True:
        new = False
        for seat in seats:
            doc = exchange.read_state(seat.state)
            if doc is None or (doc["batch"], doc["move"]) == seat.last:
                continue
            new = True
            seen = time.perf_counter()
            seat.last, last_new = (doc["batch"], doc["move"]), time.monotonic()
            busy_until = last_new + BUSY_S
            if doc.get("done"):
                out(f"{seat.label}battle {doc['batch']} done after move {doc['move']}")
                seat.done = True
                continue
            seat.done = False
            orders, think_ms, uses = seat.decide(doc)
            attempts = exchange.write_atomic(seat.orders, exchange.orders_text(doc["batch"], doc["move"], orders,
                                                                               think_ms, uses))
            turn_ms = (time.perf_counter() - seen) * 1000
            used = "".join(f" | ability {a['unit'].removeprefix('own_').removeprefix('enemy_')} {a['key']}"
                           for a in uses)
            out(f"{seat.label}move {doc['move']:4d}  t {doc.get('t', 0) / 1000:6.1f} s  think {think_ms:5.1f} ms  "
                f"turn {turn_ms:5.1f} ms  | {exchange.summary(orders)} | {describe(orders)}{used}")
            if seat.log:
                with open(seat.log, "a", encoding="utf-8", newline="\n") as f:
                    f.write(json.dumps({"batch": doc["batch"], "move": doc["move"], "t": doc.get("t"),
                                        "think_ms": round(think_ms, 2), "turn_ms": round(turn_ms, 2),
                                        "write_attempts": attempts, "orders": orders, "abilities": uses}) + "\n")
        if exit_on_done and all(seat.done for seat in seats[:1]):
            return
        if not new:
            if idle_exit_s and time.monotonic() - last_new > idle_exit_s:
                out(f"nothing new for {idle_exit_s} s: stopping")
                return
            time.sleep(poll_s if time.monotonic() < busy_until else idle_poll_s)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game", required=True, help="the game folder (where the state file appears)")
    parser.add_argument("--checkpoint", help="a training checkpoint (default: build/nn-train/random.pt if "
                                             "present, else a fresh untrained actor)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the fresh actor")
    parser.add_argument("--greedy", action="store_true", help="the most likely order instead of sampling")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for torch")
    parser.add_argument("--log", help="append every decision as a JSON line to this file")
    parser.add_argument("--exit-on-done", action="store_true", help="stop when the battle's last state comes")
    parser.add_argument("--idle-exit", type=float, default=0, help="stop after this many seconds without a state")
    parser.add_argument("--enemy-script", help="also command the enemy side by this simulator script "
                                               "(tools/nn/train/opponents.py, e.g. ai_like): nn-arena --enemy-ai")
    parser.add_argument("--enemy-log", help="append every enemy decision as a JSON line to this file "
                                            "(default: the --log file's name with _enemy)")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    torch.set_num_threads(args.threads)
    actor, what = companion_policy.load(args.checkpoint, args.seed)
    print(f"companion: {what}; {'greedy' if args.greedy else f'sampling, temperature {args.temperature}'}; "
          f"{args.threads} threads; watching {Path(args.game) / exchange.STATE}")
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    enemy, enemy_log = None, None
    if args.enemy_script:
        from tools.nn.companion.script import ScriptSide
        enemy = ScriptSide(args.enemy_script)
        enemy_log = args.enemy_log or (str(Path(args.log).with_name(Path(args.log).stem + "_enemy.jsonl"))
                                       if args.log else None)
        print(f"companion: the enemy side by the script {args.enemy_script}; watching "
              f"{Path(args.game) / exchange.ENEMY_STATE}")
    try:
        run(args.game, Brain(actor, args.greedy, args.temperature), args.log, exit_on_done=args.exit_on_done,
            idle_exit_s=args.idle_exit, enemy=enemy, enemy_log=enemy_log)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

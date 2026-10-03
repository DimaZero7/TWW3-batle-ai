"""The companion: waits for the game's state file, runs the network, writes the orders file.

    bash tools/nn/dock.sh ...   (see docs/en/launch/watch.md: the watch launcher starts it in the
    snake-ai-trainer container with the game folder mounted at /game)
    python -m tools.nn.companion --game /game [--checkpoint build/nn-train/random.pt] [--greedy]

One line per decision: move number, battle time, how long the network took, the orders.
A new battle (a new batch in the state) starts the network's memory afresh.
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

    def decide(self, doc):
        """-> (orders list, think ms, abilities to use [{unit, key}]) for one state document."""
        t0 = time.perf_counter()
        if self.battle is None or self.battle.batch != doc["batch"]:
            self.battle, self.memory, self.h = exchange.battle(doc), None, None
            self.given, self.points, self.moved = {}, {}, None
        b = self.battle
        state = exchange.arrays(doc, b.names, b.slots)
        self.moved = exchange.running_by_speed(state, b.walk, self.moved)
        exchange.engaged_targets(state, b.side)
        self.points = exchange.order_points(state, b.names, b.side, self.given, self.points)
        obs, self.memory = ob.observe(state, b.setup, SIDE, self.memory)
        orders, self.h, _, _ = decide.act(self.actor, obs, b.setup, self.h, self.greedy, self.temperature)
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
        name = o["unit"].removeprefix("own_")
        if o["kind"] in ("move", "withdraw"):
            parts.append(f"{name} {o['kind']}({o['x']:.0f},{o['z']:.0f}){'!' if o['run'] else ''}")
        elif o["kind"] == "attack":
            parts.append(f"{name} attack {o['target'].removeprefix('enemy_')}")
        else:
            parts.append(f"{name} {o['kind']}")
    more = f" +{len(orders) - limit}" if len(orders) > limit else ""
    return ", ".join(parts) + more


BUSY_S = 5.0     # after a state, look for the next one every poll_s; later (loading, menus) every idle_poll_s


def run(game, brain, log=None, poll_s=0.005, idle_poll_s=0.1, exit_on_done=False, idle_exit_s=0, out=print):
    """The loop. Returns when the battle is done (exit_on_done) or nothing new came for idle_exit_s."""
    state_path, orders_path = Path(game) / exchange.STATE, Path(game) / exchange.ORDERS
    first = exchange.read_state(state_path)
    last = (first["batch"], first["move"]) if first else None   # a file of an earlier run: not answered
    last_new, busy_until = time.monotonic(), 0.0
    while True:
        doc = exchange.read_state(state_path)
        if doc is None or (doc["batch"], doc["move"]) == last:
            if idle_exit_s and time.monotonic() - last_new > idle_exit_s:
                out(f"nothing new for {idle_exit_s} s: stopping")
                return
            time.sleep(poll_s if time.monotonic() < busy_until else idle_poll_s)
            continue
        seen = time.perf_counter()
        last, last_new = (doc["batch"], doc["move"]), time.monotonic()
        busy_until = last_new + BUSY_S
        if doc.get("done"):
            out(f"battle {doc['batch']} done after move {doc['move']}")
            if exit_on_done:
                return
            continue
        orders, think_ms, uses = brain.decide(doc)
        attempts = exchange.write_atomic(orders_path, exchange.orders_text(doc["batch"], doc["move"], orders, think_ms,
                                                                           uses))
        turn_ms = (time.perf_counter() - seen) * 1000
        used = "".join(f" | ability {a['unit'].removeprefix('own_')} {a['key']}" for a in uses)
        out(f"move {doc['move']:4d}  t {doc.get('t', 0) / 1000:6.1f} s  think {think_ms:5.1f} ms  "
            f"turn {turn_ms:5.1f} ms  | {exchange.summary(orders)} | {describe(orders)}{used}")
        if log:
            with open(log, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps({"batch": doc["batch"], "move": doc["move"], "t": doc.get("t"),
                                    "think_ms": round(think_ms, 2), "turn_ms": round(turn_ms, 2),
                                    "write_attempts": attempts, "orders": orders, "abilities": uses}) + "\n")


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
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    torch.set_num_threads(args.threads)
    actor, what = companion_policy.load(args.checkpoint, args.seed)
    print(f"companion: {what}; {'greedy' if args.greedy else f'sampling, temperature {args.temperature}'}; "
          f"{args.threads} threads; watching {Path(args.game) / exchange.STATE}")
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    try:
        run(args.game, Brain(actor, args.greedy, args.temperature), args.log, exit_on_done=args.exit_on_done,
            idle_exit_s=args.idle_exit)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""The companion: waits for the game's state file, runs the chain, writes the orders file; answers the strategist's
deployment document once a battle (bridge/docs/apps/bridge.md, bridge/docs/launch/watch.md).

    bash ops/dock.sh ...   (bridge/launcher/watch.ps1 starts it in the snake-ai-trainer container with the game folder
                            mounted at /game)
    python -m bridge.companion --game /game
        [--strategist [default|formula|<strategist checkpoint>] --strategist-task hold|destroy]
                (bridge.build nn-arena --deploy strategist: the strategist answers the deployment document once a
                 battle, bridge/companion/deploy.py)
        [--chain [--chain-op formula|net|<ckpt>] [--chain-tactic formula|net|<ckpt>] [--chain-unit <ckpt>]
                 [--chain-greedy]]
                (our side under the chain operative -> tacticians -> unit network: bridge/companion/chain.py; with
                 --strategist its groups start it). Without --chain only the deployment is answered.

One line per decision: move number, battle time, how long the chain took, the orders; the levels' ms, the operative's
groups when it decided. A new battle (a new batch in the state) starts the chain afresh.
"""
import argparse
import json
import signal
import sys
import time
from pathlib import Path

from bridge.companion import exchange
from bridge.config import resolve as _path


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
    """The side the companion answers: its state and orders files, who decides (chain.Chain), its log."""

    def __init__(self, state, orders, owner, log=None):
        self.state, self.orders, self.owner, self.log = Path(state), Path(orders), owner, log
        first = exchange.read_state(self.state)
        self.last = (first["batch"], first["move"]) if first else None   # a file of an earlier run: not answered
        self.done = False


class DeploySeat:
    """The deployment's document and plan (bridge/companion/deploy.py): answered once a battle (a new batch)."""

    def __init__(self, game, strategist, log=None):
        from bridge.companion import deploy
        self.deploy = deploy
        self.doc, self.plan = Path(game) / deploy.DOC, Path(game) / deploy.PLAN
        self.strategist, self.log = strategist, log
        first = exchange.read_state(self.doc)
        self.last = first["batch"] if first else None     # a file of an earlier run: not answered

    def poll(self, out):
        doc = exchange.read_state(self.doc)
        if doc is None or not doc.get("deploy") or doc["batch"] == self.last:
            return False
        self.last = doc["batch"]
        text, record = self.strategist.decide(doc)
        attempts = exchange.write_atomic(self.plan, text)
        out(self.deploy.describe(record))
        if self.log:
            with open(self.log, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(dict(record, event="deploy", write_attempts=attempts)) + "\n")
        return True


def answer(seat, doc, out=print):
    """One state document answered: the chain's orders written (exchange.orders_text, 'mode chain'), the log lines.
    Returns (orders, think ms, write attempts)."""
    seen = time.perf_counter()
    orders, think_ms, uses = seat.owner.decide(doc)
    attempts = exchange.write_atomic(seat.orders, exchange.orders_text(
        doc["batch"], doc["move"], orders, think_ms, uses, mode=getattr(seat.owner, "MODE", None)))
    turn_ms = (time.perf_counter() - seen) * 1000
    out(f"move {doc['move']:4d}  t {doc.get('t', 0) / 1000:6.1f} s  think {think_ms:5.1f} ms  "
        f"turn {turn_ms:5.1f} ms  | {exchange.summary(orders)} | {describe(orders)}")
    chain = seat.owner.log_extra() if hasattr(seat.owner, "log_extra") else None
    if chain is not None:
        ms = chain["ms"]
        out(f"  chain sec {chain['sec']}: raw {ms['raw_ms']:.0f} op {ms['op_ms']:.0f} "
            f"tac {ms['tac_ms']:.0f} unit {ms['unit_ms']:.0f} ms, inside {sum(chain['inside'].values())}"
            f"/{len(chain['inside'])}" + (f", leash {chain['leash']}" if chain.get("leash") else ""))
        if chain.get("operative") is not None:
            from bridge.companion import chain as companion_chain
            out(f"  operative: {companion_chain.describe(chain['operative'])}")
    if seat.log:
        with open(seat.log, "a", encoding="utf-8", newline="\n") as f:
            row = {"batch": doc["batch"], "move": doc["move"], "t": doc.get("t"),
                   "think_ms": round(think_ms, 2), "turn_ms": round(turn_ms, 2),
                   "write_attempts": attempts, "orders": orders, "abilities": uses}
            if chain is not None:
                row["chain"] = chain                     # the levels' ms, groups, areas, inside
            f.write(json.dumps(row) + "\n")
    return orders, think_ms, attempts


def run(game, chain=None, log=None, poll_s=0.005, idle_poll_s=0.1, exit_on_done=False, idle_exit_s=0, out=print,
        strategist=None):
    """The loop. Returns when the battle is done (exit_on_done) or nothing new came for idle_exit_s.
    chain: who answers our side's state (chain.Chain), None: nobody (only the deployment); strategist: who answers
    the deployment document (deploy.Strategist), None: nobody. No clock of its own: a paused battle writes no state,
    which is only a long wait (idle_exit_s 0: forever)."""
    seat = Seat(Path(game) / exchange.STATE, Path(game) / exchange.ORDERS, chain, log) if chain is not None else None
    deploy = DeploySeat(game, strategist, log) if strategist is not None else None
    assert seat or deploy, "nobody to answer"
    last_new, busy_until = time.monotonic(), 0.0
    while True:
        new = False
        if deploy is not None and deploy.poll(out):
            new, last_new = True, time.monotonic()
            busy_until = last_new + BUSY_S
        doc = exchange.read_state(seat.state) if seat is not None else None
        if doc is not None and (doc["batch"], doc["move"]) != seat.last:
            new = True
            seat.last, last_new = (doc["batch"], doc["move"]), time.monotonic()
            busy_until = last_new + BUSY_S
            if doc.get("done"):
                out(f"battle {doc['batch']} done after move {doc['move']}")
                seat.done = True
            else:
                seat.done = False
                answer(seat, doc, out)
        if exit_on_done and seat is not None and seat.done:
            return
        if not new:
            if idle_exit_s and time.monotonic() - last_new > idle_exit_s:
                out(f"nothing new for {idle_exit_s} s: stopping")
                return
            time.sleep(poll_s if time.monotonic() < busy_until else idle_poll_s)


def unit_network(path=None, device="cpu"):
    """The frozen unit network: path, or net/config/tactic.json train.unit_ckpt (the chain's default)."""
    from net.tactic import ckpt as TC
    return TC.unit_network(device, str(_path(path)) if path else None)


def load_chain(op="formula", tactic="formula", unit=None, greedy=False, plans=None, task="hold", device="cpu"):
    """(Chain, description) of companion --chain (the nets frozen, evaluation only): op / tactic 'formula', 'net' (the
    strategist's config: operative_ckpt, tactic_ckpt) or a checkpoint; unit None: the tactic config's unit_ckpt."""
    from bridge.companion.chain import Chain
    from net.operative import ckpt as OC
    from net.strategist import cfg as scfg
    from net.tactic import ckpt as TC
    from net.tactic import cfg as tcfg
    unit_net = unit_network(unit, device)
    op_net, op_what = None, "formula"
    if op != "formula":
        path = scfg()["operative_ckpt"] if op == "net" else op
        op_net, _, _ = OC.restore(str(_path(path)), device)
        op_net.eval()
        op_what = path
    tac_net, tac_what = None, "formula"
    if tactic != "formula":
        path = scfg()["tactic_ckpt"] if tactic == "net" else tactic
        tac_net, _, _ = TC.restore(str(_path(path)), device)
        tac_net.eval()
        tac_what = path
    chain = Chain(unit_net, op_net, tac_net, op_greedy=greedy, tactic_greedy=greedy, unit_greedy=greedy, plans=plans,
                  task=task, device=device)
    what = (f"operative {op_what}, tacticians {tac_what}, unit network {unit or tcfg()['train']['unit_ckpt']}, "
            f"{'greedy' if greedy else 'sampling'}")
    return chain, what


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game", required=True, help="the game folder (where the state file appears)")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for torch")
    parser.add_argument("--log", help="append every decision as a JSON line to this file")
    parser.add_argument("--exit-on-done", action="store_true", help="stop when the battle's last state comes")
    parser.add_argument("--idle-exit", type=float, default=0, help="stop after this many seconds without a state")
    parser.add_argument("--strategist", nargs="?", const="default",
                        help="the deployment by the strategist (bridge.build nn-arena --deploy strategist): no value or "
                             "'default' - net/config/strategist.json default, 'formula' or a strategist checkpoint, "
                             "greedy; answers the deployment document once a battle (bridge/companion/deploy.py)")
    parser.add_argument("--strategist-task", choices=("hold", "destroy"), default="hold",
                        help="the battle's task the strategist is given (no level above it yet)")
    parser.add_argument("--chain", action="store_true",
                        help="our side under the chain operative -> tacticians -> unit network (bridge/companion/"
                             "chain.py): the operative every 15 s, the tacticians every 5 s, the unit network every "
                             "second, as the simulator's chain; with --strategist its groups start the battle")
    parser.add_argument("--chain-op", default="formula",
                        help="the chain's operative: formula (default: every battle in the game so far), net "
                             "(net/config/strategist.json operative_ckpt; run8 has 6 groups, the config 40 - no network "
                             "of 40 groups yet) or an operative checkpoint")
    parser.add_argument("--chain-tactic", default="formula",
                        help="the chain's tacticians: formula (net/tactic/teacher.drive, the simulator's default), net "
                             "(net/config/strategist.json tactic_ckpt) or a tactic checkpoint")
    parser.add_argument("--chain-unit", default=None,
                        help="the unit network's checkpoint (default net/config/tactic.json train.unit_ckpt)")
    parser.add_argument("--chain-greedy", action="store_true",
                        help="every level of the chain takes its most likely choice (default: sampling, as the "
                             "simulator's chain)")
    args = parser.parse_args(argv)
    if not (args.chain or args.strategist):
        parser.error("nobody to answer: give --chain and / or --strategist")
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    import torch
    torch.set_num_threads(args.threads)
    strategist = None
    if args.strategist:
        from bridge.companion.deploy import Strategist
        strategist = Strategist(args.strategist, args.strategist_task)
        print(f"companion: the strategist ({args.strategist}, task {args.strategist_task}) answers "
              f"{Path(args.game) / 'tww3_bai_nn_deploy.json'}")
    chain = None
    if args.chain:
        chain, what = load_chain(args.chain_op, args.chain_tactic, args.chain_unit, args.chain_greedy,
                                 strategist.plans if strategist is not None else None, args.strategist_task)
        print(f"companion: the chain commands our side: {what}; {args.threads} threads; watching "
              f"{Path(args.game) / exchange.STATE}")
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    try:
        run(args.game, chain, args.log, exit_on_done=args.exit_on_done, idle_exit_s=args.idle_exit,
            strategist=strategist)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

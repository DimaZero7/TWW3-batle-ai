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
                --chain-tactic <a tactic_v2 checkpoint>: the chain 'tactic_v2 only' (bridge/companion/tactic_chain.py).

One line per decision: move number, battle time, how long the chain took, the orders; the levels' ms, the operative's
groups when it decided. A new battle (a new batch in the state) starts the chain afresh.
"""
import argparse
import json
import os
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


READY = "tww3_bai_nn_ready.json"   # the companion's readiness mark in the game folder (watch.ps1 waits for it)
BUSY_S = 5.0     # after a state, look for the next one every poll_s; later (loading, menus) every idle_poll_s


class Seat:
    """The side the companion answers: its state and orders files, who decides (chain.Chain), its log."""

    def __init__(self, state, orders, owner, log=None, before_game=False):
        self.state, self.orders, self.owner, self.log = Path(state), Path(orders), owner, log
        first = exchange.read_state(self.state)
        key = (first["batch"], first["move"]) if first else None
        # a state already there: a battle that ended (done) or is dead (not written within LIVE_S) - never answered, its
        # 'done' is not this run's end; a live battle whose move is answered (a restart) - not answered again, its
        # 'done' ends this run; a live unanswered move (a companion started after the game wrote it) - answered
        self.dead = first["batch"] if key is not None and before_game else None   # never answered at all
        if key is None:
            self.last = self.stale = None
        elif before_game or first.get("done") or not live(self.state):
            self.last = self.stale = key
        else:
            self.last = key if answered(self.orders, key) else None
            self.stale = None
        self.done = False
        self.pending = None                     # (key, text): orders computed but not yet published


LIVE_S = 30.0     # a state file older than this when the companion starts is a dead battle's (the game writes it a second)


def live(path):
    """Was the state file written within LIVE_S of now (file time against the clock: a dead battle left it)?"""
    try:
        return time.time() - os.stat(path).st_mtime <= LIVE_S
    except OSError:
        return False


def log_row(log, row, out):
    """One JSON row appended to the log; a failing log never stops the companion (said on the console)."""
    if not log:
        return
    try:
        with open(log, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row) + "\n")
    except OSError as e:
        out(f"log row not written ({e})")


def publish(seat, key, text, out):
    """The orders text for (batch, move) key published (exchange.write_atomic); on failure kept in seat.pending and tried
    again by the loop with the same text - the network is not asked twice. Returns the write's attempts or None."""
    try:
        attempts = exchange.write_atomic(seat.orders, text)
    except OSError as e:
        if seat.pending is None or seat.pending[0] != key:
            out(f"move {key[1]}: orders not published ({e}); the same text is tried again")
            log_row(seat.log, {"event": "write_failed", "batch": key[0], "move": key[1], "error": str(e)}, out)
        seat.pending = (key, text)
        return None
    if seat.pending is not None and seat.pending[0] == key:
        out(f"move {key[1]}: orders published on a retry")
        log_row(seat.log, {"event": "write_retried", "batch": key[0], "move": key[1]}, out)
    seat.pending = None
    return attempts


def answered(orders, key):
    """Is there an orders file answering (batch, move) key?"""
    try:
        doc = exchange.parse_orders(Path(orders).read_text(encoding="utf-8"))
    except (OSError, ValueError, IndexError):
        return False
    return bool(doc) and (doc.get("batch"), doc.get("move")) == key


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
        for i in range(PLAN_TRIES):          # the same plan text until it lands; then fail loudly (no silent no_plan)
            try:
                attempts = exchange.write_atomic(self.plan, text)
                break
            except OSError:
                if i == PLAN_TRIES - 1:
                    raise
                time.sleep(0.2)
        out(self.deploy.describe(record))
        log_row(self.log, dict(record, event="deploy", write_attempts=attempts), out)
        return True


PLAN_TRIES = 10    # x write_atomic's 50 tries: ~4 s for the deployment's plan before the companion stops with the error


def answer(seat, doc, out=print):
    """One state document answered: the chain's orders written (exchange.orders_text, 'mode chain'), the log lines.
    Returns (orders, think ms, write attempts)."""
    seen = time.perf_counter()
    orders, think_ms, uses = seat.owner.decide(doc)
    attempts = publish(seat, (doc["batch"], doc["move"]), exchange.orders_text(
        doc["batch"], doc["move"], orders, think_ms, uses, mode=getattr(seat.owner, "MODE", None)), out)
    turn_ms = (time.perf_counter() - seen) * 1000
    out(f"move {doc['move']:4d}  t {doc.get('t', 0) / 1000:6.1f} s  think {think_ms:5.1f} ms  "
        f"turn {turn_ms:5.1f} ms  | {exchange.summary(orders)} | {describe(orders)}")
    chain = seat.owner.log_extra() if hasattr(seat.owner, "log_extra") else None
    if chain is not None and hasattr(seat.owner, "describe_extra"):
        out(seat.owner.describe_extra(chain))          # (the chain 'tactic_v2 only': tactic_chain.py)
    elif chain is not None:
        ms = chain["ms"]
        out(f"  chain sec {chain['sec']}: raw {ms['raw_ms']:.0f} op {ms['op_ms']:.0f} "
            f"tac {ms['tac_ms']:.0f} unit {ms['unit_ms']:.0f} ms, inside {sum(chain['inside'].values())}"
            f"/{len(chain['inside'])}" + (f", leash {chain['leash']}" if chain.get("leash") else ""))
        if chain.get("operative") is not None:
            from bridge.companion import chain as companion_chain
            out(f"  operative: {companion_chain.describe(chain['operative'])}")
    if seat.log:
        row = {"batch": doc["batch"], "move": doc["move"], "t": doc.get("t"),
               "think_ms": round(think_ms, 2), "turn_ms": round(turn_ms, 2),
               "write_attempts": attempts, "orders": orders, "abilities": uses}
        if chain is not None:
            row["chain"] = chain                     # the levels' ms, groups, areas, inside
        log_row(seat.log, row, out)
    return orders, think_ms, attempts


def run(game, chain=None, log=None, poll_s=0.005, idle_poll_s=0.1, exit_on_done=False, idle_exit_s=0, out=print,
        strategist=None, before_game=False):
    """The loop. Returns when the battle is done (exit_on_done) or nothing new came for idle_exit_s.
    chain: who answers our side's state (chain.Chain), None: nobody (only the deployment); strategist: who answers
    the deployment document (deploy.Strategist), None: nobody. No clock of its own: a paused battle writes no state,
    which is only a long wait (idle_exit_s 0: forever)."""
    seat = Seat(Path(game) / exchange.STATE, Path(game) / exchange.ORDERS, chain, log,
                before_game) if chain is not None else None
    deploy = DeploySeat(game, strategist, log) if strategist is not None else None
    assert seat or deploy, "nobody to answer"
    last_new, busy_until = time.monotonic(), 0.0
    while True:
        new = False
        if seat is not None and seat.pending is not None:
            publish(seat, seat.pending[0], seat.pending[1], out)
        if deploy is not None and deploy.poll(out):
            new, last_new = True, time.monotonic()
            busy_until = last_new + BUSY_S
        doc = exchange.read_state(seat.state) if seat is not None else None
        if doc is not None and doc["batch"] == seat.dead:
            doc = None                       # (a dead battle's state: not this run's)
        key = (doc["batch"], doc["move"]) if doc is not None else None
        if doc is not None and doc.get("done"):
            # the battle's last state has the same (batch, move) as its last decision (bridge adapter finish): the end
            # is read apart from the decisions' dedup
            if not seat.done and key != seat.stale:
                new, seat.done, seat.last, last_new = True, True, key, time.monotonic()
                out(f"battle {doc['batch']} done after move {doc['move']}")
        elif doc is not None and key != seat.last:
            new = True
            seat.last, last_new = key, time.monotonic()
            busy_until = last_new + BUSY_S
            seat.done = False
            seat.pending = None              # (a newer move: an older unpublished answer is dropped)
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
    if tactic != "formula":
        path = scfg()["tactic_ckpt"] if tactic == "net" else tactic
        tac_net, _, ck = TC.restore(str(_path(path)), device)
        if ck.get("kind") in (TC.KIND2, TC.KIND3):
            # the chain 'tactic_v2 only' (bridge/companion/tactic_chain.py): the trained tactic level, the unit's
            # script, the operative and the strategist's groups not used (one group for the whole army)
            from bridge.companion.tactic_chain import TacticChain
            if op != "formula" or unit is not None:
                raise ValueError("a tactic_v2 / tactic_v3 checkpoint runs without an operative network and a unit network "
                                 "(--chain-op formula, no --chain-unit)")
            tac_net.eval()
            return TacticChain(tac_net, greedy=True, device=device), (
                f"{ck.get('kind')} only: tactic {path} (NetPlayer greedy, one group), unit script, no operative")
    unit_net = unit_network(unit, device)
    op_net, op_what = None, "formula"
    if op != "formula":
        path = scfg()["operative_ckpt"] if op == "net" else op
        op_net, _, _ = OC.restore(str(_path(path)), device)
        op_net.eval()
        op_what = path
    tac_what = "formula"
    if tactic != "formula":
        tac_net.eval()               # (restored above: a tactic_v1 TacticNet)
        tac_what = path
    else:
        tac_net = None
    chain = Chain(unit_net, op_net, tac_net, op_greedy=greedy, tactic_greedy=greedy, unit_greedy=greedy, plans=plans,
                  task=task, device=device)
    what = (f"operative {op_what}, tacticians {tac_what}, unit network {unit or tcfg()['train']['unit_ckpt']}, "
            f"{'greedy' if greedy else 'sampling'}")
    return chain, what


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game", required=True, help="the game folder (where the state file appears)")
    parser.add_argument("--threads", type=int, default=2,
                        help="CPU threads for torch (2: in a 4-CPU container on a loaded host 4 threads made the tactic "
                             "tick 400-750 ms instead of 70-190 with the same orders; the log's first row records it)")
    parser.add_argument("--log", help="append every decision as a JSON line to this file")
    parser.add_argument("--exit-on-done", action="store_true", help="stop when the battle's last state comes")
    parser.add_argument("--idle-exit", type=float, default=0, help="stop after this many seconds without a state")
    parser.add_argument("--strategist", nargs="?", const="default",
                        help="the deployment by the strategist (bridge.build nn-arena --deploy strategist): no value or "
                             "'default' - net/config/strategist.json default, 'formula' or a strategist checkpoint, "
                             "greedy; answers the deployment document once a battle (bridge/companion/deploy.py); "
                             "a .json - the place switch: my infantry on the file's line (deploy.LinePlace)")
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
                             "(net/config/strategist.json tactic_ckpt) or a tactic checkpoint; a tactic_v2 one (weights/"
                             "tactic/attack1/final.pt) - the chain 'tactic_v2 only' (bridge/companion/tactic_chain.py: "
                             "one group, the unit script, no operative; --chain-op formula)")
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
        from bridge.companion.deploy import make
        strategist = make(args.strategist, args.strategist_task)     # (a .json: the place switch, deploy.LinePlace)
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
        with open(args.log, "a", encoding="utf-8", newline="\n") as f:     # (readers take rows with 'move' / 'event')
            f.write(json.dumps({"event": "companion_start", "threads": args.threads, "torch": torch.__version__,
                                "chain": what if chain is not None else None, "strategist": args.strategist}) + "\n")
    # the readiness mark (watch.ps1 starts the game only once it is there with its run's token): the networks are loaded
    exchange.write_atomic(Path(args.game) / READY, json.dumps({"token": os.environ.get("BAI_READY_TOKEN", ""),
                                                             "threads": args.threads, "pid": os.getpid()}))
    try:   # (started by watch.ps1 before the game - the token: whatever state is there belongs to a dead battle)
        run(args.game, chain, args.log, exit_on_done=args.exit_on_done, idle_exit_s=args.idle_exit,
            strategist=strategist, before_game=bool(os.environ.get("BAI_READY_TOKEN")))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

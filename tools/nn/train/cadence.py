"""How often the networks decide in the simulator and how late their orders land, as in the game
(docs/en/training/training.md "Decisions").

In the game the companion decides once a second of battle time (the bridge's tick_ms 1000) and its
orders reach the units ~0.36 s of battle time later (the gate recordings' nn_orders wait_model_ms, 2 gates
03.10.2026, 9251 decisions: median 0.3 s, mean 0.36 s, p10-p90 0.2-0.6 s; gates play at speed 20, the
real wait is ~40 ms). The simulator's physics step stays params.dt (0.5 s): one decision spans
k = decide_s / dt steps. Its orders land after `delay` of them (until then the orders in force go on,
KEEP) and stay in force (KEEP) for the rest; the latency is rounded to whole steps at random so that
its mean is latency_s (0.36 s at dt 0.5: one step late with probability 0.72, at once otherwise).
Cadence(0.5, 0) is the old simulator cadence (a decision every step, orders at once).

The discount and GAE's lambda are given per REF_S (0.5 s, the old decision step); per decision they are
x ** (decision_s / REF_S), so the horizon in seconds stays when the cadence changes (gamma 0.9997 per
0.5 s -> 0.99940 per 1 s decision, lambda 0.95 -> 0.9025)."""
import math
from dataclasses import dataclass

REF_S = 0.5           # s: the step --gamma and lambda are given for
GAME_DECIDE_S = 1.0   # s: the companion's decision interval (tick_ms 1000)
GAME_LATENCY_S = 0.36  # s of battle time from the state to the orders in the game (mean wait_model_ms)
EPS = 1e-6


@dataclass(frozen=True)
class Cadence:
    decide_s: float = GAME_DECIDE_S
    latency_s: float = GAME_LATENCY_S

    def steps(self, dt):
        """Simulator steps a decision spans (decide_s a whole number of steps of dt, at least one)."""
        k = round(self.decide_s / dt)
        if k < 1 or abs(k * dt - self.decide_s) > EPS:
            raise ValueError(f"decide_s {self.decide_s} is not a whole number of simulator steps ({dt} s)")
        return k

    def delay(self, dt):
        """(whole steps, the chance of one more): the latency in steps of dt, its mean latency_s."""
        if self.latency_s < 0:
            raise ValueError(f"order latency {self.latency_s} < 0")
        x = self.latency_s / dt
        whole = math.floor(x + EPS)
        frac = x - whole if x - whole > EPS else 0.0
        if whole + (frac > 0) > self.steps(dt) - 1:
            raise ValueError(f"order latency {self.latency_s} s does not fit a decision of {self.decide_s} s "
                             f"(steps of {dt} s: the orders must land before the next decision)")
        return whole, frac

    def delays(self, dt, B, device):
        """[B] long: the step of this decision at which each battle's new orders land. Random only when
        the latency is not a whole number of steps (torch's default generator: CUDA-graph safe)."""
        import torch                       # (the rest of the module works without torch: the .venv tests)
        whole, frac = self.delay(dt)
        if frac == 0:
            return torch.full((B,), whole, dtype=torch.long, device=device)
        return whole + (torch.rand(B, device=device) < frac).long()

    def discount(self, x, dt=None):
        """A per-REF_S factor (gamma, lambda) -> per decision: x ** (decision seconds / REF_S)."""
        seconds = self.steps(dt) * dt if dt else self.decide_s
        return float(x ** (seconds / REF_S))

    def text(self, dt):
        whole, frac = self.delay(dt)
        late = f"{whole} step(s) late" + (f", one more with p {frac:.2f}" if frac else "")
        return f"decide every {self.decide_s:g} s ({self.steps(dt)} steps of {dt:g} s), orders {self.latency_s:g} s late ({late})"

    def meta(self):
        return {"decide_s": self.decide_s, "latency_s": self.latency_s}


GAME = Cadence()
STEP = Cadence(REF_S, 0.0)     # the old cadence: a decision every 0.5 s step, orders at once


def add_args(ap):
    """--decide-s and --order-latency (run.py, evaluate.py)."""
    ap.add_argument("--decide-s", type=float, default=GAME_DECIDE_S,
                    help="s of battle between the networks' decisions (the game: 1; the old simulator cadence: 0.5)")
    ap.add_argument("--order-latency", type=float, default=GAME_LATENCY_S,
                    help="s of battle from a decision's state to its orders (the game: 0.36 mean; 0: at once)")


def of_args(args):
    return Cadence(args.decide_s, args.order_latency)

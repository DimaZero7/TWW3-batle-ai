"""The next bank of armies, built in a process of its own (run.py --bank-ahead; docs/en/training/training.md).

Every --bank-refresh minutes training takes a new bank of random battles (scenes.Generated, or drills/source.Mixed with
the drills; kitesp.Source around it with --kite-share). Building one is mostly Python (2048 armies ~18 s). Built in a
thread of the training process it shared the GIL with the collection loop: a collection of 7.5 s took 16-35 s while it
ran. Here a worker process builds it on the CPU, and training moves it to its device when it takes it. The bank is the
one built on the device directly:
* the run's random numbers are drawn in the training process (Spec), in the order run.py always drew them;
* the bank's tensors are made by exact operations (copies, indexing, IEEE division), the same numbers on any device;
* a source's random generator (torch.Generator, unused until the source is taken) is made again on the device from
  its seed.

    Spec(...)                        the next bank's random numbers and settings
    build(spec, device) -> source    the bank, as run.py builds it
    Ahead()                          the worker: submit(spec) -> pending; take(pending, device) -> source; close()
"""
import dataclasses
import json
import os
import pickle
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch


@dataclasses.dataclass
class Spec:
    seeds: np.ndarray            # the generated battles' train seeds (generate.TRAIN_SEEDS)
    seed: int                    # the source's own random stream (its pick generator; Mixed: its drills' draws too)
    max_units: int
    params: object               # the simulator's numbers (rollout.params_with_limit)
    layout: object = None        # league.Layout: the drills' rows and the kite battles' rows
    small: tuple = None          # (share, units): run.py --small
    drills: bool = False         # drill battles in the bank (drills/source.Mixed)
    per_drill: int = 256
    kite_share: float = 0.0      # --kite-share: kite battles beside the bank (kitesp.Source)
    kite_seeds: np.ndarray = None
    kite_seed: int = 0


def build(spec, device):
    """The bank's source (module doc), as run.py's source() built it."""
    from tools.nn.train import kitesp, scenes
    from tools.nn.train.drills import source as drill_source
    if spec.drills:
        inner = drill_source.Mixed(spec.seeds, spec.max_units, spec.params, device, spec.layout, seed=spec.seed,
                                   small=spec.small, per_drill=spec.per_drill)
    else:
        inner = scenes.Generated(spec.seeds, spec.max_units, spec.params, device, seed=spec.seed, small=spec.small)
    if not spec.kite_share:
        return inner
    kite = kitesp.armies(spec.kite_seeds)
    return kitesp.Source(inner, kite, spec.kite_share, kitesp.rows_of(spec.layout), spec.params, device,
                         seed=spec.kite_seed)


class _Seed:
    """A generator not used yet, by its seed (a torch.Generator does not cross processes)."""

    def __init__(self, seed):
        self.seed = int(seed)


def _walk(x, leaf, memo):
    """x with leaf(y) put for every y that leaf takes (not None); dicts, lists and the project's objects in place."""
    if id(x) in memo:
        return memo[id(x)]
    y = leaf(x)
    if y is not None:
        memo[id(x)] = y
        return y
    memo[id(x)] = x
    if isinstance(x, dict):
        for k in list(x):
            x[k] = _walk(x[k], leaf, memo)
    elif isinstance(x, list):
        for i in range(len(x)):
            x[i] = _walk(x[i], leaf, memo)
    elif isinstance(x, tuple):
        y = tuple(_walk(v, leaf, memo) for v in x)
        memo[id(x)] = y
        return y
    elif type(x).__module__.startswith("tools.") and hasattr(x, "__dict__"):
        for k, v in list(vars(x).items()):
            setattr(x, k, _walk(v, leaf, memo))
    return x


def _pack_leaf(x):
    if isinstance(x, torch.Generator):
        seed = x.initial_seed()
        if not torch.equal(x.get_state(), torch.Generator(device=x.device).manual_seed(seed).get_state()):
            raise RuntimeError("bankproc: a source's generator was used before it was taken")
        return _Seed(seed)
    return None


def to_device(source, device):
    """A source built on the CPU (its generators as _Seed or still fresh) on the device (module doc)."""
    def leaf(x):
        if torch.is_tensor(x):
            return x.to(device)
        if isinstance(x, _Seed):
            return torch.Generator(device=device).manual_seed(x.seed)
        if isinstance(x, torch.Generator):
            return torch.Generator(device=device).manual_seed(_pack_leaf(x).seed)
        return None
    return _walk(source, leaf, {})



def _job(spec, path):
    """The worker's build: the bank on the CPU, its generators as _Seed, saved to path."""
    torch.save(_walk(build(spec, "cpu"), _pack_leaf, {}), path)


def worker():
    """The worker process's loop: (spec, path) pickled on stdin -> the bank in path, "ok" or the error on its stdout.
    Its stdout is the channel only: whatever the build prints goes to stderr."""
    channel = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    torch.set_num_threads(1)                    # (its few tensor operations: the training keeps the cores)
    try:
        os.nice(5)
    except (AttributeError, OSError):
        pass
    while True:
        try:
            spec, path = pickle.load(sys.stdin.buffer)
        except EOFError:
            return
        try:
            _job(spec, path)
            channel.write("ok\n")
        except Exception:                       # noqa: BLE001 (sent to the training process, raised there)
            channel.write(json.dumps(traceback.format_exc()) + "\n")
        channel.flush()


class Ahead:
    """The worker process: a python of its own (python -m tools.nn.train.bankproc), never on the GPU; the banks it
    builds in order, one file each in a folder of its own."""

    def __init__(self):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        self.dir = Path(tempfile.mkdtemp(prefix="bank_"))
        self.n = 0
        self.proc = subprocess.Popen([sys.executable, "-m", "tools.nn.train.bankproc"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, env=env, cwd=Path(__file__).resolve().parents[3])

    def submit(self, spec):
        """-> the pending bank (its file)."""
        self.n += 1
        path = self.dir / f"bank_{self.n}.pt"
        pickle.dump((spec, str(path)), self.proc.stdin)
        self.proc.stdin.flush()
        return path

    def take(self, pending, device):
        """The bank (waits if it is not ready yet) on the device. The banks are taken in the order submitted."""
        line = self.proc.stdout.readline().decode("utf-8").strip()
        if line != "ok":
            raise RuntimeError("bankproc: the worker " + (f"failed:\n{json.loads(line)}" if line else "ended"))
        src = torch.load(pending, weights_only=False)
        pending.unlink()
        return to_device(src, device)

    def close(self):
        """The worker ends (a bank being built is dropped)."""
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        for f in self.dir.glob("bank_*.pt"):
            f.unlink()
        self.dir.rmdir()


if __name__ == "__main__":
    from tools.nn.train import bankproc      # (its classes by the module's name, not __main__, in the files)
    bankproc.worker()

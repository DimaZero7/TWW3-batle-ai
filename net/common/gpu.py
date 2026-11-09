"""The GPU lock (the old train/test5.gpu_lock, unchanged): one heavy GPU job at a time. A training on CUDA waits while
build/gpu-train.lock exists (polled every 30 s), then holds it (its label and start time) until it ends, also on
failure and on SIGTERM (docker stop); a job started within 60 s of a release (build/gpu-train.released) waits those
60 s out first, so the jobs already waiting go before it. The tactician's train.lock(device, label) takes it on CUDA
only (the operative's and the strategist's training too).
"""
import contextlib
import os
import signal
import time
from pathlib import Path

from net import ROOT

LOCK = ROOT / "build" / "gpu-train.lock"


_HELD = {}                  # the GPU locks this process holds: path -> (its .released stamp, label)


def _free(path, released, label):
    with contextlib.suppress(OSError):
        released.write_text(f"{time.time():.3f} {label}\n", encoding="utf-8", newline="\n")
    with contextlib.suppress(FileNotFoundError):
        path.unlink()


def _stop(signum, frame):
    """SIGTERM (docker stop) -> the held lock freed at once, then SystemExit: the stack unwinds (as on Ctrl+C).
    The container's first process ignores SIGTERM without a handler: docker stop then killed it after 10 s, and
    the lock stayed until tools.ops.leftovers --unlock. Freed here, not only in gpu_lock's finally: the
    unwinding (inner finally blocks, joins) may outlast docker's 10 s, and a SystemExit raised inside a
    finaliser is swallowed (the lock of test5 s16_kite stayed after a docker stop; a docker stop of a test5
    training 2 min in, build/archers/locktest, freed it in 2 s)."""
    for path, (released, label) in list(_HELD.items()):
        if _HELD.pop(path, None) is not None:
            _free(path, released, label)
    raise SystemExit(128 + signum)


@contextlib.contextmanager
def gpu_lock(label, path=LOCK, poll_s=30.0, gap_s=60.0):
    """Wait until no other heavy GPU job holds `path`, hold it while the block runs, then free it (also on
    SIGTERM: _stop)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    released = path.with_suffix(".released")
    # Fair turns: a job that starts within `gap_s` of a release waits that long first, so the jobs
    # already waiting (polling every poll_s) take the lock before it.
    with contextlib.suppress(FileNotFoundError, ValueError):
        since = time.time() - float(released.read_text(encoding="utf-8").split()[0])
        if -gap_s < since < gap_s:              # (a stamp a little ahead: rounding, another clock)
            since = max(since, 0.0)
            print(f"the GPU lock was freed {since:.0f} s ago: waiting {gap_s - since:.0f} s for the queue", flush=True)
            time.sleep(gap_s - since)
    said = False
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if not said:
                held = path.read_text(encoding="utf-8", errors="replace").strip() if path.exists() else ""
                print(f"waiting for the GPU lock {path} ({held})", flush=True)
                said = True
            time.sleep(poll_s)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{label} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    _HELD[path] = (released, label)
    old = None
    with contextlib.suppress(ValueError):        # (signal handlers only in the main thread)
        old = signal.signal(signal.SIGTERM, _stop)
    try:
        yield
    finally:
        if old is not None:
            signal.signal(signal.SIGTERM, old)
        if _HELD.pop(path, None) is not None:     # (not when _stop freed it: it may be another job's by now)
            _free(path, released, label)

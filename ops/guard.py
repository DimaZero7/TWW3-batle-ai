"""Is the game free for us? The marks of state/ (not in git), the same rule as ops/guard.ps1 and ops/guard.sh:

    state/HUMAN_PLAYING     the user plays (game mode): no game of ours starts - only the user's own battle (target
                            human: the live labelling he starts himself)
    state/probe_game.lock   one runner holds the game (a probe series, the chain's battles, the live labelling); its
                            text is the holder's token. A process whose BAI_LOCK_HELD equals that text is the holder
                            (its own launches pass: bridge/launcher/launch.ps1 checks the same); everyone else is refused.

    with hold("probe hill"):     # takes the lock (or rides on the caller's, BAI_LOCK_HELD) and frees it at the end
        ...
    python -m ops.guard          # prints why the game is busy (exit 1) or 'free' (exit 0)
"""
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "state"
HUMAN = STATE / "HUMAN_PLAYING"
LOCK = STATE / "probe_game.lock"
ENV = "BAI_LOCK_HELD"           # the launcher's name (bridge/launcher/launch.ps1)


class Busy(RuntimeError):
    pass


def owner(lock=None):
    """The lock's token (its text, trimmed) or None."""
    try:
        return Path(lock or LOCK).read_text(encoding="utf-8-sig").strip()
    except OSError:
        return None


def busy(target=None, state=None):
    """Why our game may not start now ('' when it may): the user plays, another runner holds the lock."""
    state = Path(state) if state else STATE
    if (state / HUMAN.name).exists() and target != "human":
        return "the user plays (state/HUMAN_PLAYING)"
    tok = owner(state / LOCK.name)
    if tok is not None and tok != os.environ.get(ENV, "").strip():
        return f"the game is held (state/probe_game.lock: {tok!r})"
    return ""


def human(state=None):
    return ((Path(state) if state else STATE) / HUMAN.name).exists()


@contextmanager
def hold(label, target=None, state=None):
    """Takes the lock for a runner (reentrant: a caller's own lock - BAI_LOCK_HELD - is ridden on, not taken again)."""
    state = Path(state) if state else STATE
    lock = state / LOCK.name
    why = busy(target, state)
    if why:
        raise Busy(why)
    mine = owner(lock)
    if mine is not None:                       # the caller's (busy() passed): ride on it
        yield mine
        return
    state.mkdir(parents=True, exist_ok=True)
    tok = f"{label} {time.strftime('%Y-%m-%dT%H:%M:%S')} pid {os.getpid()}"
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        raise Busy(f"the game is held (state/probe_game.lock: {owner(lock)!r})") from None
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(tok + "\n")
    old = os.environ.get(ENV)
    os.environ[ENV] = tok
    try:
        yield tok
    finally:
        if owner(lock) == tok:
            lock.unlink()
        if old is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = old


if __name__ == "__main__":
    why = busy(sys.argv[1] if len(sys.argv) > 1 else None)
    print(why or "free")
    sys.exit(1 if why else 0)

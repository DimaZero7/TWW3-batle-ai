"""Is the game free for us? The marks of state/ (not in git), the same rule as ops/guard.ps1 and ops/guard.sh:

    state/HUMAN_PLAYING     the user plays (game mode): no game of ours starts - only the user's own battle (target
                            human: the live labelling he starts himself)
    state/probe_game.lock   one runner holds the game (a probe series, the chain's battles, the live labelling); its
                            text is the holder's token. A process whose BAI_LOCK_HELD equals that text is the holder
                            (its own launches pass: bridge/launcher/launch.ps1 checks the same); everyone else is refused.
                            The holder of the whole game also holds every launcher slot (it writes its token into
                            probe_game.1.lock .. probe_game.3.lock first, then into probe_game.lock).
    state/probe_game.N.lock one runner holds launcher slot N = 1..3 (bridge/docs/launch/slots.md; BAI_SLOT_LOCK_HELD_N).

    The protocol (the same in the three files): a lock file is taken ONLY by an exclusive create; the whole game takes
    the files 1, 2, 3, whole in this order and gives back what it took when one is held. A token is ASCII:
    '<label> <time> pid <Windows pid> id <8 hex>'. A lock is DEAD when its token has that form and no process with that
    pid runs, or when it is empty and older than 60 s; a dead lock is removed only by the one who created
    '<lock>.break' (exclusive) and re-read the same dead token under it. Anything else (unreadable, an old token) is busy.

    with hold("probe hill"):     # takes the lock (or rides on the caller's, BAI_LOCK_HELD) and frees it at the end
        ...
    python -m ops.guard          # prints why the game is busy (exit 1) or 'free' (exit 0)
"""
import os
import re
import secrets
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "state"
HUMAN = STATE / "HUMAN_PLAYING"
LOCK = STATE / "probe_game.lock"
ENV = "BAI_LOCK_HELD"           # the launcher's name (bridge/launcher/launch.ps1)
SLOT_ENV = "BAI_SLOT_LOCK_HELD_"  # + N
SLOTS = (1, 2, 3)
EMPTY_DEAD_S = 60
TOKEN_RE = re.compile(r" pid (\d+) id [0-9a-f]{8}$")
UNREADABLE = "<unreadable>"


def slot_lock(n, state=None):
    return (Path(state) if state else STATE) / f"probe_game.{n}.lock"


class Busy(RuntimeError):
    pass


def owner(lock=None):
    """The lock's token (its text, trimmed), None when there is no lock, '<unreadable>' when it cannot be read."""
    try:
        return Path(lock or LOCK).read_bytes().decode("utf-8-sig", "replace").strip()
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError:
        return UNREADABLE


def _pid_alive(pid):
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    import ctypes
    k = ctypes.windll.kernel32
    k.OpenProcess.restype = ctypes.c_void_p
    h = k.OpenProcess(0x1000, False, pid)            # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return ctypes.GetLastError() != 87           # 87: no such process; access denied: it runs
    try:
        code = ctypes.c_ulong()
        ok = k.GetExitCodeProcess(ctypes.c_void_p(h), ctypes.byref(code))
        return (not ok) or code.value == 259         # STILL_ACTIVE
    finally:
        k.CloseHandle(ctypes.c_void_p(h))


def dead(lock, tok=None):
    """The one rule for a dead holder (see the module's text)."""
    lock = Path(lock)
    tok = owner(lock) if tok is None else tok
    if tok is None or tok == UNREADABLE:
        return False
    if tok == "":
        try:
            return time.time() - lock.stat().st_mtime > EMPTY_DEAD_S
        except OSError:
            return False
    m = TOKEN_RE.search(tok)
    return bool(m) and not _pid_alive(int(m.group(1)))


def _unlink(path):
    for _ in range(40):
        try:
            Path(path).unlink()
            return True
        except FileNotFoundError:
            return True
        except PermissionError:
            time.sleep(0.05)
    return False


def _break_dead(lock):
    """Removes a dead lock under '<lock>.break'; True when the lock is gone (the caller may create it again)."""
    tok = owner(lock)
    if tok is None:
        return True
    if not dead(lock, tok):
        return False
    brk = Path(str(lock) + ".break")
    try:
        fd = os.open(brk, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except OSError:
        return False                                  # someone else breaks it (a stale .break stays busy: by hand)
    os.close(fd)
    try:
        if owner(lock) == tok:
            _unlink(lock)
    finally:
        _unlink(brk)
    return owner(lock) is None


def _create(lock, tok):
    """Exclusive create of the lock with the token; one retry after a dead lock was broken."""
    for attempt in (0, 1):
        try:
            fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0))
        except OSError:
            if attempt == 0 and _break_dead(lock):
                continue
            return False
        try:
            os.write(fd, (tok + "\n").encode("ascii"))
        except OSError:
            os.close(fd)
            _unlink(lock)
            return False
        os.close(fd)
        return True
    return False


def _drop(lock, tok):
    if owner(lock) == tok:
        _unlink(lock)


def token(label):
    label = re.sub(r"[^ -~]", "?", str(label)).strip() or "lock"
    return f"{label} {time.strftime('%Y-%m-%dT%H:%M:%S')} pid {os.getpid()} id {secrets.token_hex(4)}"


def busy(target=None, state=None):
    """Why our game may not start now ('' when it may): the user plays, another runner holds the lock."""
    state = Path(state) if state else STATE
    if (state / HUMAN.name).exists() and target != "human":
        return "the user plays (state/HUMAN_PLAYING)"
    tok = owner(state / LOCK.name)
    if tok is not None and dead(state / LOCK.name, tok):                # a dead holder's lock is no one's
        tok = None
    if tok is not None and not (tok and tok == os.environ.get(ENV, "").strip()):
        return f"the game is held (state/probe_game.lock: {tok!r})"
    if tok is None:
        for n in SLOTS:
            s = owner(slot_lock(n, state))
            if s is not None and not dead(slot_lock(n, state), s):        # a dead holder's slot lock is no one's
                return f"launcher slots hold the game (state/probe_game.{n}.lock: {s!r})"
    return ""


def busy_slot(n, state=None):
    """Why launcher slot n may not start now ('' when it may): the user plays, someone else's whole-game lock or slot lock."""
    state = Path(state) if state else STATE
    if (state / HUMAN.name).exists():
        return "the user plays (state/HUMAN_PLAYING)"
    tok = owner(state / LOCK.name)
    if tok is not None and dead(state / LOCK.name, tok):                # a dead holder's lock is no one's
        tok = None
    if tok is not None:
        return "" if tok and tok == os.environ.get(ENV, "").strip() else f"the game is held (state/probe_game.lock: {tok!r})"
    s = owner(slot_lock(n, state))
    if s is not None and not (s and s == os.environ.get(SLOT_ENV + str(n), "").strip()) and not dead(slot_lock(n, state), s):
        return f"slot {n} is held (state/probe_game.{n}.lock: {s!r})"
    return ""


def _busy_live(state, target=None, slot=None):
    """busy() / busy_slot() for a taker: a dead holder's lock counts as free (the taker breaks it)."""
    if (state / HUMAN.name).exists() and (slot is not None or target != "human"):
        return "the user plays (state/HUMAN_PLAYING)"
    whole = state / LOCK.name
    tok = owner(whole)
    if tok is not None:
        if tok and tok == os.environ.get(ENV, "").strip():
            return ""
        if not dead(whole, tok):
            return f"the game is held (state/probe_game.lock: {tok!r})"
    for n in (SLOTS if slot is None else (slot,)):
        s = owner(slot_lock(n, state))
        if s is None:
            continue
        if slot is not None and s and s == os.environ.get(SLOT_ENV + str(n), "").strip():
            return ""
        if dead(slot_lock(n, state), s):
            continue
        if slot is not None:
            return f"slot {n} is held (state/probe_game.{n}.lock: {s!r})"
        return f"launcher slots hold the game (state/probe_game.{n}.lock: {s!r})"
    return ""


def human(state=None):
    return ((Path(state) if state else STATE) / HUMAN.name).exists()


@contextmanager
def hold(label, target=None, state=None):
    """Takes the lock for a runner (reentrant: a caller's own lock - BAI_LOCK_HELD - is ridden on, not taken again)."""
    state = Path(state) if state else STATE
    lock = state / LOCK.name
    mine = owner(lock)
    if mine and mine == os.environ.get(ENV, "").strip() and not ((state / HUMAN.name).exists() and target != "human"):
        yield mine                             # the caller's own lock: ride on it
        return
    why = _busy_live(state, target)
    if why:
        raise Busy(why)
    state.mkdir(parents=True, exist_ok=True)
    tok = token(label)
    taken = []
    for f in [slot_lock(n, state) for n in SLOTS] + [lock]:
        if not _create(f, tok):
            for t in reversed(taken):
                _drop(t, tok)
            held = owner(f)
            raise Busy(f"the game is held (state/probe_game.lock: {held!r})" if f == lock else
                       f"launcher slots hold the game (state/{f.name}: {held!r})")
        taken.append(f)
    if (state / HUMAN.name).exists() and target != "human":
        for t in reversed(taken):
            _drop(t, tok)
        raise Busy("the user plays (state/HUMAN_PLAYING)")
    old = os.environ.get(ENV)
    os.environ[ENV] = tok
    try:
        yield tok
    finally:
        for t in [lock] + [slot_lock(n, state) for n in reversed(SLOTS)]:
            _drop(t, tok)
        if old is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = old


@contextmanager
def hold_slot(n, label, state=None):
    """Takes launcher slot n (or rides on the caller's whole-game lock or own slot lock); bridge/launcher/launch.ps1 -Slot n."""
    state = Path(state) if state else STATE
    if n not in SLOTS:
        raise ValueError(f"slot must be one of {SLOTS}")
    lock, whole, env = slot_lock(n, state), state / LOCK.name, SLOT_ENV + str(n)
    why = _busy_live(state, slot=n)
    if why:
        raise Busy(why)
    w, s = owner(whole), owner(lock)
    if (w and w == os.environ.get(ENV, "").strip()) or (s and s == os.environ.get(env, "").strip()):
        yield None                             # the caller's whole-game lock or own slot lock: ride on it
        return
    state.mkdir(parents=True, exist_ok=True)
    tok = token(label)
    if not _create(lock, tok):
        raise Busy(f"slot {n} is held ({lock.name}: {owner(lock)!r})")
    if owner(whole) is not None and not _break_dead(whole):       # an older runner's whole-game lock
        _drop(lock, tok)
        raise Busy(f"the game is held (state/probe_game.lock: {owner(whole)!r})")
    if (state / HUMAN.name).exists():
        _drop(lock, tok)
        raise Busy("the user plays (state/HUMAN_PLAYING)")
    old = os.environ.get(env)
    os.environ[env] = tok
    try:
        yield tok
    finally:
        _drop(lock, tok)
        if old is None:
            os.environ.pop(env, None)
        else:
            os.environ[env] = old


if __name__ == "__main__":
    why = busy(sys.argv[1] if len(sys.argv) > 1 else None)
    print(why or "free")
    sys.exit(1 if why else 0)

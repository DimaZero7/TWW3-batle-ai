"""Leftovers between steps: our containers, the GPU lock, wait loops, the game, the GPU; safe
cleanup of our own (docs/en/training/workflow.md).

    python -m tools.ops.leftovers                 # report; exit 1 when something is left
    python -m tools.ops.leftovers --kill          # end the wait loops older than --min-age minutes (bash with until/while ... sleep, and their sleep children)
    python -m tools.ops.leftovers --unlock        # remove a stale build/gpu-train.lock (no training container runs)
    python -m tools.ops.leftovers --stop NAME     # docker stop one of OUR containers (image snake-ai-trainer or the companion)

What is ours: containers from the image snake-ai-trainer or named tww3-bai-companion (the other
projects' containers are not touched or listed), the lock build/gpu-train.lock (test5 / run.py
write it; stale when no training container runs), bash processes whose command line is a wait loop
(until|while ... sleep) with their sleep children. Also shown: Warhammer3.exe, python processes of
tools.nn (a gate, the companion) and the GPU's load (nvidia-smi) when available. Nothing is ended or
removed without its flag; the shell this runs from and its parents are never ended.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from tools import config as project

LOCK = project.BUILD / "nn-train" / "gpu-train.lock"
OUR_IMAGES = ("snake-ai-trainer",)
OUR_NAMES = ("tww3-bai-companion",)
LOOP = re.compile(r"\b(until|while)\b.*\bsleep\b", re.I | re.S)
PS_LIST = ("Get-CimInstance Win32_Process -Filter \"name='sleep.exe' or name='bash.exe' or name='Warhammer3.exe' or "
           "name='python.exe' or name='powershell.exe'\" | Select-Object ProcessId,ParentProcessId,Name,CreationDate,CommandLine "
           "| ConvertTo-Json -Compress")


def run(cmd, timeout=30):
    """stdout of a command ("" when it fails or is missing); tests replace it."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace").stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def containers():
    """[{name, image, status, since}] of OUR running containers."""
    out = []
    for line in run(["docker", "ps", "--format", "{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.RunningFor}}"]).splitlines():
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        name, image, status, since = parts[:4]
        if image.split(":")[0] in OUR_IMAGES or name in OUR_NAMES:
            out.append({"name": name, "image": image, "status": status, "since": since})
    return out


def lock(path=None, training=False):
    """{text, age_min, stale} of the GPU lock or None; stale when no training container runs."""
    path = LOCK if path is None else path
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace").strip()
        age = (time.time() - Path(path).stat().st_mtime) / 60
    except OSError:
        return None
    return {"text": text, "age_min": age, "stale": not training}


def processes():
    """[{pid, ppid, name, age_min, cmd}] from PowerShell (Windows); [] elsewhere or when it fails."""
    text = run(["powershell", "-NoProfile", "-Command", PS_LIST])
    try:
        items = json.loads(text) if text.strip() else []
    except ValueError:
        return []
    if isinstance(items, dict):
        items = [items]
    now = time.time() * 1000
    out = []
    for it in items:
        m = re.search(r"(\d+)", str(it.get("CreationDate") or ""))
        age = (now - int(m.group(1))) / 60000 if m else None
        out.append({"pid": it.get("ProcessId"), "ppid": it.get("ParentProcessId"), "name": it.get("Name") or "",
                    "age_min": age, "cmd": it.get("CommandLine") or ""})
    return out


def ancestors(procs, pid):
    """The pids of `pid` and its parents (never ended)."""
    by = {p["pid"]: p["ppid"] for p in procs}
    out, cur = set(), pid
    while cur in by and cur not in out:
        out.add(cur)
        cur = by[cur]
    return out


def classify(procs, own=None):
    """{"loops": bash wait loops (not ours), "sleeps": sleep.exe by parent, "game": Warhammer3, "nn": python of tools.nn}."""
    own = ancestors(procs, own if own is not None else os.getpid())
    loops = [p for p in procs if p["name"].lower() == "bash.exe" and LOOP.search(p["cmd"]) and p["pid"] not in own]
    sleeps = [p for p in procs if p["name"].lower() == "sleep.exe"]
    game = [p for p in procs if p["name"].lower() == "warhammer3.exe"]
    nn = [p for p in procs if p["name"].lower() == "python.exe" and re.search(r"tools[./\\]nn\b", p["cmd"]) and p["pid"] not in own]
    return {"loops": loops, "sleeps": sleeps, "game": game, "nn": nn}


def gpu():
    text = run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], 10)
    parts = [x.strip() for x in text.strip().split(",")] if text.strip() else []
    return {"util": int(parts[0]), "used_mb": int(parts[1]), "total_mb": int(parts[2])} if len(parts) == 3 and all(parts) else None


def snippet(cmd, n=90):
    """The loop's own text out of a bash -c command line."""
    m = LOOP.search(cmd)
    s = cmd[max(0, m.start() - 20):m.end() + 40] if m else cmd
    s = re.sub(r"\s+", " ", s)
    return s if len(s) <= n else s[:n] + "..."


def report(procs=None, min_age=10.0):
    """(lines, found) of the current state; found: anything left (our containers, a stale lock, wait loops)."""
    procs = processes() if procs is None else procs
    cons = containers()
    lk = lock(training=any(c["image"].split(":")[0] in OUR_IMAGES for c in cons))
    cl = classify(procs)
    lines, found = [], False
    if cons:
        found = True
        lines += [f"containers (ours): {len(cons)}" + (" - MORE THAN ONE: the rule is one container in total" if len(cons) > 1 else "")]
        lines += [f"  {c['name']}  {c['image']}  {c['status']}" for c in cons]
    else:
        lines.append("containers (ours): none")
    if lk:
        lines.append(f"GPU lock: {lk['text']!r}, {lk['age_min']:.0f} min old" + (" - STALE (no training container runs)" if lk["stale"] else ""))
        found = found or lk["stale"]
    else:
        lines.append("GPU lock: free")
    old = [p for p in cl["loops"] if p["age_min"] is None or p["age_min"] >= min_age]
    if cl["loops"]:
        lines.append(f"wait loops (bash until/while ... sleep): {len(cl['loops'])}" + (f", {len(old)} older than {min_age:g} min" if old else ""))
        lines += [f"  pid {p['pid']}  {p['age_min'] or 0:.0f} min  {snippet(p['cmd'])}" for p in cl["loops"]]
        found = found or bool(old)
    else:
        lines.append("wait loops: none")
    if cl["sleeps"]:
        lines.append(f"sleep processes: {len(cl['sleeps'])} (parents " + ", ".join(str(p["ppid"]) for p in cl["sleeps"]) + ")")
    if cl["game"]:
        lines.append("Warhammer3.exe is running")
    if cl["nn"]:
        lines.append("python of tools.nn: " + "; ".join(f"pid {p['pid']} {snippet(p['cmd'])}" for p in cl["nn"]))
    g = gpu()
    if g:
        lines.append(f"GPU: {g['util']} % busy, {g['used_mb']} / {g['total_mb']} MiB")
    return lines, found


def kill(procs, min_age=10.0):
    """End the wait loops older than min_age and their sleep children; the pids ended."""
    cl = classify(procs)
    pids = [p["pid"] for p in cl["loops"] if p["age_min"] is None or p["age_min"] >= min_age]
    pids += [p["pid"] for p in cl["sleeps"] if p["ppid"] in pids]
    for pid in pids:
        run(["powershell", "-NoProfile", "-Command", f"Stop-Process -Id {int(pid)} -Force -ErrorAction SilentlyContinue"])
    return pids


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kill", action="store_true", help="end the wait loops older than --min-age and their sleeps")
    ap.add_argument("--unlock", action="store_true", help="remove a stale GPU lock")
    ap.add_argument("--stop", metavar="NAME", help="docker stop this container of ours")
    ap.add_argument("--min-age", type=float, default=10.0, help="minutes a wait loop must be old to count / be ended")
    args = ap.parse_args(argv)
    procs = processes()
    lines, found = report(procs, args.min_age)
    print("\n".join(lines))
    if args.kill:
        pids = kill(procs, args.min_age)
        print(f"ended: {pids or 'nothing'}")
    if args.unlock:
        lk = lock(training=any(c["image"].split(":")[0] in OUR_IMAGES for c in containers()))
        if lk and lk["stale"]:
            LOCK.unlink()
            print(f"removed the stale lock {LOCK}")
        else:
            print("lock not removed: " + ("none" if not lk else "a training container runs"))
    if args.stop:
        if any(c["name"] == args.stop for c in containers()):
            print(run(["docker", "stop", "-t", "10", args.stop], 60).strip() or f"stopped {args.stop}")
        else:
            print(f"not stopped: {args.stop} is not one of our running containers")
    return 1 if found and not (args.kill or args.unlock or args.stop) else 0


if __name__ == "__main__":
    sys.exit(main())

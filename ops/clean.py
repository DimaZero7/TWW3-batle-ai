"""What piles up on the disk and its cleaning (CLAUDE.md 'Мусор').

    python -m ops.clean                  # report only
    python -m ops.clean --apply          # also prune the torch cache (nothing else is ever removed here)

  torch cache   the Docker volume tww3-torch-cache (ops/dock.sh: /cache/inductor, /cache/triton). Every edit of sim/
                compiles the step anew and the old kernels stay for ever (13.11.2026: 106 GB, 103 of them not read for
                three days and more). Removed: an entry no file of which was read for --days days - a kernel's folder of
                triton, the files of one name in a folder of inductor (or a nested folder there). What a running
                container uses was read today, so the prune is safe while battles run; a pruned entry costs only a
                compile if it is ever needed again.
  tmp           folders of tmp/ over --tmp-gb GB not changed for --tmp-days days: listed only - tmp/ is removed on the
                user's word, after a look at who refers to it (sim/config `_why`, game/manual.md point into tmp/).
  disk          the free space of the project's drive and the size of Docker's virtual disk: it never shrinks by
                itself (compact: quit Docker, `wsl --shutdown`, Optimize-VHD or diskpart `compact vdisk`).
The wait loops and containers left behind: python -m ops.leftovers.
"""
import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VOLUME = "tww3-torch-cache"
IMAGE = "snake-ai-trainer"

# runs inside the container: /cache/<kind>/..., an entry kept when any of its files was read within DAYS
PRUNE = r"""
import os, shutil, sys, time
days, apply = float(sys.argv[1]), sys.argv[2] == "1"
cut = time.time() - days * 86400
def newest(path):
    t = 0.0
    for d, _, fs in os.walk(path):
        for f in fs:
            try:
                s = os.stat(os.path.join(d, f)); t = max(t, s.st_atime, s.st_mtime)
            except OSError:
                pass
    return t
def size(path):
    n = 0
    for d, _, fs in os.walk(path):
        for f in fs:
            try:
                n += os.path.getsize(os.path.join(d, f))
            except OSError:
                pass
    return n
kept = gone = 0
def drop(paths, t):
    global kept, gone
    b = sum(size(p) if os.path.isdir(p) else os.path.getsize(p) for p in paths)
    if t >= cut:
        kept += b
        return
    gone += b
    if apply:
        for p in paths:
            shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)
tri = "/cache/triton"
for e in (os.listdir(tri) if os.path.isdir(tri) else []):
    p = os.path.join(tri, e)
    drop([p], newest(p) if os.path.isdir(p) else max(os.stat(p).st_atime, os.stat(p).st_mtime))
ind = "/cache/inductor"
for e in (os.listdir(ind) if os.path.isdir(ind) else []):
    p = os.path.join(ind, e)
    if not os.path.isdir(p):
        drop([p], max(os.stat(p).st_atime, os.stat(p).st_mtime)); continue
    groups = {}
    for f in os.listdir(p):
        groups.setdefault(f.split(".")[0], []).append(os.path.join(p, f))
    for paths in groups.values():
        t = 0.0
        for q in paths:
            t = max(t, newest(q) if os.path.isdir(q) else max(os.stat(q).st_atime, os.stat(q).st_mtime))
        drop(paths, t)
print("%.1f %.1f" % (kept / 1e9, gone / 1e9))
"""


def torch_cache(days, apply):
    """(kept GB, removable GB) of the volume, or None without Docker / the volume."""
    cmd = ["docker", "run", "--rm", "--cpus", "2", "--memory", "1g", "-v", f"{VOLUME}:/cache", IMAGE,
           "python", "-c", PRUNE, str(days), "1" if apply else "0"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=3600,
                             env={**os.environ, "MSYS_NO_PATHCONV": "1"})
    except (OSError, subprocess.TimeoutExpired):
        return None
    line = (out.stdout.strip().splitlines() or [""])[-1].split()
    return (float(line[0]), float(line[1])) if out.returncode == 0 and len(line) == 2 else None


def _link(path):
    """A symlink or a Windows junction (a folder that is another folder: tmp/gemini_re/ws/reverse/ghidra)."""
    try:
        return os.path.islink(path) or bool(getattr(os.lstat(path), "st_file_attributes", 0) & 0x400)
    except OSError:
        return True


def tmp_folders(min_gb, min_days):
    """[(GB, days since the last change, path)] of tmp/'s folders two levels deep (links are not entered)."""
    out, now = [], time.time()
    tmp = ROOT / "tmp"
    tops = [p for p in tmp.iterdir() if p.is_dir() and not _link(p)] if tmp.is_dir() else []
    for top in tops:
        subs = [p for p in top.iterdir() if p.is_dir() and not _link(p)] if top.name.startswith("old-") else [top]
        for p in subs:
            n, last = 0, 0.0
            for d, dirs, fs in os.walk(p):
                dirs[:] = [x for x in dirs if not _link(os.path.join(d, x))]
                for f in fs:
                    try:
                        s = os.stat(os.path.join(d, f))
                    except OSError:
                        continue
                    n += s.st_size
                    last = max(last, s.st_mtime)
            if n / 1e9 >= min_gb and (now - last) / 86400 >= min_days:
                out.append((n / 1e9, (now - last) / 86400, str(p.relative_to(ROOT))))
    return sorted(out, reverse=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="prune the torch cache (the only thing removed)")
    ap.add_argument("--days", type=float, default=3.0, help="a cache entry not read for this many days goes")
    ap.add_argument("--tmp-gb", type=float, default=1.0)
    ap.add_argument("--tmp-days", type=float, default=14.0)
    a = ap.parse_args()
    free = shutil.disk_usage(ROOT).free / 1e9
    print(f"free on the project's drive: {free:.1f} GB")
    vhd = Path(os.environ.get("LOCALAPPDATA", "")) / "Docker" / "wsl" / "disk" / "docker_data.vhdx"
    if vhd.is_file():
        print(f"Docker's virtual disk: {vhd.stat().st_size / 1e9:.1f} GB (shrinks only when compacted)")
    tc = torch_cache(a.days, a.apply)
    if tc is None:
        print("torch cache: not measured (Docker or the volume is not there)")
    else:
        print(f"torch cache: {tc[0]:.1f} GB in use, {tc[1]:.1f} GB not read for {a.days:g} days - "
              + ("removed" if a.apply else "python -m ops.clean --apply removes them"))
    rows = tmp_folders(a.tmp_gb, a.tmp_days)
    print(f"tmp/ folders over {a.tmp_gb:g} GB untouched for {a.tmp_days:g} days (removed on the user's word only): "
          + ("none" if not rows else ""))
    for gb, days, path in rows:
        print(f"  {gb:6.1f} GB  {days:5.0f} d  {path}")


if __name__ == "__main__":
    main()

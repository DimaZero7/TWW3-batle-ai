"""The script references of the next training step, ready before it starts (docs/en/training/workflow.md
"Baseline cache and canary").

    .venv/Scripts/python -m tools.ops.baselines                                  # which files the code here needs
    .venv/Scripts/python -m tools.ops.baselines --run                            # play the missing ones (container)
    cd <worktree> && .venv/Scripts/python -m tools.ops.baselines --build <main checkout>/build --run

A simulator change (tools/nn/train/version.py VERSION_FILES) or a drill change (tools/nn/train/drills) makes
the next step's evaluation miss its script references: the three scripts' baselines (256 pairs each) and
the drills' check scripts (128 battles of each verified drill). test5 plays the missing ones itself on
the CPU (tools/nn/train/refs.py, ~10 min with 8 cores); this command plays them AHEAD: started from the
worktree with the change while the previous step still trains, with the main checkout's build folder
(--build), the files are there under the new version when the step starts. The version is a hash of
the files' contents, so the worktree's change gives the version the commit will have.

Host side (no torch): the versions and the files; --run starts the container (one, CPU only:
tools.nn.train.refs; DOCK_CPUS cores, default 8, so it leaves the training's own CPU alone).
"""
import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

from tools import config as project
from tools.nn.train import refs
from tools.nn.train import version as sim_ver
from tools.ops.step import posix

ROOT = project.ROOT
DRILLS_INIT = ROOT / sim_ver.DRILL_FILES / "__init__.py"
CONTAINER = "orch-baselines"
CPUS = 8


def ready_drills(path=DRILLS_INIT):
    """drills.READY read from its source (the drills package imports torch)."""
    for node in ast.parse(Path(path).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "READY" for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    return ()


def status(folder, version=None, drill_version=None, drills=None, pairs=refs.PAIRS, drill_battles=refs.DRILL_BATTLES):
    """(version, drill version, [missing tasks]) of the code here against the cache folder: a script name
    when its baseline file is absent, "drill:<name>" when no file of that drill and drill version is."""
    version = version or sim_ver.sim_version()
    drill_version = drill_version or sim_ver.drill_version()
    folder = Path(folder)
    out = []
    for n in refs.SCRIPTS:
        p = folder / sim_ver.baseline_name(n, refs.MAX_UNITS, refs.LIMIT_S, version)
        try:
            ok = len(json.loads(p.read_text(encoding="utf-8")).get("seed", [])) >= pairs
        except (OSError, ValueError, AttributeError):
            ok = False
        if not ok:
            out.append(n)
    for n in (ready_drills() if drills is None else drills):
        if not list(folder.glob(f"drill_{n}_{drill_battles}_*_{drill_version}.json")):
            out.append(f"drill:{n}")
    return version, drill_version, out


def command(build, name=CONTAINER, cpus=CPUS, root=ROOT):
    """(env, argv) of the container run that plays the missing references into `build`/nn-train/baselines."""
    env = dict(os.environ, DOCK_NAME=name, DOCK_CPUS=str(cpus), DOCK_BUILD=posix(build))
    return env, [bash(), posix(Path(root) / "tools" / "nn" / "dock.sh"), "tools.nn.train.refs"]


def bash():
    """Git Bash on Windows (a bare "bash" there resolves to WSL's, which cannot run dock.sh), else bash."""
    if os.name == "nt":
        git_bash = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git" / "bin" / "bash.exe"
        if git_bash.exists():
            return str(git_bash)
    return "bash"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", default=str(project.BUILD), help="the build folder whose nn-train/baselines to fill "
                                                                "(from a worktree: the main checkout's build)")
    ap.add_argument("--run", action="store_true", help="play the missing ones now (a container; blocks until done)")
    ap.add_argument("--cpus", type=int, default=CPUS, help="the container's cores (DOCK_CPUS)")
    ap.add_argument("--name", default=CONTAINER, help="the container's name")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    folder = Path(args.build) / "nn-train" / "baselines"
    version, dversion, missing = status(folder)
    print(f"simulator version {version}, drill version {dversion}; cache {folder}")
    if not missing:
        print("script references: all present (the next step's evaluation finds them)")
        return 0
    print(f"missing: {', '.join(missing)} (a test5 start plays them itself on the CPU: ~10 min with 8 cores)")
    env, cmd = command(args.build, args.name, args.cpus)
    shown = f"DOCK_NAME={env['DOCK_NAME']} DOCK_CPUS={env['DOCK_CPUS']} DOCK_BUILD={env['DOCK_BUILD']} {' '.join(cmd)}"
    if not args.run:
        print(f"ahead of the step:  {shown}")
        return 0
    print(f"running: {shown}", flush=True)
    rc = subprocess.call(cmd, env=env, cwd=ROOT)
    if rc == 0:
        missing = status(folder)[2]
        print("script references: all present" if not missing else f"still missing: {', '.join(missing)}")
    return rc or (1 if missing else 0)


if __name__ == "__main__":
    sys.exit(main())

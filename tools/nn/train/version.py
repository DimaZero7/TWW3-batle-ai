"""The simulator's version for the script baselines' cache, and the comparison that lets an older
version's baseline be adopted (docs/en/training/training.md "Test protocol").

Plain Python (no torch): tools/ops/step.py asks it on the host whether the next run's "before"
evaluation will find its baselines (a cache miss: the three scripts play their 256 pairs again, with
the drills' check scripts ~8 min on the CPU, tools/nn/train/refs.py), and tools/nn/train/evaluate.py
imports it for the same hash.

    python -m tools.nn.train.version            # the version and which baseline files exist for it
"""
import hashlib
import json
from pathlib import Path

from tools import config as project

ROOT = project.ROOT
BASELINES = project.BUILD / "nn-train" / "baselines"
# What a script-vs-script battle depends on. Changing any of these files changes the version and the
# cache misses, also when the change cannot touch a script battle (the network's code in tools/nn/model):
# the canary (evaluate.baselines, test5 --baseline-canary) then recovers the old baseline when the first
# pairs come out identical.
# Files inside VERSION_FILES that script-vs-script battles never run (the replay check of recordings).
VERSION_SKIP = ("tools/nn/sim/check.py",)
VERSION_FILES = ("tools/nn/sim", "tools/nn/armies", "tools/nn/train/opponents.py", "tools/nn/train/scenes.py",
                 "tools/nn/train/reward.py", "tools/nn/train/randomise.py", "tools/nn/scenario.py", "tools/nn/units.py",
                 "tools/nn/abilities.py", "tools/nn/model", "config/nn")
# What the drills' script references depend on besides the simulator (drill_version()).
DRILL_FILES = "tools/nn/train/drills"
# What an evaluation's numbers depend on besides the simulator and the drills (eval_version()): test5 takes the
# previous step's last evaluation as its "before" only when these are the same too.
EVAL_FILES = ("tools/nn/train/evaluate.py", "tools/nn/train/rollout.py", "tools/nn/train/behaviour.py",
              "tools/nn/train/skill.py", "tools/nn/train/matchups.py", "tools/nn/train/profiles.py",
              "tools/nn/train/cadence.py", "tools/nn/train/league.py")
# The baseline document's battle fields (evaluate.script_battles): lists over the pairs, in seed order.
FIELDS = ("seed", "winner", "lost", "start", "budget", "factions", "attacker")


def sim_version():
    """A hash of what a script-vs-script battle depends on: VERSION_FILES (line ends ignored) and the
    generator's budget factors as loaded (an override in memory changes the armies too)."""
    from tools.nn.armies import generate
    h = hashlib.sha256()
    h.update(json.dumps(sorted((f, p.budget_factor) for f, p in generate.default().pools.items())).encode())
    for rel in VERSION_FILES:
        p = ROOT / rel
        for f in (sorted(p.rglob("*")) if p.is_dir() else [p]):
            if (f.is_file() and f.suffix in (".py", ".json") and "__pycache__" not in f.parts
                    and f.relative_to(ROOT).as_posix() not in VERSION_SKIP):
                h.update(f.relative_to(ROOT).as_posix().encode())
                h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:12]


def drill_version():
    """sim_version() and the drills' code (tools/nn/train/drills/*.py): what a drill's script references
    (evaluate.drill_scripts: BASELINES/drill_<name>_..._<drill_version>.json) depend on."""
    h = hashlib.sha256(sim_version().encode())
    for f in sorted((ROOT / DRILL_FILES).glob("*.py")):
        h.update(f.name.encode())
        h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:12]


def eval_version():
    """drill_version() and the evaluation's own code (EVAL_FILES): what test5's evaluation of a network depends on
    besides the network and the settings (test5.eval_key)."""
    h = hashlib.sha256(drill_version().encode())
    for rel in EVAL_FILES:
        h.update(rel.encode())
        h.update((ROOT / rel).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:12]


def baseline_name(name, max_units, limit_s, version):
    return f"{name}_{max_units}u_{limit_s:g}s_{version}.json"


def older_baselines(folder, name, max_units, limit_s, version, seeds):
    """[(path, doc)] of the opponent `name`'s baselines of OTHER versions in `folder` (same units and
    limit) whose seeds start with `seeds`, the newest file first: the candidates the canary compares
    the current simulator against."""
    folder = Path(folder)
    out = []
    for p in sorted(folder.glob(f"{name}_{max_units}u_{limit_s:g}s_*.json"), key=lambda p: -p.stat().st_mtime):
        if p.name == baseline_name(name, max_units, limit_s, version):
            continue
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict) and doc.get("seed", [])[:len(seeds)] == list(seeds):
            out.append((p, doc))
    return out


def same_prefix(doc, canary, n):
    """True when the first n pairs of `doc` (an older baseline) equal the canary's battles in every
    field: the same winner, gold lost and start on both sides, budget, factions and attacker, exactly."""
    for k in FIELDS:
        if k not in canary or k not in doc:
            return False
        a, b = doc[k][:n], canary[k][:n]
        if len(a) != n or len(b) != n or a != b:
            return False
    return True


def main():
    v = sim_version()
    have = sorted(p.name for p in BASELINES.glob(f"*_{v}.json")) if BASELINES.exists() else []
    print(f"simulator version {v}")
    print("baselines for it: " + (", ".join(have) if have else "none (the next 'before' evaluation plays the scripts' "
                                                              "battles: ~8 min more on the CPU; the canary may adopt an older one)"))


if __name__ == "__main__":
    main()

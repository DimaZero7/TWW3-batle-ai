"""The simulator's version for the script baselines' cache, and the comparison that lets an older
version's baseline be adopted (docs/en/training/training.md "Test protocol").

Plain Python (no torch): tools/ops/step.py asks it on the host whether the next run's "before"
evaluation will find its baselines (a cache miss costs ~20-30 minutes: the three scripts play their
256 pairs again), and tools/nn/train/evaluate.py imports it for the same hash.

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
VERSION_FILES = ("tools/nn/sim", "tools/nn/armies", "tools/nn/train/opponents.py", "tools/nn/train/scenes.py",
                 "tools/nn/train/reward.py", "tools/nn/train/randomise.py", "tools/nn/scenario.py", "tools/nn/units.py",
                 "tools/nn/abilities.py", "tools/nn/model", "config/nn")
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
            if f.is_file() and f.suffix in (".py", ".json") and "__pycache__" not in f.parts:
                h.update(f.relative_to(ROOT).as_posix().encode())
                h.update(f.read_bytes().replace(b"\r\n", b"\n"))
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
                                                              "battles: ~20-30 min more; the canary may adopt an older one)"))


if __name__ == "__main__":
    main()

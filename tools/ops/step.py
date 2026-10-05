"""One step of the training chain from a few parameters (docs/en/training/workflow.md).

    python -m tools.ops.step --label n4_morale --from n3_consolidate
    python -m tools.ops.step --label n4 --from n3_consolidate/m15 --minutes 15 --set drills=0.1 --drop teach-normal
    python -m tools.ops.step --label n4 --from n3_consolidate --write build/steps/n4.sh -- --lord-rout 0.3

Builds the test5 trend-run command of the chain (docs/en/training/training.md "The chain's settings")
and prints it, or writes it as a bash script (--write): the previous step's last network is the
--init and the --reference, its run's latest.pt the --critic-init, the standing options come from
config/train-chain.json (--set key=value overrides one, --drop key removes one, anything after `--`
is appended to run.py's options as given), the container is named <dock_prefix><label>, the test
plays a baseline canary (--baseline-canary) so an unrelated code change does not cost the baselines.
--profile picks the evaluations' metric profiles (test5 --profile, tools/nn/train/profiles.py; default
config/train-chain.json test5.profile, else full): the mandatory set always, "drills,transfer" adds those.
It also says whether the baselines' cache will hit (tools/nn/train/version.py) and the expected wall
time. It never runs anything itself.
"""
import argparse
import json
import re
import shlex
import sys
from pathlib import Path

from tools import config as project
from tools.nn.train import profiles

ROOT = project.ROOT
CHAIN = ROOT / "config" / "train-chain.json"
TEST5 = project.BUILD / "nn-train" / "test5"
RUNS = project.BUILD / "nn-train" / "runs"
EVAL_S = 200.0            # one full evaluation with the baselines cached (n3: 194 s)
MISS_S = 600.0            # extra on a reference cache miss: tools/nn/train/refs.py on the CPU (~10 min, 8 cores)
STARTUP_S = 120.0         # the container, the compile warm-up


def load_chain(path=CHAIN):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def chain_point(spec, test5=None, runs=None):
    """("label", init, critic) of the previous step `spec`: "<label>" takes its last m<minute>.pt,
    "<label>/m<minute>" that network; the critic is its run's latest.pt (None when missing)."""
    test5, runs = TEST5 if test5 is None else test5, RUNS if runs is None else runs
    label, _, point = spec.replace("\\", "/").partition("/")
    folder = Path(test5) / label
    if not folder.is_dir():
        raise SystemExit(f"no test5 folder {folder}")
    if point:
        init = folder / (point if point.endswith(".pt") else f"{point}.pt")
        if not init.is_file():
            raise SystemExit(f"no network {init}")
    else:
        nets = sorted(folder.glob("m*.pt"), key=lambda p: float(p.stem[1:]) if re.fullmatch(r"m\d+(\.\d+)?", p.stem) else -1)
        if not nets:
            raise SystemExit(f"no m<minute>.pt in {folder}")
        init = nets[-1]
    critic = Path(runs) / f"test5_{label}" / "latest.pt"
    return label, init, (critic if critic.is_file() else None)


def rel(path):
    """The path as the container sees it (relative to the repository, forward slashes); a path outside
    the repository stays absolute (tests)."""
    try:
        return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return Path(path).resolve().as_posix()


def option_args(options):
    """["--key", "value", ...] of an options dict: true a bare flag, null/false dropped, a dict or list as JSON."""
    out = []
    for k, v in options.items():
        if v is None or v is False:
            continue
        out.append(f"--{k}")
        if v is True:
            continue
        out.append(json.dumps(v, separators=(",", ":")) if isinstance(v, (dict, list)) else str(v))
    return out


def apply_sets(options, sets, drops):
    """options with --set key=value (a JSON value when it parses, else the text) and --drop key applied."""
    out = dict(options)
    for s in sets:
        k, eq, v = s.partition("=")
        if not eq:
            raise SystemExit(f"--set needs key=value, got {s!r}")
        try:
            out[k.lstrip("-")] = json.loads(v)
        except ValueError:
            out[k.lstrip("-")] = v
    for k in drops:
        out.pop(k.lstrip("-"), None)
    return out


def posix(path):
    """C:\\x\\y -> /c/x/y for Git Bash's cd."""
    p = Path(path).resolve().as_posix()
    return re.sub(r"^([A-Za-z]):/", lambda m: f"/{m.group(1).lower()}/", p)


def baseline_cache():
    """(version, [missing references]) of the current code: the script baselines and the drills' check
    scripts (tools/ops/baselines.py status); the import of the generator may fail outside the project's
    environment -> (None, [])."""
    try:
        from tools.nn.train import version
        from tools.ops import baselines
        v, _, missing = baselines.status(version.BASELINES)
    except Exception as e:                    # noqa: BLE001 - a report, not a failure
        print(f"(baseline cache not checked: {e})", file=sys.stderr)
        return None, []
    return v, missing


def wall_minutes(minutes, every, miss):
    """A rough end-to-end estimate: training + the evaluations (before, every point, after) + start-up."""
    evals = (int(minutes / every) if every else 0) + 2
    return (minutes * 60 + evals * EVAL_S + (MISS_S if miss else 0) + STARTUP_S) / 60, evals


def build(label, prev, chain, minutes=None, every=None, sets=(), drops=(), extra=(), canary=None, prefix=None,
          profile=None):
    """(script lines, info) of the step."""
    t5 = chain.get("test5", {})
    profile = t5.get("profile") if profile is None else profile
    if profile:
        profiles.parse(profile, profiles.TEST5, profiles.TEST5_ALIASES)     # a wrong name fails here, not in the run
    minutes = t5.get("minutes", 25) if minutes is None else minutes
    every = t5.get("every", 5) if every is None else every
    canary = t5.get("baseline_canary", 0) if canary is None else canary
    prefix = t5.get("dock_prefix", "orch-") if prefix is None else prefix
    prev_label, init, critic = chain_point(prev)
    options = apply_sets(chain.get("options", {}), sets, drops)
    head = ["--critic-init", rel(critic)] if critic else []
    run_opts = head + ["--reference", rel(init)] + option_args(options) + list(extra)
    test5_args = ["--label", label, "--init", rel(init), "--updates", "0", "--minutes", f"{minutes:g}", "--every", f"{every:g}"]
    if canary:
        test5_args += ["--baseline-canary", str(canary)]
    if profile and profile != profiles.FULL:
        test5_args += ["--profile", profile]
    cmd = (f"DOCK_NAME={shlex.quote(prefix + label)} bash tools/nn/dock.sh tools.nn.train.test5 "
           + " ".join(shlex.quote(a) for a in test5_args) + " -- " + " ".join(shlex.quote(a) for a in run_opts))
    lines = [f'cd "{posix(ROOT)}"', cmd]
    info = {"label": label, "prev": prev_label, "init": rel(init), "critic": rel(critic) if critic else None,
            "minutes": minutes, "every": every, "canary": canary, "container": prefix + label, "options": options,
            "extra": list(extra), "out": f"build/nn-train/test5/{label}", "profile": profile or profiles.FULL}
    return lines, info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help="the new step's test5 label (build/nn-train/test5/<label>)")
    ap.add_argument("--from", dest="prev", required=True, help="the previous step: <label> (its last m*.pt) or <label>/m<minute>")
    ap.add_argument("--minutes", type=float, help="training minutes (default: config/train-chain.json)")
    ap.add_argument("--every", type=float, help="minutes between evaluations (default: the config)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a run.py option (JSON value or text)")
    ap.add_argument("--drop", action="append", default=[], metavar="KEY", help="remove a run.py option")
    ap.add_argument("--canary", type=int, help="baseline canary pairs (default: the config; 0 off)")
    ap.add_argument("--prefix", help="the container's name prefix (default: the config)")
    ap.add_argument("--profile", help="metric profiles of the evaluations (test5 --profile: mandatory, full or a "
                                      "comma list; default: the config's test5.profile, else full)")
    ap.add_argument("--chain", default=str(CHAIN), help="the options file")
    ap.add_argument("--write", help="write the script here (LF); print the command to run it")
    args, extra = ap.parse_known_args(argv)
    extra = [a for a in extra if a != "--"]
    chain = load_chain(args.chain)
    lines, info = build(args.label, args.prev, chain, args.minutes, args.every, args.set, args.drop, extra,
                        args.canary, args.prefix, args.profile)
    print("\n".join(lines))
    print()
    print(f"chain: {info['prev']} -> {info['label']}: init {info['init']}, critic {info['critic'] or 'NONE (warm start of the critic!)'}")
    version, missing = baseline_cache()
    if version is not None:
        if missing:
            print(f"baseline cache: MISS for {', '.join(missing)} (simulator version {version}): the step plays them "
                  f"on the CPU first (~{MISS_S / 60:.0f} min); ahead of it: .venv/Scripts/python -m tools.ops.baselines --run")
        else:
            print(f"baseline cache: hit (simulator version {version})")
    wall, evals = wall_minutes(info["minutes"], info["every"], bool(missing))
    print(f"expected wall time ~{wall:.0f} min: {info['minutes']:g} min training + {evals} evaluations x ~{EVAL_S / 60:.1f} min"
          + (f" + ~{MISS_S / 60:.0f} min of script references" if missing else ""))
    if args.write:
        path = Path(args.write)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        log = path.with_suffix(".log")
        print(f"written {path}; run:  bash {posix(path)} > {posix(log)} 2>&1")
        print(f"wait:   bash tools/ops/wait.sh -t {int(wall * 60 * 1.5) + 600} {posix(log)} '^written:|Traceback|Error'")
    print(f"card:   .venv/Scripts/python -m tools.ops.card {info['out']} --prev build/nn-train/test5/{info['prev']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""The card (tmp/engine/design_merged.md section 3; brief_A2 Part 1): one drill as data. A card is a module
`net/drill/cards/<id>.py` with `card() -> Card`; `run.py train --card <id>` reads it by get(id) and its level by
levels.get(card.level) - the engine names no card. What the run reads from the card:

    id, level           the card's shown id ("T0-2") and its level module (net/drill/levels/__init__.py); --card and
                        parent name the card's MODULE ("t0_2")
    parent              the previous card's module name or None. A card with a parent needs `--init PATH` (no default weights):
                        PATH's sha256 is checked against the parent card's accepted run.json (`--parent-run DIR`, or
                        the run.json beside PATH) when that exists; the parent's sha enters the run key; V0 = the
                        parent's marks, read before any child weight is loaded. No parent: the net from zero, no V0
    files               the card's own files beyond its module (hashed into the run key with the module of THIS card and
                        of its parents only - editing another card leaves the key)
    level_args          the level's knobs of this drill (tactic: drill 'attack' / 'side', the teacher, the pre-pass,
                        the checkpoint tags) - read by the level only
    stages              the collection plan the level's collect_round reads (teacher, sets, shares of old drills and
                        of general battles); training ranges are the registry's entries of this card's split 'train'
    masks               {head: masks.of kind} - the ONE definition of "this label trains this head" (fit, census, V,
                        probe)
    census              {head: {class: minimum labels with a gradient a round}} (default CENSUS_MIN); dont_care_max
                        {head: {class: share}} (over it halts); band_share_max (tactic approach: the band's share);
                        masks_moved_max (old-drill labels the child's masks may move against the parent's - more halts)
    marks(v) -> {mark: {path: cell}}   the stop's marks from the level's V read, cell {share, two_errors, n}; the
                        mark's name says its denominator
    marks_lines         {mark: (">=" | "<=", line)} on path T; path N: not worse than T by more than two errors; every
                        cell needs n >= its minimum (mark_min {mark: {path: n}}, default MARK_MIN_N), else the round
                        is "not judged" (no stop); a mark whose cells carry no "T" is judged on path N alone
    reference           {path, sha256, why} or None: a network whose V read (v_round_ref.json, read once before the
                        training like V0, its sha and the run key in the file) the stop compares with - path T (or N
                        alone) not worse than the reference's by more than two errors; the absolute lines (absolute,
                        marks_lines) are then printed only. No parent and no reference: the absolute lines gate
    retention(v) -> {mark: (">=" | "<=", cell)}   the old cards' marks (vs V0 of the parent: not worse beyond two
                        errors); None for a card without a parent
    absolute(v) -> (ok, fails, judged)   the card's own absolute stop lines (T0-1: imitate.m3), n >= MARK_MIN_N
    pieces_before, pieces_after, m2(tables, ...) the exam's pieces (exam.py / verdict.py): kind skill | safety |
                        report, strata with n minima BEFORE pooling, the declared window and the execution denominator
    value_gate          (errors, absolute) - which of "half the gap" / "not worse than the teacher" applies is decided
                        on the teacher's exam and stored with the card's exam files (exam.value_gate_of)
The schedule (net/config/drill.json "schedule") is the run's, not the card's; kind 'aggregate' is refused (not built).
"""
import importlib
from dataclasses import dataclass, field
from typing import Callable, Literal

CENSUS_MIN = 50          # (brief_A2 Part 4: a class's labels reaching its head with a gradient, a round - the default)
MARK_MIN_N = 50          # (brief_A2 Part 3: every share the stop reads; validate.py's legacy n >= 50 on path T)


@dataclass(frozen=True)
class Piece:
    id: str
    kind: Literal["skill", "safety", "report"]
    legacy: bool
    arms: tuple                       # all on the same paired battle ids, else NOT JUDGED
    strata: tuple                     # e.g. "taught:cause[-comer]"
    denominators: tuple               # each must pass
    n_min: int                        # per stratum and denominator
    read: Callable                    # (tables, stratum, denominator) -> (value, two errors, n)
    line: Callable                    # (value, two errors) -> "pass" | "fail" | "undecided"


@dataclass(frozen=True)
class SetSpec:
    seed0: int
    n: int
    chunk: int
    held: str | None = None
    blocks: int = 256                 # balanced blocks, registered append-only


@dataclass(frozen=True)
class Scenes:
    make: Callable                    # (seed, held=None, twin=None) -> spec
    axes: dict                        # {axis: held value(s)}
    twins: tuple = ()
    family: Callable | None = None    # spec -> family id (seeds.family by default)
    binds_spec: Callable | None = None
    binds_play: Callable | None = None


@dataclass(frozen=True)
class Card:
    id: str
    words: str
    cause: str
    level: Literal["tactic", "operative", "strategist"]
    parent: str | None
    stage: str                        # tactic "target" | "approach"; operative "line" | "skill"
    teacher: Callable | str           # memoryless stage: (view, out so far) -> out (tactic: the runner's player name)
    disturb: Callable | None          # (Random, list) -> plans (tactic: train.learn.disturb, in the level)
    arms: dict
    below: dict
    opponent: tuple
    scenes: Scenes | None
    sets: dict                        # train, dev, V, probe_train, probe_held, taught, held:*, G -> SetSpec
    mass: float
    classes: Callable | None          # Batch -> {class: [M, G] bool}
    band_share_max: float | None
    masks_changed: dict
    marks: Callable                   # (module doc)
    marks_lines: dict
    pieces_before: tuple = ()
    pieces_after: tuple = ()
    value_gate: tuple = (4.0, 0.02)
    head: Callable | None = None
    watch: type | None = None
    probe_ablate: tuple = ()
    lane: str | None = None
    files: tuple = field(default_factory=tuple)
    # (brief_A2 Part 1, additive: the run's reads)
    level_args: dict = field(default_factory=dict)
    stages: dict = field(default_factory=dict)
    masks: dict = field(default_factory=dict)
    census: dict = field(default_factory=dict)
    dont_care_max: dict = field(default_factory=dict)
    masks_moved_max: int = 0
    retention: Callable | None = None
    absolute: Callable | None = None  # (v) -> (ok, fails, judged): a card's own absolute stop lines (T0-1's M3)
    m2: Callable | None = None
    exec_denominator: str | None = None
    # (brief_A3, additive)
    reference: dict | None = None     # {path, sha256, why}: the stop relative to this network's V read (module doc)
    mark_min: dict = field(default_factory=dict)  # {mark: {"T" | "N": n}} a cell's minimum, default MARK_MIN_N


def get(card_id):
    """card() of the module net.drill.cards.<card_id>."""
    try:
        mod = importlib.import_module(f"net.drill.cards.{card_id}")
    except ModuleNotFoundError as e:
        raise SystemExit(f"--card {card_id}: no card module net/drill/cards/{card_id}.py ({e})")
    if not hasattr(mod, "card"):
        raise SystemExit(f"net/drill/cards/{card_id}.py has no card()")
    return mod.card()


def chain(card_id):
    """[card ids] from the root to card_id along the parents (a cycle halts)."""
    out, cur = [], card_id
    while cur is not None:
        if cur in out:
            raise SystemExit(f"card parents form a cycle: {out + [cur]}")
        out.append(cur)
        cur = get(cur).parent
    return out[::-1]


def module_file(card_id):
    from pathlib import Path
    return Path(__file__).resolve().parent / f"{card_id}.py"

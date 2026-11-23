"""Level adapters of the drill engine (tmp/engine/design_merged.md sections 2-4; brief_A2 Part 1) - the ONE interface
through which `python -m net.drill.run train --card <id>` trains a card of any level. run.py, collect.py, fit.py,
census.py and validate.py name no level and no card: what is level-specific is a module `net/drill/levels/<name>.py`
(name = Card.level), what is drill-specific is the card (net/drill/cards/__init__.py). Loaded by get(name).

A level module defines (ctx below = the run's context: SimpleNamespace(device, sim, smoke, no_general, params, chunk,
log, registry, out) - the run fills it, the level reads what it needs):

    NAME            the level's name (= the module's)
    tick_s          the decision tick, s
    IMPL_FILES      globs (relative to the project root) of the level's and the unit script's implementation hashed into
                    the run key (tactic: net/tactic/*.py, net/unit/*.py, net/stage0/*.py); the engine's own net/drill/*.py
                    less net/drill/cards/ is hashed by the run itself
    conf(card)      the fit's knobs {lr, minibatch, max_grad, tseed, chunk, pre_pass (bool: fit's pass -1, the losses
                    before any update - run.py reads it, a KeyError without)} and under "key" the level's effective configs
                    for the run key with every cap removed (a cap is the schedule's n_max, never a config)
    sim_key()       the simulator's key (the run key and V's cache key)
    net_new(card, device)              the network from zero (a card without a parent); the run seeds torch before
    net_parent(card, path, device)     the network loaded from the parent card's accepted weights file `path`
    net_reference(card, path, device)  (a card with a reference) the reference network in the card's own form
    collect_round(card, r, net, ctx, store_dir) -> (store.Store, stats)   round r's collection (0-based) into a
                    fresh shard folder (store.Store(store_dir, ctx.run_key, r + 1)): the card's stages (teacher, sets,
                    shares, disturbance), the teacher or the network driving; once a chunk it calls
                    ctx.check_starts([(generator, key, value)]) with the realised starts (spec seed, substitute, retry
                    base, torch chunk start; collect.starts_check -> seeds.assert_starts on the card's training ranges);
                    stats carries "done_rows" {head: (rows, left_out)} (census.check iterates its items - never an
                    int), "starts_checked", "NK" and the rebuild-check counts
    remap(rows, table)                 a shard's handles rewritten for this process (store.mixture_of's remap; tactic:
                    the map registry) - None: none
    pad(parts) -> parts                rows of shards of different battle sizes made stackable (store.mixture_of; zeros
                    to the max N, "n_slots" recorded per row) - None: shapes always agree
    weights(card, rows, meta) -> (w [M, G], info)   the mixture-wide weights (masses, class splits), the drop from
                    masks.of only
    batch_of(rows, cards, idx) -> record.Batch     the fit's / census's / V's view of stored rows (fit.passes' batch_of)
    losses(net, batch, w) -> {term: tensor | None} and logits(net, batch)   record.Level (record.loss sums the terms)
    census_batch(rows) -> record.Batch   the mixture's view for the census (no input needed)
    census_classes(card, batch, w, meta) -> {head: {class: census.Cls}}  per head and class the label masks: labels,
                    dont_care, weighted, grad (weight > 0, masks.of eligible and >= 2 open candidates: a label that
                    moves the head), removed {reason: mask}; meta per battle for the cells (start)
    census_extra(card, batch, rows, meta, w) -> dict   the level's printed numbers (tactic: side_count,
                    flank_weighted, band_share, band_n) - census.check halts on band_share > card.band_share_max
    parent_masks(card) -> (parent_fn, child_fn, classes_fn)   old_masks' two eligibility functions (batch -> [M, G])
                    and the old drill's label classes (batch, meta) -> {class: [M, G]}
    v_read(card, net, ctx, cache) -> dict   validation V: path T from validate.TeacherCache (played once under the
                    level's cache key - teacher and unit-script files, stage0 / lord config, the recorder's files, the
                    specs with their twin / held annotations, seeds, chunk; on reuse its first chunk replayed and
                    compared), path N played; the card names the marks (card.marks / card.absolute / card.retention).
                    Once a chunk it calls ctx.check_starts_v([(generator, key, value)]) with V's realised starts
                    (collect.starts_check on the registry's 'V' ranges of the card and its parents)
    brief(v) -> dict                   V's numbers in one row of the log
    seed_check(card, log)              (optional) the level's own legacy seed checks
    save(path, net, opt, card, state)  an atomic checkpoint: the net, opt, the generator and rng states, state's run key,
                    round and stop
    load(path, device) -> dict          {net, opt, gen_state, rng, cuda_rng, run_key, round, stop}
"""
import importlib

NAMES = ("tactic", "operative", "strategist")


def get(name):
    """The level module `net.drill.levels.<name>` (one of NAMES)."""
    if name not in NAMES:
        raise SystemExit(f"level {name!r}: not one of {NAMES}")
    return importlib.import_module(f"net.drill.levels.{name}")

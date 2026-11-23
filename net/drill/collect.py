"""A training round's collection (tmp/engine/design_merged.md section 4), level-blind: the level's collect_round plays
(net/drill/levels/__init__.py); this module gives it a fresh shard folder under a writer's lock and the registry's start
check.

    fresh(out, r)          DIR/collect_rNN/ empty: an unfinished round's shards (no done-record) moved away to
                           collect_rNN.staleK (collected again from the same draw)
    starts_check(entries, cards, split)   the callable a level calls once a chunk with its realised starts [(generator,
                           key, value)]: seeds.assert_starts on the registry's `split` ranges ('train' by default; V's:
                           'V') of the card and its parents
    round_of(level, card, r, net, ctx, out, cards)   under the round's lock DIR/collect_rNN.writer.lock (store.Lock:
                           a second writer refused, a dead or silent holder of the same run key taken over), fresh(),
                           then level.collect_round with ctx.check_starts set -> (Store, stats)
"""
from pathlib import Path

from net.drill import seeds as SD
from net.drill import store as ST


def fresh(out, r):
    sdir = Path(out) / f"collect_r{r + 1:02d}"
    if (sdir / "manifest.json").exists():
        n = 1
        while (Path(out) / f"collect_r{r + 1:02d}.stale{n}").exists():
            n += 1
        sdir.rename(Path(out) / f"collect_r{r + 1:02d}.stale{n}")
    return sdir


def starts_check(entries, cards, split="train"):
    cards = set(cards)

    def check(starts):
        return SD.assert_starts(entries, cards, split, starts)
    return check


def round_of(level, card, r, net, ctx, out, cards):
    """(module doc) the round's lock (DIR/collect_rNN.writer.lock, outside the folder) FIRST, then fresh() moves an
    unfinished folder away, then the level collects (brief_A3 R4c)."""
    ctx.check_starts = starts_check(ctx.registry, cards)
    with ST.Lock(Path(out) / f"collect_r{r + 1:02d}.writer.lock", getattr(ctx, "run_key", None),
                 getattr(ctx, "log", None)):
        sdir = fresh(out, r)
        return level.collect_round(card, r, net, ctx, sdir)

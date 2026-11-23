"""Per-head masks (tmp/engine/design_merged.md section 4): ONE function per head - fit, probe, V and census call only
it. of(head, ...) -> Masks(legal, dont_care, eligible, why).

Reason bits (why, int32; a unit may carry several):
    R_DELEGATED  not decided at all (tactic: not head-free; operative: melee veto, no facts, off-tick)
    R_DONE       the battle is over (its frozen state seen again)
    R_MELEE      a member / the unit in melee (operative: the hygiene veto)
    R_NO_FACTS   the level had no facts (operative: no standing member or no E)
    R_OFF_TICK   not a decision second of the level
    R_PAD        a padded slot (tactic: G < GMAX)
    R_DROPPED    the tactic label dropped (T0-1's 'dropped')
    R_DC_BAND    the label's key band (T0-1's dc_label: band part)
    R_DC_THR     the label 'through own' (T0-1's dc_label: threshold part)
    R_THR_LIFT   through own lifted for a flank approach (drill T0-2, imitate.py:91-97): counted, not dropped
    R_SIDE_BAND  the side label at its rule's threshold (learn.side_band)
    R_SIDE_CLOSED the label's row closed for the pointer under the walk's mask of its slot (an earlier reservation)
    R_OUT_OF_SET the pointer label not allowed (heads.loss: closed row or a don't-care key)
    R_SAME       operative: the line pointer equals the order in force (lines.Level.facts' 20 m equivalence)
"""
from dataclasses import dataclass

import torch

R = ("DELEGATED", "DONE", "MELEE", "NO_FACTS", "OFF_TICK", "PAD", "DROPPED", "DC_BAND", "DC_THR", "THR_LIFT",
     "SIDE_BAND", "SIDE_CLOSED", "OUT_OF_SET", "SAME")
BIT = {n: 1 << i for i, n in enumerate(R)}
R_DELEGATED, R_DONE, R_MELEE, R_NO_FACTS, R_OFF_TICK, R_PAD = (BIT[n] for n in R[:6])
R_DROPPED, R_DC_BAND, R_DC_THR, R_THR_LIFT, R_SIDE_BAND, R_SIDE_CLOSED, R_OUT_OF_SET, R_SAME = (BIT[n] for n in R[6:])


def names(bits):
    """The reason names of one int."""
    return [n for n in R if int(bits) & BIT[n]]


@dataclass
class Masks:
    legal: torch.Tensor          # [.., C] bool
    dont_care: torch.Tensor      # [.., C] bool
    eligible: torch.Tensor       # [..] bool: the unit's label counts (decided, not dropped)
    why: torch.Tensor            # [..] int32 reason bits
    item: torch.Tensor = None    # [..] bool: decided and its label not dropped (learn.weights' item; tactic pointer)


def _bits(*pairs, like):
    w = torch.zeros(like.shape, dtype=torch.int32, device=like.device)
    for m, bit in pairs:
        if m is not None:
            w |= m.to(w.device).bool().int() * bit
    return w


# ------------------------------------------------------------------ the tactic
FLANKS = (1, 2)                      # net/tactic/orders.py APPROACHES: front 0, left 1, right 2


def tactic_drop(drop_base, dc_thr, ap=None, lift=True):
    """[M, G] the dropped / don't-care pointer labels (imitate.drop_of, imitate.py:91-97): T0-1's dropped | dc_band
    (drop_base) | through own (dc_thr) - drill T0-2 (ap given, lift) lifts through own for a LEFT / RIGHT approach,
    FRONT keeps T0-1's exclusion; lift=False the T0-1 rule (flank_weight_share's 'without the fix')."""
    if ap is None or not lift:
        return drop_base.bool() | dc_thr.bool()
    flank = (ap == FLANKS[0]) | (ap == FLANKS[1])
    return drop_base.bool() | (dc_thr.bool() & ~flank)


def tactic_why(head, aux):
    """[M, G] the reason bits of stored tactic rows (record.from_netbuffer): delegated = not head-free (done rows are
    never stored, imitate.py:173); the label drop's parts when the buffer kept them (drill T0-2: dc_thr)."""
    w = _bits((~head, R_DELEGATED), like=head)
    if "drop" in aux:
        w |= _bits((aux["drop"].bool() & head, R_DROPPED), like=head)
    if "dc_thr" in aux:
        thr = aux["dc_thr"].bool() & head
        w |= _bits((thr & aux["drop"].bool(), R_DC_THR), (thr & ~aux["drop"].bool(), R_THR_LIFT), like=head) \
            if "drop" in aux else _bits((thr, R_DC_THR), like=head)
    return w


# ------------------------------------------------------------------ the operative
def op_defaults():
    """O0-1's head (net/operative/heads.py, NEEDS_A 1): candidates heads.CANDIDATES (KEEP + LS - L0 closed: the teacher
    never emits it; LM out), LS open unless it equals the line in force (facts 'ls_same')."""
    from net.operative import heads as HD
    return {"cands": tuple(HD.CANDIDATES), "open_": ("LS",), "same_key": "ls_same", "same_ptr": "LS"}


def operative(facts, done=None, on_tick=None, cands=None, open_=None, same_key=None, same_ptr=None):
    """The operative head's masks of B groups at one tick. facts: list of B lines.Level.facts dicts or None (no facts);
    done [B] bool (battle over); on_tick [B] bool (a decision second; default all). KEEP legal on every decided group;
    a pointer in open_ legal unless it equals the order in force (f[same_key]: lines.py:100's 20 m equivalence - a
    moved line is legal again, held seed 72001's re-issue at 465 s); the others closed. Delegated (no label, in no
    denominator): melee veto, no facts, off-tick, done. B may pass its own candidate names (the stand-off rename)."""
    d = op_defaults() if None in (cands, open_, same_key, same_ptr) else {}
    cands = d.get("cands") if cands is None else cands
    open_ = d.get("open_") if open_ is None else open_
    same_key = d.get("same_key") if same_key is None else same_key
    same_ptr = d.get("same_ptr") if same_ptr is None else same_ptr
    B = len(facts)
    C = len(cands)
    done = torch.zeros(B, dtype=torch.bool) if done is None else torch.as_tensor(done, dtype=torch.bool)
    on_tick = torch.ones(B, dtype=torch.bool) if on_tick is None else torch.as_tensor(on_tick, dtype=torch.bool)
    nof = torch.tensor([f is None for f in facts], dtype=torch.bool)
    melee = torch.tensor([bool(f is not None and f.get("melee")) for f in facts], dtype=torch.bool)
    same = torch.tensor([bool(f is not None and f.get(same_key)) for f in facts], dtype=torch.bool)
    legal = torch.zeros((B, C), dtype=torch.bool)
    legal[:, 0] = True
    for j, n in enumerate(cands):
        if n in open_:
            legal[:, j] = ~same if n == same_ptr else True
    deleg = nof | melee | ~on_tick | done
    legal &= ~deleg[:, None]
    why = _bits((deleg, R_DELEGATED), (done, R_DONE), (melee, R_MELEE), (nof, R_NO_FACTS), (~on_tick, R_OFF_TICK),
                (same & ~deleg, R_SAME), like=deleg)
    return Masks(legal=legal, dont_care=torch.zeros((B, C), dtype=torch.bool), eligible=~deleg, why=why)


def strategist(legal):
    """The strategist's one decision a battle (S0-1; was levels/strategist.mask_of): legal [M, C] -> [M, 1, C] (KEEP
    always legal, a candidate when valid - the level's legal as given), no don't-care, every battle eligible."""
    lg = torch.as_tensor(legal, dtype=torch.bool)[:, None, :]
    M = lg.shape[0]
    return Masks(legal=lg, dont_care=torch.zeros_like(lg), eligible=torch.ones((M, 1), dtype=torch.bool),
                 why=torch.zeros((M, 1), dtype=torch.int32))


def of(head, *a, **k):
    """The one entry: of('operative', facts, done=, on_tick=, ...) | of('tactic_pointer', batch, w) |
    of('tactic_approach', batch, w) | of('strategist', legal)."""
    if head == "strategist":
        return strategist(*a, **k)
    if head == "operative":
        return operative(*a, **k)
    if head == "tactic_pointer":
        return tactic_pointer(*a, **k)
    if head == "tactic_approach":
        return tactic_approach(*a, **k)
    raise KeyError(head)


def tactic_pointer(batch, w=None, lift=True):
    """Pointer head on stored tactic rows: legal / dont_care as recorded; eligible = decided and not dropped (the drop
    from its parts, tactic_drop, when the record kept them; else the buffer's own); the out-of-set labels (heads.loss: a
    closed row or a don't-care key at the label) carry R_OUT_OF_SET and are not eligible."""
    from net.drill.record import DECIDED, label_of
    dec = batch.status == DECIDED
    ax = batch.aux
    if "drop_base" in ax:
        drop = tactic_drop(ax["drop_base"], ax.get("dc_thr", torch.zeros_like(dec)), ax.get("ap"), lift) & dec
    else:
        drop = ax["drop"].bool() if "drop" in ax else torch.zeros_like(dec)
    why = batch.why.clone()
    lab = label_of(batch)
    allowed = batch.legal & ~batch.dont_care
    ok = allowed.gather(-1, (lab + 1).clamp(min=0)[..., None])[..., 0]
    oos = dec & ~ok & (lab >= 0)
    why |= _bits((oos, R_OUT_OF_SET), like=dec)
    elig = dec & ~drop & ~oos
    return Masks(legal=batch.legal, dont_care=batch.dont_care, eligible=elig, why=why, item=dec & ~drop)


def tactic_approach(batch, w=None):
    """Approach head (drill T0-2) on stored rows: eligible = an attack label with an approach, decided, (w > 0 when
    given), not at the rule's threshold (learn.side_band: R_SIDE_BAND), and its row open for the pointer under the
    walk's mask of its slot (R_SIDE_CLOSED) - imitate.side_mask."""
    from net.tactic import learn as LN
    from net.drill.record import DECIDED, label_of
    lab, ap = label_of(batch), batch.aux["ap"].long()
    base = (lab >= 0) & (ap >= 0) & (batch.status == DECIDED)
    if w is not None:
        base = base & (w > 0)
    band = LN.side_band(batch.aux["x_k"], ap) & base
    opn = batch.legal[..., 1:]
    closed = ~(opn.gather(-1, lab.clamp(min=0)[..., None])[..., 0] & (lab >= 0)) & base
    why = batch.why | _bits((band, R_SIDE_BAND), (closed, R_SIDE_CLOSED), like=base)
    C = len(("FRONT", "LEFT", "RIGHT"))
    legal = torch.ones(base.shape + (C,), dtype=torch.bool, device=base.device)
    return Masks(legal=legal, dont_care=~legal, eligible=base & ~band & ~closed, why=why)

"""Recorded tokens in today's token width and recorded ability slots in today's slots, without torch (the observation
store, store.py, and the student's data, tools/nn/student/dataset.py, read files written before they grew)."""
import numpy as np


def widen(a):
    """Tokens [..., F] in today's token: ones written before the new units (11.10.2026; 155 columns) get their columns
    at their new places and zeros for the new ones (observation.old_columns), as the network's weights (encoder.py);
    today's width as it is."""
    from tools.nn.model import observation as ob
    if a.shape[-1] == ob.TOKEN:
        return a
    cols = ob.old_columns(a.shape[-1])
    if cols is None:
        raise SystemExit(f"recorded tokens of {a.shape[-1]} columns, the network's {ob.TOKEN}: convert or record "
                         "again")
    out = np.zeros(a.shape[:-1] + (ob.TOKEN,), a.dtype)
    out[..., cols] = a
    return out


def widen_abil(a):
    """A recorded side's arrays (dict) with its abilities in today's slots (tools/nn/model/abilities.py), in place:
    one written before 13.10.2026 - abil [..., 3, OLD_SIZE] (the simulator's 3 slots: state and passport), abil_ok
    [..., 3], no abil_idx - gets abil [..., SLOTS, STATE] (its actives keep their slots, its passives move to ACTIVE +
    their place among them; on now: an active's active now, a passive's not known: 0), abil_idx (each slot's passport
    matched to the table's) and abil_ok [..., ACTIVE]. The abilities the 3 slots dropped stay missing (convert again
    for them). Today's as it is."""
    from tools.nn.model import abilities as mab
    if "abil" not in a or "abil_idx" in a:
        return a
    old = np.asarray(a["abil"], np.float32)
    if old.shape[-1] != mab.OLD_SIZE:
        raise SystemExit(f"recorded ability slots of {old.shape[-1]} columns: convert or record again")
    D, K = mab.DYNAMIC, old.shape[-2]
    tab = mab.table()[:len(mab.keys()), :mab.OLD_SIZE - D]                 # [A, the old passport] (no spell then)
    owned = old[..., mab.INDEX["owned"]] > 0.5                           # [..., K]
    flat = old[..., D:].reshape(-1, mab.OLD_SIZE - D)
    uniq, inv = np.unique(flat.round(3), axis=0, return_inverse=True)
    dist = np.abs(uniq[:, None, :] - tab[None]).max(-1)                  # [U, A]
    row = dist.argmin(-1)[inv.reshape(-1)].reshape(owned.shape)
    if (dist.min(-1)[inv.reshape(-1)].reshape(owned.shape)[owned] > 0.02).any():
        raise SystemExit("recorded ability slots that match no passport: convert again")
    passive = (tab[row, mab.INDEX["passive"] - D] > 0.5) & owned
    n_act = (owned & ~passive).sum(-1, keepdims=True)
    k = np.arange(K)
    dest = np.where(passive, mab.ACTIVE + k - n_act, k)                 # [..., K]
    state = np.zeros(owned.shape[:-1] + (mab.SLOTS, mab.STATE), np.float32)
    idx = np.full(owned.shape[:-1] + (mab.SLOTS,), -1, np.int64)
    st = np.concatenate([old[..., :D], np.where(passive, 0.0, old[..., mab.INDEX["active_now"]])[..., None],
                         np.zeros(owned.shape + (mab.STATE - mab.STATE_OLD,), np.float32)], -1)    # (no spell: 0)
    st = np.where(owned[..., None], st, 0.0)
    dest = np.where(owned, dest, mab.SLOTS - 1 - k)                      # (not owned: zeros on a free place)
    np.put_along_axis(state, np.broadcast_to(dest[..., None], dest.shape + (mab.STATE,)), st, -2)
    np.put_along_axis(idx, dest, np.where(owned, row, -1), -1)
    a["abil"] = state.astype(a["abil"].dtype)
    a["abil_idx"] = idx.astype(np.int16)
    if "abil_ok" in a:
        ok = np.zeros(owned.shape[:-1] + (mab.ACTIVE,), bool)
        ok[..., :K] = np.asarray(a["abil_ok"], bool) & ~passive
        a["abil_ok"] = ok
    return a

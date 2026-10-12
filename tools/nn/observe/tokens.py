"""Recorded tokens in today's token width, without torch (the observation store, store.py, and the student's data,
tools/nn/student/dataset.py, read files written before the token grew)."""
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

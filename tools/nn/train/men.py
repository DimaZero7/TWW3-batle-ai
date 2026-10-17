"""The fighters lab in training (run.py --fighters; docs/en/training/fighters_lab.md "In training").

With --fighters the training's simulator runs with the lab's men (tools/nn/fighters/hook.py, installed for the whole
run with adopt=False, shape=True): every man a point with his health and place; contact, melee, missiles and spells'
blows by the men; morale, routs, orders and the rest the simulator's, as in the lab. The network sees the units' shape
(observation SHAPE: hook.shape_of after every step).

The men live beside the batch's State (st.men, a tools/nn/fighters/state.Fighters of B battles x M places) and go
wherever the State goes:

    open(st, params)           a batch's start: every battle's men (rollout.Battles.start)
    too_big(st, rows)          [B] the battles of rows with more men than M (rollout: picked again from the bank)
    restart(st, params, rows)  the battles of rows start again (rollout.Battles._reset: after the bank's row, the
                               randomised places and the domain randomisation): their men built anew into the batch's
                               rows, their shape now
    narrow(st, new, keep)      the evaluation's smaller batch (rollout.Battles.narrow): the men of those battles
    snap(st, idx)              the ability branches' snapshot (ability_branch.snapshot): the men of battles idx, as
                               the snapshot's "f/<field>" entries (selected and joined with the rest of it)
    branch(template, st, snap) a branch batch's State (ability_branch.branch_env) gets the snapshot's men

The batch's sizes are fixed (the step is compiled for them, tools/nn/fighters/hook.Hook._runner; a new size compiles
~4 min): M places a battle (Men.M; --fighters-men, default 7168: the v2 banks' battles have 3,070 men on average,
p99 5,900, max 6,800 of 6,144 battles), L men of splash units a battle (32; the banks' most 2), the most men a blow
splashes 8 (state.build's cap; the results the same for any value >= the batch's). A battle of more men than M is not
played (picked again; none in 6,144 battles at 7168). M None (evaluation, no restarts): the opened batch's most,
rounded up to 256.

Needs torch."""
import torch

from tools.nn.fighters import hook as FH
from tools.nn.fighters import state as FS

M_DEFAULT = 7168
L_DEFAULT = 32
SPLASH_S = 8
ROUND = 256


def row_fields(F):
    """The names of F's per-battle tensors (first dimension B): everything but its scalars (tick, seed, ...)."""
    B = F.B
    return [k for k, v in vars(F).items() if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == B]


def select(F, idx):
    """The men of battles idx (a long tensor) as Fighters of their own; the scalars shared."""
    rows = set(row_fields(F))
    return FS.Fighters(**{k: (v[idx] if k in rows else (v.clone() if torch.is_tensor(v) else v))
                          for k, v in vars(F).items()})


def _fresh(v):
    """A per-man field a step sets (melee.resolve's eng, fired) for a battle that has not stepped yet."""
    if v.dtype == torch.bool:
        return torch.zeros((), dtype=v.dtype, device=v.device)
    if v.dtype in (torch.int64, torch.int32):
        return torch.full((), -1, dtype=v.dtype, device=v.device)
    return torch.zeros((), dtype=v.dtype, device=v.device)


class _View(dict):
    """A State's u restricted to battles idx: each field indexed when first read (fighters' build and shape read a few
    of the ~250)."""

    def __init__(self, u, idx):
        super().__init__()
        self.base, self.idx = u, idx

    def __missing__(self, k):
        v = self.base[k][self.idx]
        self[k] = v
        return v


class _Rows:
    """Battles idx of a State as fighters.state.build reads one (u, keys)."""

    def __init__(self, st, idx):
        self.u = _View(st.u, idx)
        self.keys = [st.keys[i] for i in idx.tolist()]


def men_of(st):
    """[B] the men a battle has places for (its units' men0)."""
    u = st.u
    return torch.where(u["side"] > 0, u["men0"].round(), torch.zeros_like(u["men0"])).sum(1)


class Men:
    """The men of a training's batches (module doc). hook: the installed fighters hook (default fighters.hook.HOOK)."""

    def __init__(self, M=M_DEFAULT, L=L_DEFAULT, splash_S=SPLASH_S, seed=0, hook=None):
        self.M = None if M is None else int(M)
        self.L, self.S = int(L), int(splash_S)
        self.seed = int(seed)
        self.hook = hook
        self.builds = 0            # every build draws its men's places' scatter anew (state.build's seed)
        self.restarts = 0          # battles started again (the log)

    @property
    def h(self):
        h = self.hook or FH.HOOK
        if h is None:
            raise RuntimeError("men: the fighters hook is not installed (fighters.hook.installed(adopt=False))")
        return h

    def _start_phase(self):
        return bool(self.h.lab["melee"].get("start_phase"))

    def _build(self, st, params):
        F = FS.build(st, params, seed=self.seed + 7919 * self.builds, start_phase=self._start_phase(), M=self.M,
                     L=self.L, splash_S=self.S)
        self.builds += 1
        return F

    def too_big(self, st, rows):
        """[B] the battles of rows (bool [B]) whose men the batch has no places for."""
        if self.M is None:
            return torch.zeros_like(rows)
        return rows & (men_of(st) > self.M)

    def open(self, st, params):
        """Every battle of the batch st gets its men (st.men) and its units' shape."""
        if self.M is None:
            need = int(men_of(st).max())
            self.M = max(ROUND, -(-need // ROUND) * ROUND)
        F = self._build(st, params)
        F.seed = self.seed
        self.h.attach(st, F)
        return F

    def restart(self, st, params, rows):
        """The battles of rows (bool [B]) started again in st (in place): their men anew, their shape now."""
        idx = rows.nonzero().flatten()
        if not len(idx):
            return
        F = st.men
        sub = _Rows(st, idx)
        G = self._build(sub, params)
        for k in row_fields(F):
            v = getattr(F, k)
            g = getattr(G, k, None)
            if k in ("x_last", "z_last"):
                g = st.u["x" if k == "x_last" else "z"][idx]
            # (out of place: a field may share its storage with another or with the State's - b_last, a step's output)
            g = _fresh(v).expand((len(idx),) + tuple(v.shape[1:])) if g is None else g.to(v.dtype)
            setattr(F, k, v.index_put((idx,), g))
        u = st.u
        if "shape" in u:
            s, ok = FH.shape_of(sub.u, G)
            u["shape"] = u["shape"].index_put((idx,), s)
            u["shape_ok"] = u["shape_ok"].index_put((idx,), ok)
        self.restarts += len(idx)

    def narrow(self, st, new, keep):
        """new (the battles keep of st, rollout.Battles.narrow) gets their men."""
        self.h.attach(new, select(st.men, keep))

    def snap(self, st, idx):
        """{"f/<field>": [len(idx), ...]} the men of battles idx (ability_branch.snapshot)."""
        F = st.men
        return {"f/" + k: getattr(F, k)[idx] for k in row_fields(F)}

    def branch(self, template, st, snap):
        """st (a branch batch of snapshot battles, ability_branch.branch_env) gets the snapshot's men; the scalars (the
        random draws' seed and step, the splash size) the template batch's."""
        T = template.men
        rows = {k[2:]: v.clone() for k, v in snap.items() if k.startswith("f/")}
        scal = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in vars(T).items() if k not in rows
                and not (torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == T.B)}
        F = FS.Fighters(**scal, **rows)
        self.h.attach(st, F)
        return F


__all__ = ["Men", "row_fields", "select", "men_of", "M_DEFAULT"]

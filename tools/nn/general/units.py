"""The general's army input for the simulator's batches: army_input.py's slot numbers of every unit of a battle, as
tensors, without a Python loop per decision.

The battle's own N slots are the general's slots (both sides in one row, `own` says whose): the slots' order means
nothing to the army reader and the councils (they read every slot with the same weights, test
test_general_net_torch.py "permutation"), so the design's "40 own + 40 enemy" order is not needed - N <= 80.

    table = UnitTable()                    # every passport of config/nn/units.json: numbers [K, 49], labels [K, 3]
    num, lab, full = table.battle(st.keys) # [B, N, 49], [B, N, 3] (empty slots 0), [B, N, 3] - once a battle
    num = dynamic(num, full, u, own)       # the side flag, the come shares and the state now (army_input "state")
"""
import numpy as np
import torch

from tools.nn.general import army_input as ai
from tools.nn.model import passport as pp

SIDE, MEN_COME, HP_COME = ai.INDEX["side"], ai.INDEX["men_come"], ai.INDEX["hp_come"]
MEN_FULL, HP_FULL = ai.INDEX["men_full"], ai.INDEX["hp_full"]
MORALE, WAVERING, FATIGUE, AMMO_COME = (ai.INDEX[k] for k in ai.STATE)
FATIGUE_MAX = 5.0          # the fatigue states 0 fresh .. 5 exhausted (the simulator's fat, the token's fatigue_*)
WAVERING_MS = 4.5          # the morale state 5 wavering, 6 routing, 7 shattered (MORALE_STATES)


class UnitTable:
    """Every passport's slot (army_input.slot of final_stats: the passport's numbers are the final ones in the simulator,
    rank 0) as one table; row 0 is the empty slot (zeros)."""

    def __init__(self, passports=None):
        passports = passports or pp.load()
        self.keys = [""] + sorted(passports)
        self.index = {k: i for i, k in enumerate(self.keys)}
        nums = np.zeros((len(self.keys), ai.SIZE), np.float32)
        labs = np.zeros((len(self.keys), len(ai.LABELS)), np.int64)
        # men, health and projectiles of the full unit (the come shares; projectiles: passport.ammo, the whole unit's,
        # as the simulator's ammo0 and the token's ammo)
        full = np.zeros((len(self.keys), 3), np.float32)
        for k in self.keys[1:]:
            st = ai.final_stats(passports[k], unit=k)
            nums[self.index[k]], labs[self.index[k]] = ai.slot(st, own=False, unit=k)
            full[self.index[k]] = (st["men_full"], st["hp_full"], pp.ammo([k], passports)[0])
        self.numbers, self.labels, self.full = nums, labs, full
        self._dev = {}

    def on(self, device):
        key = str(device)
        if key not in self._dev:
            self._dev[key] = tuple(torch.as_tensor(a, device=device) for a in (self.numbers, self.labels, self.full))
        return self._dev[key]

    def rows(self, keys):
        """[B, N] int64 of unit keys [B][N] ("" empty). A key without a passport is an error (army_input's rule)."""
        try:
            return np.array([[self.index[k or ""] for k in row] for row in keys], np.int64)
        except KeyError as e:
            raise KeyError(f"unit {e.args[0]!r} has no passport (config/nn/units.json)") from None

    def battle(self, keys, device="cpu"):
        """(numbers [B, N, 49], labels [B, N, 3], full [B, N, 3]) of the battles' units (keys [B][N])."""
        nums, labs, full = self.on(device)
        r = torch.as_tensor(self.rows(keys), device=device)
        return nums[r], labs[r], full[r]


def dynamic(num, full, u, own, visible=None):
    """The slot numbers now: num [B, N, 49] (UnitTable.battle) with the side flag (own [B, N]: the general's side), the
    come shares - men and health left / the full unit's (army_input: "men come", "hp come"; in a re-plan "what is
    left now") - and the state group (army_input STATE): morale mp / 2 within +-MORALE_CLIP, wavering or worse (ms >=
    5), the fatigue state / 5, projectiles left / the full unit's (0 without a missile weapon). u: the simulator's
    state fields ("men", "hp" share of the start's health, "hp0"; "mp", "ms", "fat", "a" - a field missing or not read
    (NaN) keeps the fresh unit's number). full [B, N, 3]: men, health and projectiles of the full unit (an older
    [B, N, 2]: the ammunition stays the fresh unit's). visible [B, N]: the enemies the side sees; a hidden enemy keeps
    the fresh unit's numbers (come shares 1, the state's fresh values: the simulator hides nobody)."""
    out = num.clone()
    men = u["men"].float()
    hp = u["hp"].float() * u["hp0"].float()
    mc = (men / full[..., 0].clamp(min=1e-6)).clamp(0, 1)
    hc = (hp / full[..., 1].clamp(min=1e-6)).clamp(0, 1)
    if visible is not None:
        seen = own | visible
        mc = torch.where(seen, mc, torch.ones_like(mc))
        hc = torch.where(seen, hc, torch.ones_like(hc))
    out[..., SIDE] = own.float()
    out[..., MEN_COME] = mc
    out[..., HP_COME] = hc
    # the state: units seen now (all without `visible`) of a slot that holds a unit (an empty slot stays zeros)
    seen = (torch.ones_like(own) if visible is None else own | visible) & (full[..., 0] > 0)
    state = {}
    if u.get("mp") is not None:
        state[MORALE] = (u["mp"].float() / 2).clamp(-ai.MORALE_CLIP, ai.MORALE_CLIP)
    if u.get("ms") is not None:
        ms = u["ms"].float()
        state[WAVERING] = torch.where(torch.isfinite(ms), (ms >= WAVERING_MS).float(), torch.full_like(ms, float("nan")))
    if u.get("fat") is not None:
        state[FATIGUE] = (u["fat"].float() / FATIGUE_MAX).clamp(0, 1)
    if u.get("a") is not None and full.shape[-1] > 2:
        a0 = full[..., 2]
        state[AMMO_COME] = torch.where(a0 > 0, (u["a"].float() / a0.clamp(min=1e-6)).clamp(0, 1), torch.zeros_like(a0))
    for col, v in state.items():
        out[..., col] = torch.where(seen & torch.isfinite(v), v, num[..., col])
    return out

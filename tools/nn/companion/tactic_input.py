"""The tactician's input in the game: the bridge's state documents -> the same raw state (tools/nn/tactic/raw.py RAW)
and the same input (tools/nn/tactic/inputs.py) the simulator's training builds (CLAUDE.md "Вход сети один в один";
build/diag_g49/bridge_parity.md, section "тактик"). Only what the game gives:

    the base columns (view.RAW)   tools/nn/companion/unit_input.py UnitSeer (game_raw: the unit level's own formulas;
                                  the velocity and running from the centre's shift - the known difference)
    the effects                   the bridge's row fx (the phases on the unit: CCO ActiveEffectList), tr (CCO
                                  IsTerrified), hid (unit:is_hidden()); the abilities of others on it (exchange.received)
    the buttons                   own units: the companion's count of the uses (exchange.ability_timers by
                                  abilities_used); an enemy: active while its card shows the phase; owned: the passport
    the passport's flags          encourages, fear, brace_ok, min_range of the key (params.static, as scenario.build)
    the map                       the captured map (build/maps/<name>: heights, ground, trunks, 'cannot stand', field)
Rule effects (charge, braced, lord near, fear near, terror immunity) - tactic/raw.py's rules on the raw state, the same
as in the simulator. The five effects the bridge does not see (raw.WAIT) are 0 here as there.

TacticSeer keeps the previous raw snapshot: one call a decision (decide_ms 1000). Numpy in, torch out.
"""
from functools import lru_cache

import numpy as np
import torch

from tools.nn.companion import exchange
from tools.nn.companion import unit_input as UI
from tools.nn.tactic import raw as TR

STATIC_MORE = TR.STATIC_MORE


@lru_cache(maxsize=512)
def _static_more(key):
    from tools.nn.sim.params import load
    P = load()
    if key not in P.units:
        return None
    s = P.static(key)
    return tuple(float(s.get(k, 0.0)) for k in STATIC_MORE)


def static_more(keys, device="cpu"):
    """{encourages, fear, brace_ok, min_range: [1, N]} of the units by key (an unknown key: zeros)."""
    out = np.zeros((len(STATIC_MORE), len(keys)), np.float32)
    for i, k in enumerate(keys):
        s = _static_more(k or "")
        if s is not None:
            out[:, i] = s
    st = {k: torch.tensor(out[j][None], device=device) for j, k in enumerate(STATIC_MORE)}
    for k in ("encourages", "fear", "brace_ok"):
        st[k] = st[k] > 0.5
    return st


@lru_cache(maxsize=512)
def arsenal_has(key):
    """The buttons a unit key owns (a bitmask over raw.ARSENAL): its passport's abilities and spells."""
    from tools.nn.model import passport
    p = passport.load().get(key) or {}
    own = set(p.get("abilities") or ()) | set(p.get("spells") or ())
    return sum(1 << a for a, k in enumerate(TR.ARSENAL) if k in own)


def observe_game(doc, battle, device="cpu", own=1):
    """The observed dict of raw.extend (raw.observe_sim's obs: fx [1, N, E], rx [1, N], a_on / a_cd [1, N, A], a_has
    [1, N]) from the bridge's state document; battle: exchange.Battle; own: the side whose bridge wrote doc."""
    names = battle.names
    N = len(names)
    rows = {u["n"]: u for u in doc["units"]}
    fx = np.zeros((1, N, TR.E), bool)
    for i, n in enumerate(names):
        row = rows.get(n) or {}
        ph = exchange._effects(row) or set()
        for name in TR.PHASE_KEY:
            fx[0, i, TR.EI[name]] = bool(TR.phases(name) & ph)
        fx[0, i, TR.EI["terror"]] = row.get("tr") is True
        fx[0, i, TR.EI["hidden"]] = row.get("hid") is True
    rx = torch.tensor(exchange.received(doc, names), device=device)
    fx = torch.tensor(fx, device=device)
    sn, htl, fo = TR.received_split(rx)
    fx[..., TR.EI["snare_net"]], fx[..., TR.EI["hold_the_line"]], fx[..., TR.EI["foreign"]] = sn, htl, fo
    on = np.zeros((1, N, TR.A), np.float32)
    cd = np.zeros((1, N, TR.A), np.float32)
    has = np.zeros((1, N), np.int64)
    tm = exchange.ability_timers(doc, names, battle.slots, own=own) if battle.slots is not None else {}
    for i, n in enumerate(names):
        key = (rows.get(n) or {}).get("key") or ""
        has[0, i] = arsenal_has(key)
        for k, ak in enumerate(battle.slots[i] if battle.slots is not None else ()):
            if ak in TR.ARSENAL:
                j = TR.ARSENAL.index(ak)
                on[0, i, j] = tm[f"ab{k}_on"][0, i]
                cd[0, i, j] = tm[f"ab{k}_cd"][0, i]
    return {"fx": fx, "rx": rx, "a_on": torch.tensor(on, device=device), "a_cd": torch.tensor(cd, device=device),
            "a_has": torch.tensor(has, device=device)}


class TacticSeer:
    """The tactician's raw state and input of a battle in the game, one call a decision (module doc)."""

    def __init__(self, map_root=UI.MAPS, device="cpu", own=1, params=None, allow_flat=False):
        from tools.nn.sim.params import load
        self.P = params or load()
        self.unit = UI.UnitSeer(map_root, device, self.P, allow_flat)
        self.device, self.own = device, own
        self.batch = self.prev = self.t = self.static = self.card = None

    def raw(self, doc):
        """[1, N, len(tactic.raw.RAW)] of this document (the battle's units in the document's order)."""
        from tools.nn.model import passport
        new = doc["batch"] != self.batch
        base = self.unit.raw(doc)
        if new:
            keys = [u.get("key") or "" for u in doc["units"]]
            self.static = static_more(keys, self.device)
            self.card = torch.from_numpy(passport.table(keys)[None]).float().to(self.device)
            self.batch, self.prev, self.t = doc["batch"], None, None
        t = float(doc.get("t", 0)) / 1000.0
        dt = (t - self.t) if self.t is not None and t > self.t else 1.0
        obs = observe_game(doc, self.unit.battle, self.device, self.own)
        r = TR.extend(base, obs, self.static, self.prev, dt, self.P)
        self.prev, self.t = r, t
        return r

    def input(self, doc, parcel, tiles, areas=None, bounds=None):
        """The tactician's input (inputs.build) of this document; the parcel in the game's coordinates (the map's)."""
        from tools.nn.tactic import inputs
        r = self.raw(doc)
        return inputs.build(r, self.card, parcel, tiles, areas, bounds)

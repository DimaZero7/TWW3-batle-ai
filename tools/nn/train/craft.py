"""Craft by unit class: where the trade is won or lost (the exam's proxy, tools/nn/train/exam_sim.py; measure "craft").

Our units fall in classes by their own numbers in the state (the same as config/nn/units.json: class com / inf_mel /
inf_mis, caste war_beast, the stalk attribute), first match wins:

    lord     the lord or another single-man character (units.json class com)
    stalk    hidden runners: the stalk attribute (the Gutter Runners)
    missile  shooters: a missile range and ammunition (class inf_mis)
    fast     beasts and other fast units: run speed FAST_RUN m/s or more, or large (the Giant Rats 9.5 m/s; infantry
             runs 2.8-5.4)
    spear    spears and halberds: a melee bonus against large (the Empire's spearmen 15, halberdiers 16, slave spears 6)
    melee    the other melee infantry (swords, greatswords, slaves, plague monks)

Per battle and class:

    killed   the enemy gold the class destroyed = kill_hp + kill_rout
             kill_hp: the gold of enemy health its units struck off - the simulator's own bookkeeping `gold_out` (HP
             dealt to an enemy / that enemy's starting HP x its cost, each attacker of each step, battle.py: melee,
             shots and bombardments; exact)
             kill_rout: the enemy's loss beyond health (reward.gold_lost: a shattered or gone unit is lost whole, a
             routing one rout_share of what it has left) laid on the enemy unit's attackers in proportion to the
             gold of health each struck off that unit. The simulator keeps no per-pair sum, so the tracker builds it:
             each step's gold_out of a unit is put on the enemy it fought or shot (its `target` before the step, else
             after) - an approximation only where a unit strikes several enemies at once (melee splash, a shot's
             spread); the health sum per attacker stays exact.
    unattributed  the enemy loss nobody struck (a unit that routed or left without a hit from us: the army's
             collapse, the lord's fall; the enemy's own friendly fire)
    lost     the gold the class lost (reward.gold_lost of its units: health, routs, shattered)
    trade    (killed - lost) / budget; summed over the classes plus unattributed / budget it is the exam's trade
    cost     the gold the class fielded; per gold fielded: killed / cost ("return"), lost / cost

Times (unit-seconds of the class's units, live battles): melee share = in melee / standing; rout share = routing /
on the field (men left, not gone); shooters: fire share = firing / standing with ammunition left, fire seconds a unit
(firing, per unit and battle), ammunition used at the end.

attribute() and summary() are plain numpy (host tests); the Tracker needs torch.
"""
import math

import numpy as np

CLASSES = ("lord", "melee", "spear", "missile", "fast", "stalk")
FAST_RUN = 6.0                 # m/s: beasts and cavalry run 8-10, infantry 2.8-5.4 (config/nn/units.json speed.run)
TIMES = ("stand_s", "melee_s", "field_s", "rout_s", "ready_s", "fire_s")
Z95 = 1.959964


def classes(u):
    """[B, N] the class index (CLASSES) of every slot (empty slots too: their numbers decide nothing, mine masks
    them). u: the state's units (torch)."""
    import torch
    lord = u["lord"].bool() | (u["men0"] <= 1)
    stalk = u["stalk"].bool() if "stalk" in u else torch.zeros_like(lord)
    missile = (u["range"] > 0) & (u["ammo0"] > 0)
    fast = (u["run"] >= FAST_RUN) | u["large"].bool()
    spear = u["bonus_v_large"] > 0
    c = torch.full_like(u["side"], CLASSES.index("melee"), dtype=torch.long)
    for name, m in (("spear", spear), ("fast", fast), ("missile", missile), ("stalk", stalk), ("lord", lord)):
        c = torch.where(m, torch.full_like(c, CLASSES.index(name)), c)         # the later the stronger
    return c


def craft_step(st, mine, live, cls, prev, tgt0, pair, sums, dt):
    """Tracker.update (in place): the step's gold of health each own unit struck off (gold_out's growth) on the enemy
    it fought or shot, and the class times."""
    import torch
    u = st.u
    run = live[:, None]
    m = mine & run
    g = u["gold_out"]
    d = (g - prev).clamp(min=0) * m.float()
    tgt = torch.where(tgt0 >= 0, tgt0, u["target"])
    ok = (tgt >= 0).float()
    pair.scatter_add_(2, tgt.clamp(min=0)[..., None], (d * ok)[..., None])
    prev.copy_(g)
    tgt0.copy_(u["target"])
    field = m & (u["side"] > 0) & (u["men"] > 0) & ~u["gone"]
    stand = field & ~u["r"]
    ready = stand & (u["range"] > 0) & (u["ammo0"] > 0) & (u["a"] > 0)
    hot = torch.nn.functional.one_hot(cls, len(CLASSES)).float()                # [B, N, C]
    for k, mask in (("stand_s", stand), ("melee_s", stand & u["m"]), ("field_s", field), ("rout_s", field & u["r"]),
                    ("ready_s", ready), ("fire_s", ready & u["fire"])):
        sums[k] += (hot * mask[..., None].float()).sum(1) * dt


class Tracker:
    """Per battle [B] the craft's sums over the learner's units (mine [B, N]) while the battle runs (a watcher like
    behaviour.Tracker: update after every step; the compact batch shrinks every [B, ...] attribute)."""

    def __init__(self, st, params, mine, wrap=None):
        import torch
        u = st.u
        B, N = u["side"].shape
        self.dt = float(params.dt)
        self.mine = mine
        self.cls = classes(u)
        self.prev = u["gold_out"].clone()
        self.tgt0 = u["target"].clone()
        self.pair = torch.zeros(B, N, N, device=st.device)
        self.sums = {k: torch.zeros(B, len(CLASSES), device=st.device) for k in TIMES}
        self._step = wrap(craft_step) if wrap else craft_step

    def update(self, st, live):
        self._step(st, self.mine, live, self.cls, self.prev, self.tgt0, self.pair, self.sums, self.dt)

    def battles(self, st, rout_share, budget):
        """-> per-battle numpy arrays at the battle's end (st: the whole batch's state): attribute()'s plus the
        times and the budget [B]."""
        import torch

        from tools.nn.train import reward
        u = st.u
        gl = reward.gold_lost(u, rout_share)
        hp_part = u["cost"] * (1 - u["hp_abs"] / u["hp0"].clamp(min=1e-6)).clamp(0, 1)
        enemy = (u["side"] > 0) & ~self.mine
        mine = self.mine & (u["side"] > 0)
        ammo = torch.where(u["ammo0"] > 0, 1 - u["a"] / u["ammo0"].clamp(min=1e-6), torch.zeros_like(u["a"]))
        npy = (lambda x: x.detach().float().cpu().numpy())
        out = attribute(npy(u["gold_out"]), npy(self.pair), npy(gl), npy(hp_part), npy(enemy) > 0, npy(mine) > 0,
                        self.cls.cpu().numpy(), npy(u["cost"]), npy(ammo),
                        missile=npy((u["range"] > 0) & (u["ammo0"] > 0)) > 0)
        out.update({k: npy(v) for k, v in self.sums.items()})
        out["budget"] = np.asarray(budget, dtype=float)
        return out


def attribute(gold_out, pair, lost, hp_part, enemy, mine, cls, cost, ammo_used=None, missile=None):
    """Per battle [B, C] the class's kill_hp, kill_rout, lost, cost, units, [B] unattributed (module doc).
    gold_out [B, N] the gold of enemy health each unit struck off; pair [B, N, N] the same by attacker i and enemy j
    (the tracker's approximation); lost [B, N] reward.gold_lost; hp_part [B, N] the gold of each unit's health lost;
    enemy, mine [B, N] bool; cls [B, N] the class; cost [B, N]; ammo_used [B, N] the share of ammunition used
    (shooters: missile [B, N] bool) -> also ammo_used_sum / ammo_units [B, C]."""
    B, N = lost.shape
    C = len(CLASSES)
    rem = np.clip(lost - hp_part, 0, None) * enemy                        # [B, N] each enemy's loss beyond health
    col = pair.sum(1)                                                     # [B, N] the gold of health struck off j
    struck = col > 1e-9
    share = np.where(struck[:, None, :], pair / np.where(struck, col, 1.0)[:, None, :], 0.0)
    rout_i = (share * rem[:, None, :]).sum(2)                             # [B, N] each attacker's share of it
    hot = np.zeros((B, N, C))
    np.put_along_axis(hot, cls[..., None].astype(int), 1.0, axis=2)
    hot *= mine[..., None]
    per = (lambda x: np.einsum("bn,bnc->bc", x, hot))
    out = {"kill_hp": per(gold_out * mine), "kill_rout": per(rout_i), "lost": per(lost), "cost": per(cost),
           "units": hot.sum(1), "destroyed": (lost * enemy).sum(1)}
    # the rest of the enemy's loss: units nobody of ours struck (rem of an unstruck one), the enemy's own fire
    out["unattributed"] = out["destroyed"] - out["kill_hp"].sum(1) - out["kill_rout"].sum(1)
    if ammo_used is not None and missile is not None:
        w = hot * missile[..., None]
        out["ammo_used_sum"] = np.einsum("bn,bnc->bc", ammo_used, w)
        out["ammo_units"] = w.sum(1)
    return out


def _mean_se(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if not n:
        return None, None
    return float(x.mean()), (float(x.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0)


def summary(b, sel=None):
    """{class: {units, cost, killed, kill_hp, kill_rout, lost (gold a battle), return (killed / cost), lost_share
    (lost / cost), trade {value, se} ((killed - lost) / budget, a battle's mean), melee_share, rout_share, and for a
    class with shooters fire_share, fire_s (a unit and battle), ammo_used}, unattributed (/ budget, mean),
    battles} of Tracker.battles' arrays b over the battles sel (numpy bool; None: all). A class nobody fielded is
    left out."""
    B = len(b["budget"])
    sel = np.ones(B, dtype=bool) if sel is None else np.asarray(sel, dtype=bool)
    bud = np.maximum(b["budget"][sel], 1e-9)
    n = int(sel.sum())
    out = {"battles": n, "classes": {}}
    if not n:
        return out
    un = _mean_se(b["unattributed"][sel] / bud)
    out["unattributed"] = round(un[0], 4)

    def ratio(num, den):
        den = float(den)
        return round(float(num) / den, 4) if den > 1e-9 else None
    for c, name in enumerate(CLASSES):
        units = b["units"][sel, c]
        if units.sum() <= 0:
            continue
        killed = b["kill_hp"][sel, c] + b["kill_rout"][sel, c]
        lost = b["lost"][sel, c]
        t, se = _mean_se((killed - lost) / bud)
        row = {"units": round(float(units.mean()), 2), "cost": round(float(b["cost"][sel, c].mean()), 1),
               "killed": round(float(killed.mean()), 1), "kill_hp": round(float(b["kill_hp"][sel, c].mean()), 1),
               "kill_rout": round(float(b["kill_rout"][sel, c].mean()), 1), "lost": round(float(lost.mean()), 1),
               "return": ratio(killed.sum(), b["cost"][sel, c].sum()),
               "lost_share": ratio(lost.sum(), b["cost"][sel, c].sum()),
               "trade": {"value": round(t, 4), "se": round(se, 4)},
               "melee_share": ratio(b["melee_s"][sel, c].sum(), b["stand_s"][sel, c].sum()),
               "rout_share": ratio(b["rout_s"][sel, c].sum(), b["field_s"][sel, c].sum())}
        shooters = float(b["ammo_units"][sel, c].sum()) if "ammo_units" in b else 0.0
        if shooters > 0:
            row["fire_share"] = ratio(b["fire_s"][sel, c].sum(), b["ready_s"][sel, c].sum())
            row["fire_s"] = round(float(b["fire_s"][sel, c].sum()) / shooters, 1)
            row["ammo_used"] = ratio(b["ammo_used_sum"][sel, c].sum(), shooters)
        out["classes"][name] = row
    return out


def lines(s):
    """The craft table of a summary s as text lines ([] without one)."""
    if not s or not s.get("classes"):
        return []
    out = [f"craft by class ({s['battles']} battles; gold a battle; trade = (killed - lost) / budget, ± 95%):",
           "  class     units   cost  killed (hp+rout)   lost  return  trade            melee  rout   fire  fire_s  ammo"]
    for name, r in s["classes"].items():
        t = r["trade"]
        f = (lambda k, fmt: "-" if r.get(k) is None else fmt.format(r[k]))
        out.append(f"  {name:8s} {r['units']:5.1f} {r['cost']:6.0f} {r['killed']:7.0f} ({r['kill_hp']:.0f}+"
                   f"{r['kill_rout']:.0f}) {r['lost']:6.0f}  {f('return', '{:.2f}'):>6s}  {t['value']:+.3f} ± "
                   f"{Z95 * t['se']:.3f}  {f('melee_share', '{:.2f}'):>5s}  {f('rout_share', '{:.2f}'):>4s}  "
                   f"{f('fire_share', '{:.2f}'):>5s}  {f('fire_s', '{:.0f}'):>6s}  {f('ammo_used', '{:.2f}'):>4s}")
    out.append(f"  unattributed (enemy loss nobody struck) / budget: {s.get('unattributed', 0):+.3f}")
    return out

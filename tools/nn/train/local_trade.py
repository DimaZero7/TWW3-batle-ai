"""The executor's local trade (run.py --exec-local-trade W; 0: off): each unit's own part of the trade over a decision,
paid to that unit alone, beside the side's advantage (docs/ru/training/general.md "Местный размен исполнителя").

The side's reward (reward.py v2) is the army's trade: a unit hardly sees how its own decision moves it among the whole
battle's noise. The local trade is the same reward cut to the unit's own skirmish, in the reward's units (a share of
the budget), per learner unit and decision:

    killed   the gold of enemy health it struck off over the decision (the simulator's `gold_out` growth: HP dealt to
             an enemy / that enemy's starting HP x its cost; craft.py) + the enemy's loss beyond health over the
             decision (a rout's share, a shattered / gone unit lost whole: reward.gold_lost less its health part) laid
             on that enemy's attackers in proportion to the gold of health each struck off it over the decision (an
             enemy nobody struck in the decision: nobody's - a collapse is the army's, the side's reward has it)
    lost     its own gold lost over the decision (reward.gold_lost's growth: health, rout, shattered, gone)
    lord     x weights.lord: the enemy lord's fall (dead; with lord_rout > 0 shattered as dead, routing lord_rout of
             it, given back on a rally - reward.parts' lord column) on its attackers as above, and the own lord's fall
             on the lord himself (his own gold lost besides)
    raw      = weights.gold x (killed - lost) / budget + lord terms

An attacker's enemy is its `target` slot at the decision's start (else at its end): an approximation only where a unit
strikes several enemies within one decision (melee splash, a volley's spread); the health part per attacker is exact.

Centring (center "class", the default): a unit that neither struck nor lost anything in the decision is "out of the
fight" and is paid 0; an engaged one is paid raw - the running mean of the raw of its class's engaged decisions
(craft.CLASSES): the class's decayed sums of the engaged raw and of
its engaged decisions (x (1 - RATE) a decision, this decision's added before the mean is read - so the first decisions
are centred too, KindNorm's way). So the term is neutral, on average, on whether to fight at all (the side's
reward and the general decide that: a melee unit's trade is mostly negative - without the centring staying out would
pay) and pays a unit for fighting better than its class usually does: killing more for what it loses. center "none":
the raw itself (the term then also prices going in or keeping out: a losing class learns to keep out).

The update (ppo.update): the paid numbers [T, R, N] -> each unit's reward-to-go at gamma x lambda within the battle and
the chunk (advantage(), as the side's GAE adds a reward to the decisions before it), / the side's advantage's spread
before its normalisation (so W = 1 weighs a unit's own gold as the side's advantage weighs the army's), x W, added to
the unit's advantage after the side's normalisation - as the mission bonus (--general-exec). The critic, the returns and
the general never see it.

advantage() is plain arithmetic (numpy and torch alike: host tests); Local needs torch.
"""
RATE = 0.01          # the classes' running mean of the engaged raw: its decayed sums x (1 - RATE) a decision (~100
#                      decisions' memory, as runtime.exec_rate)
EPS = 1e-9           # gold: less than this struck or lost is "not engaged"
CENTERS = ("class", "none")


def advantage(bonus, done, gl):
    """The local trade's part of each unit's advantage before its scale: bonus [T, R, N] a decision, done [T, R]
    (float: the battle ended at t), gl = gamma x lambda -> sum_j gl^j bonus[t + j] within the battle and the chunk
    (as the side's GAE carries a reward back; no (1 - gl), unlike the mission bonus's signal.exec_advantage)."""
    out = bonus * 0.0
    run = bonus[0] * 0.0
    for t in reversed(range(len(bonus))):
        run = bonus[t] + gl * (1.0 - done[t])[..., None] * run
        out[t] = run
    return out


def raw(st_before, st_after, weights, mine):
    """[B, N] each unit's raw local trade over a decision (module doc), in the reward's units, for the units of mine
    [B, N] (bool: the learner's side of each battle; 0 elsewhere), and engaged [B, N] (struck or lost something).
    st_before / st_after: snapshot() of the decision's start and end."""
    import torch
    a, b = st_before, st_after
    side = a["side"]
    B, N = side.shape
    d_out = (b["gold_out"] - a["gold_out"]).clamp(min=0)                       # gold of enemy health struck off
    d_lost = (b["lost"] - a["lost"]).clamp(min=0)                              # own gold lost
    d_hp = (b["hp_gold"] - a["hp_gold"]).clamp(min=0)
    rem = (d_lost - d_hp).clamp(min=0)                                         # the loss beyond health
    d_lord = (a["lord_v"] - b["lord_v"]) * a["lord"].float()                   # a lord's fall (- a rally)
    tgt = torch.where(a["target"] >= 0, a["target"], b["target"])
    hot = (tgt[..., None] == torch.arange(N, device=side.device)).float()      # [B, i, j]
    sj = side[:, None, :]
    foe = ((sj > 0) & (sj != side[..., None])).float()                         # j an enemy of i
    pair = d_out[..., None] * hot * foe                                        # [B, i, j]
    col = pair.sum(1)                                                          # [B, j] struck off j
    share = pair / col.clamp(min=EPS)[:, None, :] * (col > EPS).float()[:, None, :]
    kill_rout = (share * rem[:, None, :]).sum(2)
    kill_lord = (share * d_lord[:, None, :]).sum(2)
    budget = a["budget"][:, None]
    out = weights.gold * (d_out + kill_rout - d_lost) / budget + weights.lord * (kill_lord - d_lord)
    engaged = mine & ((d_out > EPS) | (d_lost > EPS) | (d_lord.abs() > EPS))
    return out * mine.float(), engaged


def snapshot(u, weights):
    """The numbers raw() compares of the state's units u [B, N] (copies; weights: reward.Weights, its rout_share and
    lord_rout as the reward's)."""
    import torch

    from tools.nn.train import reward
    lord_rout = float(weights.lord_rout)
    alive = (u["side"] > 0) & (u["men"] > 0) & ~u["gone"]
    if lord_rout:
        lord_v = torch.where(alive & ~u["s"], 1 - lord_rout * u["r"].float(), torch.zeros_like(u["cost"]))
    else:
        lord_v = alive.float()
    hp_gold = u["cost"] * (1 - u["hp_abs"] / u["hp0"].clamp(min=1e-6)).clamp(0, 1) * (u["side"] > 0).float()
    return {"side": u["side"].clone(), "gold_out": u["gold_out"].clone(), "target": u["target"].clone(),
            "lost": reward.gold_lost(u, weights.rout_share).clone(), "hp_gold": hp_gold, "lord_v": lord_v,
            "lord": u["lord"].bool().clone(), "budget": reward.budget(u)}


class Local:
    """The local trade of the learner's rows, decision by decision (rollout.Battles: before() ahead of a decision's
    simulator steps, after() behind them, before a finished battle restarts)."""

    def __init__(self, weights, center="class", device="cpu"):
        import torch

        from tools.nn.train import craft
        if center not in CENTERS:
            raise ValueError(f"--exec-local-center {center!r}: one of {CENTERS}")
        self.weights, self.center = weights, center
        self.C = len(craft.CLASSES)
        self.sx = torch.zeros(self.C, dtype=torch.float64, device=device)    # each class's decayed sum of the engaged raw
        self.sn = torch.zeros(self.C, dtype=torch.float64, device=device)    # ... and of its engaged decisions
        self.snap = None
        self.zero()

    @property
    def base(self):
        """[C] each class's running mean of the engaged raw (0 before its first engaged decision)."""
        return self.sx / self.sn.clamp(min=1e-9)

    def zero(self):
        self.sums = {"n": 0.0, "engaged": 0.0, "raw": 0.0, "raw2": 0.0, "paid": 0.0, "paid2": 0.0, "pos": 0.0}
        self.by_class = [[0.0, 0.0] for _ in range(self.C)]              # engaged raw sum, engaged decisions

    def before(self, st):
        self.snap = snapshot(st.u, self.weights)

    def after(self, st, rows, B, acting):
        """-> the bonus [R, N] of the learner rows `rows` (row = side x B + battle) for their units that decided
        (acting [R, N]), in the reward's units (module doc)."""
        import torch

        from tools.nn.train import craft
        now = snapshot(st.u, self.weights)
        rb, rs = rows % B, rows // B + 1
        r_all, eng_all = raw(self.snap, now, self.weights, now["side"] > 0)
        self.snap = None
        x = r_all[rb]                                                           # [R, N]
        mine = (now["side"][rb] == rs[:, None]) & acting
        eng = eng_all[rb] & mine
        x = x * mine.float()
        cls = craft.classes(st.u)[rb]
        if self.center == "class":
            hot = (cls[..., None] == torch.arange(self.C, device=x.device)) & eng[..., None]   # [R, N, C]
            hf = hot.float()
            tot = torch.stack([(hf * x[..., None]).sum((0, 1)), hf.sum((0, 1))]).double()      # [2, C]
            self.sx = self.sx * (1 - RATE) + tot[0]
            self.sn = self.sn * (1 - RATE) + tot[1]
            paid = (x - self.base.to(x.dtype)[cls]) * eng.float()
            for c, (sm, n) in enumerate(zip(*tot.double().cpu().tolist())):
                self.by_class[c][0] += sm
                self.by_class[c][1] += n
        else:
            paid = x
        md, ed = mine.double(), eng.double()
        self.sums["n"] += float(md.sum())
        self.sums["engaged"] += float(ed.sum())
        self.sums["raw"] += float((x.double() * ed).sum())
        self.sums["raw2"] += float((x.double() ** 2 * ed).sum())
        self.sums["paid"] += float((paid.double() * ed).sum())
        self.sums["paid2"] += float((paid.double() ** 2 * ed).sum())
        self.sums["pos"] += float(((x > 0) & eng).double().sum())
        return paid

    def take_stats(self):
        """The log's numbers since the last call: the engaged share of the deciding units, the raw local trade of an
        engaged unit-decision (mean, sd, share above 0), the paid one (mean, sd), each class's engaged mean raw and
        its running base."""
        from tools.nn.train import craft
        s = self.sums
        e = max(1.0, s["engaged"])

        def sd(a, a2):
            m = a / e
            return max(0.0, a2 / e - m * m) ** 0.5
        out = {"engaged": round(s["engaged"] / max(1.0, s["n"]), 4), "raw_mean": round(s["raw"] / e, 6),
               "raw_sd": round(sd(s["raw"], s["raw2"]), 6), "raw_pos": round(s["pos"] / e, 4),
               "paid_mean": round(s["paid"] / e, 6), "paid_sd": round(sd(s["paid"], s["paid2"]), 6),
               "center": self.center}
        if self.center == "class":
            base = self.base.cpu().tolist()
            out["classes"] = {craft.CLASSES[c]: {"mean": round(sm / n, 6), "base": round(base[c], 6), "n": int(n)}
                              for c, (sm, n) in enumerate(self.by_class) if n > 0}
        self.zero()
        return out

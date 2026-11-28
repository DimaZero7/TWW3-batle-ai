"""The tactic level's one side wrapper (tmp/tactic_slice/design_attack_v23.md section 6 item 6): the simulator's
stage0/battle.TacticSide (its adapter to the simulator's state) and the bridge's companion run the level through it the
same way - one group = my units but the lord, the lord's raw row hidden (not alive: no group, no tile; he holds), the
zone from the spec or tactic/zfit.zone_for, a single lower.Tacticians.apply and shape at the first tick, then every
decide_s (5 s) Tacticians.tick(raw, card, bounds, now) under a player of tactic/pin.py (TEACH, C, pin.NetPlayer, ...).

    Side(player, hygiene=True, device, lord_player=None, lord_labeller=None)   lord_player: our lord commanded by the
              level too (tactic/lordtac.py; the first hidden row of each battle is his): the tick keeps his raw row
              unhidden and hands it down (lower.Tacticians lord_raw), pass 1 still gets the raw with him hidden; None -
              the level as before
    start(x, z, b, mine, hide, zones=None)   [B, N]: the slots' places and facings at the start, in the raw state's frame
              (the simulator's battle frame; the game's map coordinates), mine - my units but the lord, hide - the rows
              hidden from the level (my lord); zones [B] the spec's {cx, cz, heading, line, rows, cols, worst} or None
              (zfit.zone_for on the slot order, the course the first unit's facing; config orders.zone_fit off or no
              zone found - a line 1 x n on the units' middle)
    tick(raw, card, bounds, now, sim=None)   the level's tick (raw [B, N, len(raw.RAW)], card [B, N, 69] passport.table,
              bounds - a flat battle's half size (None: none), now - the battle's second; sim the simulator's state, a
              tagger's - None on the bridge)
    of_raw(raw, card, own=1)  (x, z, b, mine, hide) of a raw state at the battle's start: my units the alive rows of side
              own, the lord by the passport's caste (the card's column of rows' LORD_COL) - the companion's start
After tick: tac (lower.Tacticians: orders_of(), areas(), view, counts()), worst [B] (the zone's worst first move, m).
"""
import torch

from net.tactic import cfg
from net.tactic import orders as OR
from net.tactic import raw as TR
from net.tactic import rows as RW

LORD_CARD = int(RW.ROW[OR.LORD_COL][1:])       # (the passport card's column of the caste 'lord': rows' p36)


def of_raw(raw, card, own=1):
    """(module doc)."""
    I = TR.I
    alive = raw[..., I["alive"]] > 0
    ours = (raw[..., I["side"]].round().long() == int(own))
    lord = card[..., LORD_CARD] > 0.5
    return raw[..., I["x"]], raw[..., I["z"]], raw[..., I["b"]], ours & alive & ~lord, ours & lord


def zone_of(places, course, spec_zone=None):
    """{cx, cz, heading, line, rows, cols, worst} of one battle's group (module doc: start)."""
    from net.tactic import zfit
    fit = bool(cfg()["orders"]["zone_fit"])
    if spec_zone is not None and fit:
        return spec_zone
    zn = zfit.zone_for(places, course) if (fit and places) else None
    if zn is None:
        n = max(len(places), 1)
        mx = sum(p[0] for p in places) / n if places else 0.0
        mz = sum(p[1] for p in places) / n if places else 0.0
        zn = {"cx": mx, "cz": mz, "heading": course, "line": True, "rows": 1, "cols": n, "worst": float("nan")}
    return zn


class Side:
    """(module doc)"""

    def __init__(self, player, hygiene=True, device="cpu", lord_player=None, lord_labeller=None):
        self.player, self.hygiene, self.device = player, hygiene, device
        self.lord_player, self.lord_labeller = lord_player, lord_labeller
        self.tac = None

    def start(self, x, z, b, mine, hide, zones=None):
        from net.operative import groups as GR
        from net.operative import lower as LW
        from net.tactic import decide as DC
        B, N = mine.shape
        dev = self.device
        self.mine, self.hide = mine.to(dev), hide.to(dev)
        self.tac = LW.Tacticians(B, N, 1, dev, player=self.player, hygiene=self.hygiene, lord_player=self.lord_player,
                                 lord_labeller=self.lord_labeller)
        if self.lord_player is not None:
            self.tac.lord_start(torch.where(self.hide.any(1), self.hide.float().argmax(1).long(),
                                            torch.full((B,), -1, dtype=torch.long, device=dev)))
        g = GR.Groups.empty(B, N, 1, dev)
        g.member = torch.where(self.mine, torch.zeros_like(g.member), g.member)
        g.on[:, 0] = self.mine.any(1)
        g.n0[:, 0] = self.mine.sum(1)
        g.verb[:, 0] = DC.VERBS.index("hold")
        rows, cols = [], []
        self.worst = []
        x, z, b = x.cpu(), z.cpu(), b.cpu()
        for k in range(B):
            sl = self.mine[k].nonzero()[:, 0].tolist()
            pl = [(float(x[k, i]), float(z[k, i])) for i in sl]
            course = float(b[k, sl[0]]) if sl else 0.0
            zn = zone_of(pl, course, None if zones is None else zones[k])
            g.cx[k, 0], g.cz[k, 0], g.heading[k, 0], g.line[k, 0] = zn["cx"], zn["cz"], zn["heading"], bool(zn["line"])
            rows.append(zn["rows"])
            cols.append(zn["cols"])
            self.worst.append(zn["worst"])
        self.groups0 = g
        self.shape0 = (torch.tensor(rows), torch.tensor(cols))
        self.applied = False
        return self

    def hidden(self, raw):
        """raw with the hidden rows not alive (module doc)."""
        r = raw.clone()
        a = TR.I["alive"]
        r[..., a] = torch.where(self.hide, torch.zeros_like(r[..., a]), r[..., a])
        return r

    def tick(self, raw, card, bounds=None, now=None, sim=None):
        full = raw if self.lord_player is not None else None
        raw = self.hidden(raw)
        if not self.applied:
            self.applied = True
            on = self.groups0.on
            self.tac.apply(self.groups0, on, torch.zeros_like(on), raw)
            self.tac.shape(*self.shape0)
        self.tac.tick(raw, card, bounds, now=now, sim=sim, lord_raw=full)
        return self.tac

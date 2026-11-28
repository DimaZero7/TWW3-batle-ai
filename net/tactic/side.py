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

Group mode (Side(group=True); stage0.json lord_script 'group'): the groups from THE builder groups.stage0 - the infantry
one group exactly as before, our lord (hide) a group of one ANCHORED to it, both rows of one Tacticians (G 2) under the
SAME player and order store; no lord player, no lord store. The infantry's rows tick on the raw with the lord hidden
(their inputs as before), then the lord's row on the raw with him alive (the infantry his own other units). His group's
parcel by code every tick (parcel()), the default of the level above until an operative commands it:
    zone      his post (the executor's post rule, net/unit/lord.T2._post / lord_script._post - posts(x, z, heading, ok)
              given by the caller after each executor second) - before one, where he stands, the anchor's heading
    mask      empty until the anchor group's first melee contact (a member's m on a tick); from then on the anchor's
              opponents: every seen, alive, not routing enemy within scripts.CHARGE_M edge gap of a standing member
              (fighting it or facing it at a charge's reach), never the enemy lord - with config stage0.json
              lord_group.enemy_lord on also the enemy lord on the same leash (CHARGE_M of a standing member), not
              while our lord's hp < lord_script.STEP_OUT_HP; empty while he steps out (busy). The mask is the one
              place of his legality: the pointer's keys (orders.Facts bind), the order's end (ended_off_mask); the
              side channel's lordtac (CANCEL, legal) never runs in this mode. Empty: the level can
              only keep him
    verb      hold (as the infantry's: T0-1 never saw another command)
lord_cmd() [B] his target in the common store (-1 none), lord_drop(mask, code) the executor's drop cleared there.
His row's player is his group's own (group_player, else group_copy(player): its own driver state and recorder - a
tags.Record's labels and hook rows carry View.grp 1 - the same network weights) and draws from its own random stream.
The caller each executor second, before his decision: second(raw, card) - the contact latched, a command off legal()
dropped ('off_mask'); at a tick, before it: posts() of this second's post (the executor's peek_post).
"""
import contextlib
import types

import torch

from net.tactic import cfg
from net.tactic import orders as OR
from net.tactic import raw as TR
from net.tactic import rows as RW

LORD_CARD = int(RW.ROW[OR.LORD_COL][1:])       # (the passport card's column of the caste 'lord': rows' p36)
GROUP_SEED = 0x10D                             # (group mode: the lord's group's own random stream starts here)
DROPS = ("gone", "step_out", "off_mask")       # (group mode: the lord's command drops - executor's, second()'s)


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


def _inner(p):
    """The players a wrapper holds (tags.Record main / lab, a forwarding wrapper's _drv, imitate.Disturb t,
    pin.SideTeacher base)."""
    d = getattr(p, "__dict__", {})
    return [d[k] for k in ("main", "lab", "_drv", "t", "base") if k in d and d[k] is not None]


def _refuse_side_channel(p, depth=0):
    """(the side channel's network - TacticNet4 / its LordHead, kind tactic_v4 - is not for the group's row)"""
    try:
        net = getattr(p, "net", None)
    except Exception:                                                     # noqa: BLE001 (a forwarding wrapper)
        net = None
    if type(net).__name__ == "TacticNet4" or hasattr(net, "lord_head") or hasattr(net, "lord"):
        raise ValueError("group mode: a tactic_v4 / LordHead network is the side channel's - the lord's group row runs "
                         "the infantry's own pointer (TacticNet2 / 3)")
    if depth < 4:
        for q in _inner(p):
            _refuse_side_channel(q, depth + 1)


def _shared(p, memo, depth=0):
    """memo for copy.deepcopy: every torch module the player holds kept as is (the same network weights)."""
    if isinstance(p, torch.nn.Module):
        memo[id(p)] = p
        return
    if depth > 4:
        return
    for v in getattr(p, "__dict__", {}).values():
        if isinstance(v, torch.nn.Module):
            memo[id(v)] = v
        elif hasattr(v, "__dict__") and not isinstance(v, (type, types.FunctionType, types.MethodType)):
            _shared(v, memo, depth + 1)


def group_copy(p):
    """A player for another group of the same level: its own driver state (memories keyed by the battle - pin.Teacher
    prev_pin, Attack1's, a recorder's counts and labels), the same network weights. tags.Record -> a new Record around
    copies of its driver and labeller with the same hook (the buffer gets the group's rows, View.grp names them); a
    forwarding wrapper (its _drv) -> a copy of what it wraps; a player with group_copy() -> its own; else a deep copy
    sharing the torch modules."""
    import copy
    from net.tactic import tags as TG
    if isinstance(p, TG.Record):
        r = TG.Record(group_copy(p.main), group_copy(p.lab), keep_raw=p.keep_raw, hook=p.hook)
        r.needs_input = p.needs_input
        return r
    d = getattr(p, "__dict__", {})
    if "_drv" in d:
        return group_copy(d["_drv"])
    if hasattr(type(p), "group_copy"):
        return p.group_copy()
    memo = {}
    _shared(p, memo)
    return copy.deepcopy(p, memo)


class Side:
    """(module doc)"""

    def __init__(self, player, hygiene=True, device="cpu", lord_player=None, lord_labeller=None, group=False,
                 group_player=None):
        assert not (group and lord_player is not None), "group mode: the lord is a row of the level, no lord player"
        self.player, self.hygiene, self.device = player, hygiene, device
        self.lord_player, self.lord_labeller = lord_player, lord_labeller
        self.group = bool(group)
        # (group mode: the lord's group's own player - its own driver state and recorder, the same network weights;
        # None: group_copy(player). Its sampling draws from its own random stream: _own_rng)
        self.gplayer = None
        if group:
            _refuse_side_channel(player)
            self.gplayer = group_copy(player) if group_player is None else group_player
            _refuse_side_channel(self.gplayer)
            self.g_rng = None
        # (group mode: the enemy lord inside his group's mask - config stage0.json lord_group.enemy_lord; off as before)
        from net.stage0 import cfg as scfg
        self.enemy_lord = bool((scfg().get("lord_group") or {}).get("enemy_lord", False))
        self.tac = None

    def start(self, x, z, b, mine, hide, zones=None):
        from net.operative import groups as GR
        from net.operative import lower as LW
        from net.tactic import decide as DC
        B, N = mine.shape
        dev = self.device
        self.mine, self.hide = mine.to(dev), hide.to(dev)
        g = GR.stage0(self.mine, self.hide if self.group else None, DC.VERBS.index("hold"))
        G = g.G
        self.tac = LW.Tacticians(B, N, G, dev, player=self.player, hygiene=self.hygiene, lord_player=self.lord_player,
                                 lord_labeller=self.lord_labeller)
        # (group mode: the players' View.idx is the battle - the infantry's memories keyed as with one group; View.row
        # the level's row, View.grp the group)
        self.tac.by_battle = self.group
        if self.lord_player is not None:
            self.tac.lord_start(torch.where(self.hide.any(1), self.hide.float().argmax(1).long(),
                                            torch.full((B,), -1, dtype=torch.long, device=dev)))
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
        rows, cols = torch.tensor(rows), torch.tensor(cols)
        if self.group:
            # (the lord's group: one tile; his zone where he stands, the anchor's heading, until his first post)
            ls = self.lord_slot()
            has = ls >= 0
            k = torch.arange(B, device=dev)
            g.cx[:, 1] = torch.where(has, x.to(dev)[k, ls.clamp(min=0)], g.cx[:, 0])
            g.cz[:, 1] = torch.where(has, z.to(dev)[k, ls.clamp(min=0)], g.cz[:, 0])
            g.heading[:, 1] = g.heading[:, 0]
            g.line[:, 1] = True
            rows = torch.stack([rows, torch.ones_like(rows)], 1).reshape(-1)
            cols = torch.stack([cols, torch.ones_like(cols)], 1).reshape(-1)
            self.contact = torch.zeros(B, dtype=torch.bool, device=dev)
            self.busy = torch.zeros(B, dtype=torch.bool, device=dev)
            self.post = None
            # (every chunk carries every drop count: the aggregation over chunks reads each name in each)
            for n in DROPS:
                self.tac.lcnt["lord_drop_" + n] = torch.zeros(B, device=dev)
        self.groups0 = g
        self.shape0 = (rows, cols)
        self.applied = False
        return self

    # -- group mode (module doc)
    def lord_slot(self):
        """[B] each battle's lord slot (the hidden row; -1 none)."""
        B = self.hide.shape[0]
        return torch.where(self.hide.any(1), self.hide.float().argmax(1).long(),
                           torch.full((B,), -1, dtype=torch.long, device=self.hide.device))

    def _rows(self, g):
        """[R] bool: the level's rows of group g."""
        return self.tac.gi == g

    def lord_cmd(self):
        """[B] the lord's target in the common order store (his group's row at his slot; -1 none)."""
        B, G = self.tac.B, self.tac.G
        ls = self.lord_slot()
        tg = self.tac.tgt.view(B, G, -1)[:, 1]
        on = (self.tac.mine & self.tac.active[:, None]).view(B, G, -1)[:, 1]
        k = torch.arange(B, device=ls.device)
        v = tg[k, ls.clamp(min=0)]
        return torch.where((ls >= 0) & on[k, ls.clamp(min=0)], v, torch.full_like(v, -1))

    def lord_drop(self, mask, code=None):
        """The executor dropped his command where mask [B]: cleared in the common store on the same second."""
        B, G = self.tac.B, self.tac.G
        ls = self.lord_slot()
        m = mask.to(ls.device) & (ls >= 0) & (self.lord_cmd() >= 0)
        if not bool(m.any()):
            return
        row = torch.arange(B, device=ls.device) * G + 1
        r, c = row[m], ls[m]
        self.tac.tgt[r, c] = -1
        self.tac.skill[r, c] = 0
        name = "lord_drop" + ("" if code is None else "_" + str(code))
        self.tac._lcount(name, torch.arange(B, device=ls.device), m)

    def posts(self, x, z, heading, ok):
        """The executor's post of this second [B] (ok: he has one) - his group's zone at the next tick."""
        self.post = (x, z, heading, ok)

    def legal(self, raw, card):
        """[B, N] the lord's legal targets on this raw (module doc 'mask') - the one predicate of the parcel (every
        tick) and of second() (every executor second); latches the anchor's contact (a standing member's m on). raw:
        the tactic raw or the unit level's (its first columns are the same: tactic/raw.RAW = view.RAW + MORE)."""
        from net.unit import scripts as SC
        from sim import geometry
        I = TR.I
        dev = self.device
        B, G = self.tac.B, self.tac.G
        c = lambda k: raw[..., I[k]].float()                                              # noqa: E731
        alive = c("alive") > 0
        stand = alive & ~(c("r") > 0) & ~(c("s") > 0)
        side = raw[..., I["side"]].round().long()
        anchor = self.tac.mine.view(B, G, -1)[:, 0] & alive                              # (the anchor's members)
        self.contact = self.contact | (anchor & stand & (c("m") > 0)).any(1)
        ls = self.lord_slot()
        k = torch.arange(B, device=dev)
        own = torch.where(ls >= 0, side[k, ls.clamp(min=0)], torch.ones_like(ls))
        # (a foe: seen, alive, neither routing nor shattered - the executor's own target rule, lord._take_cmd)
        foe = (side != own[:, None]) & (side > 0) & (c("vis") > 0) & stand
        gap = geometry.rect_gap(c("x"), c("z"), c("b"), c("front"), c("depth"))          # [B, i, j]
        near = torch.where((anchor & stand)[:, :, None], gap, torch.full_like(gap, float("inf"))).amin(1)  # [B, j]
        elord = foe & (card[..., LORD_CARD] > 0.5)        # (the enemy lord by the passport's caste, as of_raw)
        mask = foe & (near <= float(SC.CHARGE_M)) & ~elord
        if self.enemy_lord:
            # (the switch: the enemy group's lord a legal target too - on the same leash as every other target,
            # scripts.CHARGE_M from a standing member of the anchor; never while our lord's health is under the
            # step-out's bridge/companion/lord_script.STEP_OUT_HP)
            from bridge.companion import lord_script as LS
            hp_ok = (ls >= 0) & (c("hp")[k, ls.clamp(min=0)] >= float(LS.STEP_OUT_HP))
            mask = mask | (elord & (near <= float(SC.CHARGE_M)) & hp_ok[:, None])
        # (the anchor dead or routing to the last unit: no other line group at stage 0 - he holds his place: an empty
        # mask, his zone kept; the executor's 'alone' branch holds him)
        self.anchor_ok = (anchor & stand).any(1)
        return mask & self.contact[:, None] & ~self.busy[:, None] & self.anchor_ok[:, None]

    def second(self, raw, card):
        """Every executor second, before his decision: the contact latched (a contact wholly between two ticks opens
        the parcel at the next one) and his command in force checked against legal() on this second's raw - a target
        that left it (the leash, shattered, ...) is dropped from the common store now ('off_mask'), not at the next
        tick."""
        m = self.legal(raw, card)
        cmd = self.lord_cmd()
        k = torch.arange(cmd.shape[0], device=cmd.device)
        off = (cmd >= 0) & ~m[k, cmd.clamp(min=0)]
        self.lord_drop(off, "off_mask")

    def parcel(self, raw, card):
        """The lord's group's parcel by code (module doc) -> Groups with only group 1 changed."""
        I = TR.I
        dev = self.device
        g = self.groups0.clone()
        B, G = self.tac.B, self.tac.G
        mask = self.legal(raw, card)
        alive = raw[..., I["alive"]] > 0
        ls = self.lord_slot()
        k = torch.arange(B, device=dev)
        g.target[:, 1] = mask
        cx, cz, h = self.tac.cx.view(B, G)[:, 1], self.tac.cz.view(B, G)[:, 1], self.tac.heading.view(B, G)[:, 0]
        if self.post is not None:
            px, pz, ph, ok = (t.to(dev) for t in self.post)
            cx = torch.where(ok, px.to(cx.dtype), cx)
            cz = torch.where(ok, pz.to(cz.dtype), cz)
            h = torch.where(ok, ph.to(h.dtype), h)
        g.cx[:, 1], g.cz[:, 1], g.heading[:, 1] = cx, cz, h
        g.on[:, 1] = g.on[:, 1] & (ls >= 0) & alive[k, ls.clamp(min=0)]
        self.mask = mask
        return g

    def hidden(self, raw):
        """raw with the hidden rows not alive (module doc)."""
        r = raw.clone()
        a = TR.I["alive"]
        r[..., a] = torch.where(self.hide, torch.zeros_like(r[..., a]), r[..., a])
        return r

    def tick(self, raw, card, bounds=None, now=None, sim=None):
        if self.group:
            return self._tick_group(raw, card, bounds, now, sim)
        full = raw if self.lord_player is not None else None
        raw = self.hidden(raw)
        if not self.applied:
            self.applied = True
            on = self.groups0.on
            self.tac.apply(self.groups0, on, torch.zeros_like(on), raw)
            self.tac.shape(*self.shape0)
        self.tac.tick(raw, card, bounds, now=now, sim=sim, lord_raw=full)
        return self.tac

    def _tick_group(self, raw, card, bounds, now, sim):
        """Group mode's tick (module doc): the lord's parcel, the infantry's rows on the hidden raw, his on the full."""
        hid = self.hidden(raw)
        if not self.applied:
            self.applied = True
            on = self.groups0.on
            self.tac.apply(self.groups0, on, torch.zeros_like(on), raw)      # (the full raw: he is his group's member)
            self.tac.shape(*self.shape0)
            self.tac.lord_ok = self._rows(1) & self.enemy_lord
        g = self.parcel(raw, card)
        task = torch.zeros_like(g.on)
        task[:, 1] = True
        self.tac.apply(g, task, task.clone(), raw)                            # (moved: his tiles and order stay)
        now = self.tac.clock if now is None else float(now)                  # (one second for both passes)
        self.tac.tick(hid, card, bounds, now=now, sim=sim, rows=self._rows(0))
        # his row: his group's own player (self.gplayer: its own driver state and recorder, the same network) - the
        # infantry's player, its records, labels, memories and random stream never see his row
        self.tac.view_lord = None
        self.tac.player = self.gplayer
        try:
            with self._own_rng():
                self.tac.tick(raw, card, bounds, now=now, sim=sim, rows=self._rows(1))
        finally:
            self.tac.player = self.player
        return self.tac

    @contextlib.contextmanager
    def _own_rng(self):
        """The lord's group's pass draws from its own torch random stream (a sampled pointer, a sampled null): the
        global stream - the infantry's, the battle's - is left as it was before the pass."""
        dev = torch.device(self.device)
        devs = [dev] if dev.type == "cuda" else []
        with torch.random.fork_rng(devices=devs):
            if self.g_rng is None:
                torch.default_generator.manual_seed(GROUP_SEED)
                if devs:
                    torch.cuda.manual_seed(GROUP_SEED)
            else:
                torch.set_rng_state(self.g_rng[0])
                if devs:
                    torch.cuda.set_rng_state(self.g_rng[1], dev)
            yield
            self.g_rng = (torch.get_rng_state(), torch.cuda.get_rng_state(dev) if devs else None)

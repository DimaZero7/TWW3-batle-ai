"""The operative's groups between decisions (plan.md "Прошлые группы", "Посылки тактикам"): B battles x G groups.

    member [B, N] long   each slot's group (-1: none; enemies always -1)
    on     [B, G] bool   the group has a task (a parcel at its tactician)
    target [B, G, N]     the group's target mask (the enemy's slots)
    seed   [B, G] long   the target's seed slot (-1: no target)
    cx, cz, heading [B, G]   the zone's centre and turn (degrees, the simulator's bearings)
    line   [B, G] bool   the form "line" (else "square")
    verb, pace [B, G] long
    n0     [B, G] long   the tiles' count of the zone (the members when its tactician started: its zone keeps the size)
    age    [B, G]        seconds since the group got its members

drop_dead(): a dead unit leaves its group (free, not chosen: the parcel's slots must be alive); a group with no member
left goes off. A routing unit stays (the tactician's rule). prev(): the last groups as numbers per row (the input).
"""
from dataclasses import dataclass, fields

import torch

from net.operative import cfg
from net.tactic import raw as TR
from net.tactic import zone as ZN

KEEP, MOVE, REFORM = 0, 1, 2           # the 'keep / move / re-form' head's choices
GMAX = int(cfg()["groups"])            # the most groups (config groups; a Groups of fewer - the strategist's - pads)
PREV = 2 * (GMAX + 1)                  # per row: my unit's group GMAX + 'none', an enemy's target-of GMAX + 'nobody'
NUMBERS = 10                           # per group: on, centre right / forward, sin, cos, line, square, age, n, n0


@dataclass
class Groups:
    member: torch.Tensor
    on: torch.Tensor
    target: torch.Tensor
    seed: torch.Tensor
    cx: torch.Tensor
    cz: torch.Tensor
    heading: torch.Tensor
    line: torch.Tensor
    verb: torch.Tensor
    pace: torch.Tensor
    n0: torch.Tensor
    age: torch.Tensor

    @staticmethod
    def empty(B, N, G=None, device="cpu"):
        G = int(G or cfg()["groups"])
        z = lambda dt=torch.float32: torch.zeros((B, G), dtype=dt, device=device)        # noqa: E731
        return Groups(torch.full((B, N), -1, dtype=torch.long, device=device), z(torch.bool),
                      torch.zeros((B, G, N), dtype=torch.bool, device=device),
                      torch.full((B, G), -1, dtype=torch.long, device=device), z(), z(), z(), z(torch.bool),
                      z(torch.long), z(torch.long), z(torch.long), z())

    @property
    def G(self):
        return self.on.shape[1]

    def padded(self, G):
        """The same groups in G places (G >= self.G: the places past self.G empty) - a Groups of the strategist's 6
        handed to code sized by config groups."""
        if G == self.G:
            return self
        e = Groups.empty(self.member.shape[0], self.member.shape[1], G, self.member.device)
        for f in fields(self):
            if f.name == "member":
                continue
            v, w = getattr(self, f.name), getattr(e, f.name)
            w[:, :self.G] = v
        e.member = self.member.clone()
        return e

    def clone(self):
        return Groups(*(getattr(self, f.name).clone() for f in fields(self)))

    def take(self, k):
        return Groups(*(getattr(self, f.name)[k] for f in fields(self)))

    @staticmethod
    def stack(gs):
        return Groups(*(torch.stack([getattr(g, f.name) for g in gs]) for f in fields(Groups)))

    def members(self):
        """[B, G, N] bool."""
        G = self.G
        return self.member[:, None, :] == torch.arange(G, device=self.member.device)[None, :, None]

    def drop_dead(self, raw):
        """In place (module doc); returns the groups that went off [B, G]."""
        alive = raw[..., TR.I["alive"]] > 0
        self.member = torch.where(alive, self.member, torch.full_like(self.member, -1))
        has = self.members().any(2)
        gone = self.on & ~has
        self.on = self.on & has
        self.target = self.target & alive[:, None, :]
        return gone

    def counts(self, raw=None):
        """[B, G] the members (alive ones: raw given)."""
        m = self.members()
        if raw is not None:
            m = m & (raw[..., TR.I["alive"]] > 0)[:, None, :]
        return m.sum(2)

    def prev(self, slot, own_rows, enemy_rows):
        """[B, K, PREV] of the rows slot [B, K] (-1 empty): own rows - one-hot of the slot's group (G) + 'none';
        enemy rows - which groups target it (G) + 'nobody'."""
        B, K = slot.shape
        G = self.G
        s0 = slot.clamp(min=0)
        grp = self.member.gather(1, s0)
        own = torch.zeros((B, K, GMAX + 1), device=slot.device)
        own.scatter_(2, torch.where(grp >= 0, grp, torch.full_like(grp, GMAX))[..., None], 1.0)
        tg = self.target.transpose(1, 2).gather(1, s0[..., None].expand(B, K, G)).float()       # [B, K, G]
        tg = tg * self.on[:, None, :].float()
        tg = torch.cat([tg, torch.zeros((B, K, GMAX - G), device=slot.device)], 2)              # (fewer groups: padded)
        foe = torch.cat([tg, (tg.sum(2, keepdim=True) == 0).float()], 2)
        out = torch.cat([own * own_rows[..., None].float(), foe * enemy_rows[..., None].float()], 2)
        return out

    def numbers(self, axes, raw):
        """[B, G, NUMBERS] (module doc): the centre in the field's axes / place_m, the turn against "up"."""
        c = cfg()
        s, f = axes.local(self.cx, self.cz)
        rel = (self.heading - axes.heading[:, None]) * TR.DEG
        n = self.counts(raw).float()
        on = self.on.float()
        k = float(c["place_m"])
        x = torch.stack([on, s / k, f / k, torch.sin(rel), torch.cos(rel), self.line.float(), (~self.line).float(),
                         (self.age / 60.0).clamp(max=5.0), n / 10.0, self.n0.float() / 10.0], -1)
        return x * on[..., None]

    def zones(self):
        """{x, z, b, front, depth, on} [B, G]: the zones as turned rectangles (the tiles' full grid of n0)."""
        rows, cols = grid(self.n0, self.line)
        t = float(ZN.cfg()["tile_m"])
        return {"x": self.cx, "z": self.cz, "b": self.heading, "front": cols.float() * t, "depth": rows.float() * t,
                "on": self.on}


def grid(n, line):
    """(rows, cols) long [...] of zone.grid, batched: line 1 x n; square cols = ceil(sqrt n), rows = ceil(n / cols)."""
    n = n.long().clamp(min=1)
    cols_sq = torch.ceil(torch.sqrt(n.float()) - 1e-6).long().clamp(min=1)
    rows_sq = torch.div(n + cols_sq - 1, cols_sq, rounding_mode="floor")
    return torch.where(line, torch.ones_like(n), rows_sq), torch.where(line, n, cols_sq)

"""The cases' readings and their end (plan.md "Случаи", "Награда"):

    Readings    per clump (the units that started together; at most 2) the tactician's contact reading drill1.Watch -
                the first contact, 'the infantry between' (the shooters untouched and every touching enemy's line to
                them crosses my infantry), 'hit' (the shooters touched at once); and over the battle: gathered (the
                share of my alive units within gather_m of the first contact's point: my units touching then), advanced
                (my centre went farther than advance_m towards the enemy from the start - the go case's 'hold')
    Ender       the end after every simulator step: a side has nothing standing; fight_s after the contact of every
                clump; the battle's limit
"""
import torch

from tools.nn.operative import cfg
from tools.nn.tactic import drill1 as DR

CLUMPS = 2


class Readings:
    def __init__(self, setup, spacing):
        st = setup.st
        dev = setup.device
        B = st.B
        self.setup = setup
        self.watch = []
        self.has = []
        for k in range(CLUMPS):
            own = setup.clump == k
            self.watch.append(DR.Watch(st, own, setup.shooter & own, spacing, dev))
            self.has.append(own.any(1))
        z = lambda: torch.zeros(B, dtype=torch.bool, device=dev)                  # noqa: E731
        self.gathered = torch.full((B,), float("nan"), device=dev)
        self.advanced = z()
        u = st.u
        own = setup.own.float()
        k = own.sum(1).clamp(min=1)
        self.x0, self.z0 = (u["x"] * own).sum(1) / k, (u["z"] * own).sum(1) / k
        foe = (~setup.own & (u["side"] > 0)).float()
        kf = foe.sum(1).clamp(min=1)
        ex, ez = (u["x"] * foe).sum(1) / kf - self.x0, (u["z"] * foe).sum(1) / kf - self.z0
        d = torch.sqrt(ex * ex + ez * ez).clamp(min=1e-6)
        self.ux, self.uz = ex / d, ez / d
        r = cfg()["reward"]
        self.gather_m, self.advance_m = float(r["gather_m"]), float(r["advance_m"])
        # each clump's centre at the start and at its first contact (the armies' split: did it stay where it stood)
        self.c0 = torch.zeros((B, CLUMPS, 2), device=dev)
        self.cc = torch.full((B, CLUMPS, 2), float("nan"), device=dev)
        for k in range(CLUMPS):
            m = (setup.clump == k).float()
            n = m.sum(1).clamp(min=1)
            self.c0[:, k, 0], self.c0[:, k, 1] = (u["x"] * m).sum(1) / n, (u["z"] * m).sum(1) / n

    def step(self, st, was, dt, area=None):
        """(between [B] float: the share of the clumps that got 'between' now, hit [B] float the same of 'hit', gathered
        now [B] float (the share; 0 where no new first contact), advanced now [B] bool)."""
        B = st.B
        dev = st.device
        between = torch.zeros(B, device=dev)
        hit = torch.zeros(B, device=dev)
        nclumps = sum(h.float() for h in self.has).clamp(min=1)
        first_before = self.watch[0].contact.clone()
        u = st.u
        for k, (w, h) in enumerate(zip(self.watch, self.has)):
            before = w.contact.clone()
            _, got, ht = w.step(st, was, dt, area)
            nk = w.contact & ~before
            if bool(nk.any()):
                m = ((self.setup.clump == k) & (u["men"] > 0) & ~u["gone"]).float()
                n = m.sum(1).clamp(min=1)
                c = torch.stack([(u["x"] * m).sum(1) / n, (u["z"] * m).sum(1) / n], 1)
                self.cc[:, k] = torch.where(nk[:, None], c, self.cc[:, k])
            between = between + (got & h).float()
            hit = hit + (ht & h).float()
        u = st.u
        own = self.setup.own
        alive = own & (u["men"] > 0) & ~u["gone"]
        gat = torch.zeros(B, device=dev)
        new = self.watch[0].contact & ~first_before
        if bool(new.any()):
            N = st.N
            hs = u["hit_sec"].view(B, N, N)
            touch = ((hs > 0) | (hs.transpose(1, 2) > 0)) & own[:, :, None] & ~own[:, None, :] & (u["side"] > 0)[:, None, :]
            t = touch.any(2).float()
            k = t.sum(1).clamp(min=1)
            px, pz = (u["x"] * t).sum(1) / k, (u["z"] * t).sum(1) / k
            d = torch.sqrt((u["x"] - px[:, None]) ** 2 + (u["z"] - pz[:, None]) ** 2)
            share = ((d <= self.gather_m) & alive).float().sum(1) / alive.float().sum(1).clamp(min=1)
            self.gathered = torch.where(new, share, self.gathered)
            gat = torch.where(new, share, gat)
        w = alive.float()
        k = w.sum(1).clamp(min=1)
        mx, mz = (u["x"] * w).sum(1) / k - self.x0, (u["z"] * w).sum(1) / k - self.z0
        adv = ((mx * self.ux + mz * self.uz) > self.advance_m) & ~self.advanced & ~was
        self.advanced = self.advanced | adv
        return between / nclumps, hit / nclumps, gat, adv

    @property
    def contact(self):
        return self.watch[0].contact | self.watch[1].contact

    def all_contact(self):
        return torch.stack([w.contact | ~h for w, h in zip(self.watch, self.has)]).all(0)

    def last_contact_t(self):
        t = torch.stack([torch.where(h, w.t_contact, torch.zeros_like(w.t_contact)) for w, h in zip(self.watch, self.has)])
        return t.amax(0)

    def between_share(self):
        """[B] the share of the clumps with 'between' at their contact."""
        n = sum(h.float() for h in self.has).clamp(min=1)
        return sum((w.between & h).float() for w, h in zip(self.watch, self.has)) / n

    def stayed_share(self, within_m):
        """[B] the share of the clumps that met the enemy (their first contact) within within_m of where they started."""
        n = sum(h.float() for h in self.has).clamp(min=1)
        d = (self.cc - self.c0).norm(dim=2)                                   # nan: no contact
        ok = torch.nan_to_num(d, nan=1e9) <= within_m
        return sum((ok[:, k] & h).float() for k, h in enumerate(self.has)) / n

    def hit_share(self):
        n = sum(h.float() for h in self.has).clamp(min=1)
        return sum((w.hit & h).float() for w, h in zip(self.watch, self.has)) / n


class Ender:
    def __init__(self, setup):
        st = setup.st
        self.setup = setup
        self.over = torch.zeros(st.B, dtype=torch.bool, device=st.device)
        self.why = torch.zeros(st.B, dtype=torch.long, device=st.device)      # 1 we are out, 2 the enemy, 3 time
        self.fight = float(cfg()["cases"]["fight_s"])

    def after_step(self, st, rd):
        u = st.u
        own = self.setup.own
        out = u["r"] | (u["men"] <= 0) | u["gone"] | (u["side"] <= 0)
        we = (out | ~own).all(1)
        they = (out | own).all(1)
        late = st.t >= self.setup.limit
        fought = rd.all_contact() & (st.t >= rd.last_contact_t() + self.fight)
        new = ~self.over & (we | they | late | fought)
        self.why = torch.where(new, torch.where(we, 1, torch.where(they, 2, 3)), self.why)
        self.over = self.over | new
        st.winner = torch.where(new, torch.where(we, 2, torch.where(they, 1, 0)), st.winner)
        st.done = self.over.clone()
        return new

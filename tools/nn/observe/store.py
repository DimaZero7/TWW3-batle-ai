"""The demonstrations in training (tools/nn/train/run.py --observe-dir): the converted battles (convert.py), cut into
chunks of T decisions, a few chunks beside every PPO minibatch (loss.py's term).

* Which labels count (the advantage A of the label's decision, convert.py): "critic" (default) - GAE of the reward v2
  to the battle's end minus the network's critic V (the critic of the converting checkpoint); "mc" - the plain return
  minus V; "window20" - the fallback without the critic: the gold trade of the next 20 s minus the battle's mean, the
  trade itself above 0 too. Selected: A > 0. The weight c = min(exp(A / (beta x std)), clip), std the standard
  deviation of A over all the store's labels (AWR's exp(A / beta) on a normalised advantage; clip 20 as AWR's).
* The memory (GRU) of a chunk: the actor's memory at the chunk's start, run through the battle from its beginning
  (R2D2's stored state); computed again every --observe-refresh updates (the weights change), no gradients.
* Only chunks with a selected label are drawn, uniformly. The units of every battle are padded to the store's largest
  (not present, not attended, no orders).
* v2's inputs the recording does not have: every unit that takes orders is free (no commitment), commit 0; geo from
  the side's frame and the map.
"""
import json
from pathlib import Path

import numpy as np
import torch

from tools.nn.observe import loss as obs_loss

OBS = ("tokens", "ctx", "own", "attend", "ctrl", "target_ok", "pos", "abil", "abil_ok")
LABELS = ("kind", "target", "point", "run", "run_known", "alt")
ADV = ("critic", "mc", "window20")


def _pad(a, N, axis=1, value=0):
    """a padded along the units' axis to N (value: of the new units; -1 for the labels' "none")."""
    if a.shape[axis] == N:
        return a
    w = [(0, 0)] * a.ndim
    w[axis] = (0, N - a.shape[axis])
    return np.pad(a, w, constant_values=value)


class Demos:
    """The store (module doc)."""

    def __init__(self, folder, T=64, adv="critic", beta=1.0, clip=20.0, cfg=None, files=None):
        assert adv in ADV, adv
        paths = sorted(Path(folder).glob("*.npz")) if files is None else [Path(f) for f in files]
        if not paths:
            raise SystemExit(f"--observe-dir {folder}: no converted battles (python -m tools.nn.observe.convert)")
        raw = []
        for p in paths:
            d = np.load(p)
            meta = json.loads(str(d["meta"]))
            if cfg is not None and (meta["cfg"]["sectors"], meta["cfg"]["fine"], meta["cfg"]["grid_half"]) != \
                    (cfg.sectors, cfg.fine, cfg.grid_half):
                raise SystemExit(f"{p}: converted for another grid {meta['cfg']}: convert again with this network")
            for s in meta["sides"]:
                g = {k[len(f"s{s}_"):]: d[k] for k in d.files if k.startswith(f"s{s}_")}
                if len(g["t"]) < T:
                    continue
                g["name"] = f"{meta['run']}/s{s}"
                raw.append(g)
        if not raw:
            raise SystemExit(f"--observe-dir {folder}: no battle side of {T} decisions or more")
        self.T, self.adv, self.beta, self.clip = T, adv, beta, clip
        self.N = max(g["tokens"].shape[1] for g in raw)
        self.seqs = []
        A_all, both = [], []
        for g in raw:
            lab = g["kind"] >= 0
            a = g["w20_adv"] if adv == "window20" else g[f"adv_{adv}"]
            good = a > 0
            if adv == "window20":
                good = good & (g["w20_trade"] > 0)
            rows = np.nonzero(lab)[0]
            A_all.append(a[rows])
            w20 = (g["w20_adv"] > 0) & (g["w20_trade"] > 0)
            both.append(np.stack([good[rows], w20[rows]], -1))
            self.seqs.append({"name": g["name"], "L": len(g["t"]), "A": a, "good": good,
                              "obs": {k: torch.as_tensor(g[k] if k == "ctx" else _pad(g[k], self.N))
                                      for k in OBS if k in g},
                              "geo": torch.as_tensor(g["geo"]),
                              "lab": {k: torch.as_tensor(_pad(g[k], self.N, value=-1 if g[k].dtype.kind == "i" else 0))
                                      for k in LABELS}})
        A_all = np.concatenate(A_all)
        both = np.concatenate(both).astype(bool)
        self.std = float(A_all.std()) if len(A_all) > 1 and A_all.std() > 0 else 1.0
        # c per labelled unit-decision: the side's A of that decision
        for q in self.seqs:
            lab = q["lab"]["kind"].numpy() >= 0
            c = np.where(q["good"][:, None] & lab, np.minimum(np.exp(q["A"][:, None] / (beta * self.std)), clip), 0.0)
            q["c"] = torch.as_tensor(c.astype(np.float32))
        # chunks: (seq, start); the last one overlaps to end at the battle's end
        self.chunks = []
        for i, q in enumerate(self.seqs):
            L = q["L"]
            starts = sorted(set(list(range(0, L - T + 1, T)) + [L - T]))
            q["starts"] = starts
            for s in starts:
                if bool((q["c"][s:s + T] > 0).any()):
                    self.chunks.append((i, s))
        self.h = {}                     # (seq, start) -> memory [1 + N, d] (refresh)
        n_lab = len(A_all)
        sel = both[:, 0]
        q = np.percentile(A_all, [5, 25, 50, 75, 95]) if n_lab else np.zeros(5)
        self.info = {"battle_sides": len(self.seqs), "decisions": int(sum(q_["L"] for q_ in self.seqs)),
                     "labels": int(n_lab), "selected": int(sel.sum()), "selected_share": float(sel.mean()) if n_lab else 0,
                     "chunks": len(self.chunks), "adv": adv, "adv_std": self.std,
                     "adv_q05_25_50_75_95": [round(float(x), 4) for x in q],
                     "agree_w20": {"both": int((sel & both[:, 1]).sum()), "only_this": int((sel & ~both[:, 1]).sum()),
                                   "only_w20": int((~sel & both[:, 1]).sum()), "neither": int((~sel & ~both[:, 1]).sum())},
                     "mean_c_selected": float(np.mean(np.concatenate(
                         [q_["c"].numpy()[q_["c"].numpy() > 0] for q_ in self.seqs] or [np.zeros(1)])))}
        if not self.chunks:
            raise SystemExit(f"--observe-dir {folder}: no selected label ({adv})")

    def text(self):
        i = self.info
        a = i["agree_w20"]
        return (f"observation: {i['battle_sides']} battle sides, {i['decisions']} decisions, {i['labels']} labelled "
                f"orders, selected (A>0 by {i['adv']}) {i['selected']} ({i['selected_share']:.2f}), {i['chunks']} chunks; "
                f"A std {i['adv_std']:.3f}, quantiles 5/25/50/75/95 {i['adv_q05_25_50_75_95']}; mean weight of the "
                f"selected {i['mean_c_selected']:.2f}; with the window20 rule: both {a['both']}, only this {a['only_this']}, "
                f"only window20 {a['only_w20']}, neither {a['neither']}")

    def _obs(self, i, a, b, device):
        q = self.seqs[i]
        o = {k: v[a:b].to(device) for k, v in q["obs"].items()}
        for k in ("tokens", "ctx", "abil", "pos"):
            if k in o:
                o[k] = o[k].float()
        o["free"] = o["ctrl"].clone()
        o["commit"] = torch.zeros(*o["ctrl"].shape, 2, device=device)
        o["geo"] = q["geo"].to(device).float()[None].expand(b - a, -1)
        return o

    @torch.no_grad()
    def refresh(self, actor, device):
        """The memory at every chunk's start: each battle side run through the actor from its beginning (module doc)."""
        was = actor.training
        actor.eval()
        self.h = {}
        for i, q in enumerate(self.seqs):
            h = None
            starts = q["starts"] + [q["L"]]
            for a, b in zip(starts[:-1], starts[1:]):
                o = {k: v[:, None] for k, v in self._obs(i, a, b, device).items()}
                h0 = actor.initial({k: v[0] for k, v in o.items()}) if h is None else h
                self.h[(i, a)] = h0[0].detach().to("cpu")
                if actor.memory is None:
                    continue
                _, _, _, h = obs_loss.trunk(actor, o, h0)
        actor.train(was)

    def sample(self, n, device, gen=None):
        """n chunks -> {obs [T, n, ...], h0 [n, 1 + N, d], labels [T, n, N], c [T, n, N]}."""
        idx = torch.randint(len(self.chunks), (n,), generator=gen).tolist()
        T = self.T
        parts = [self._obs(*self.chunks[j], self.chunks[j][1] + T, device) for j in idx]
        obs = {k: torch.stack([p[k] for p in parts], 1) for k in parts[0]}
        out = {"obs": obs}
        for k in LABELS:
            out[k] = torch.stack([self.seqs[i]["lab"][k][s:s + T] for i, s in (self.chunks[j] for j in idx)], 1).to(device)
        for k in ("kind", "target", "point"):
            out[k] = out[k].long()
        for k in ("run", "run_known", "alt"):
            out[k] = out[k].bool()
        out["c"] = torch.stack([self.seqs[i]["c"][s:s + T] for i, s in (self.chunks[j] for j in idx)], 1).to(device)
        hs = [self.h.get(self.chunks[j]) for j in idx]
        if any(h is None for h in hs):
            raise RuntimeError("Demos.sample before refresh()")
        out["h0"] = torch.stack(hs).to(device)
        return out


class Observe:
    """What ppo.update needs of the observation: the demonstrations, the term's weight (decaying x decay an update),
    the KL anchor to the frozen network the run started from (its weight x anchor_decay an update), how many chunks
    beside a PPO minibatch, the memory's refresh. updates: the updates of earlier parts of the chain (the decay goes on)."""

    def __init__(self, demos, chunks, weight=0.1, decay=0.95, anchor_net=None, anchor=0.2, anchor_decay=0.995,
                 refresh=20, updates=0, device="cpu", seed=0):
        self.demos, self.chunks, self.refresh_every = demos, max(1, int(chunks)), max(1, int(refresh))
        self.w0, self.decay, self.a0, self.anchor_decay = weight, decay, anchor, anchor_decay
        self.anchor_net = anchor_net
        self.updates = int(updates)
        self.device = device
        self.gen = torch.Generator().manual_seed(seed)
        self.since = None

    @property
    def weight(self):
        return self.w0 * self.decay ** self.updates

    @property
    def anchor(self):
        return self.a0 * self.anchor_decay ** self.updates if self.anchor_net is not None else 0.0

    def before(self, actor):
        """Before an update: the stored memories again when due."""
        if self.since is None or self.since >= self.refresh_every:
            self.demos.refresh(actor, self.device)
            self.since = 0

    def term(self, actor):
        return obs_loss.term(actor, self.demos.sample(self.chunks, self.device, self.gen))

    def after(self):
        self.updates += 1
        self.since = (self.since or 0) + 1

"""S0-1's input and network (tmp/drill_s1/design_merged.md section 1; design_opus section 3): one blind decision a
battle, at tick 0.

    observe(specs, device) -> [obs]   each battle's input, built on a simulator state of the battle BEFORE the
                 deployment: both armies at their zones' default rows (net/stage0/specs.rows, no shift: the enemy's places
                 are never seen), lords by crest.lord_place (the bridge's rule); rows through the operative's real input (net/operative/heads
                 inputs_of: inputs.build's rows [K, 185] for the frozen card8 'Анализ отряда', mask, place, gframe, my
                 infantry as members, the enemy as targets, every enemy row seen - the composition is known), plus
                    seat [SEAT]   att, my count / 20, the enemy's / 20, the zone's front / 416, depth / 176
                    cand [C, F]   C = 1 + crest top: KEEP (the centre line) and the finder's <= 3 candidates (padding:
                                  zeros, illegal); F = type one-hot (KEEP, CREST) 2, valid, band, band_p25, share05,
                                  band_slot_min, band - band(centre), rise60 / 10, rise100 / 10, back20 / 10, s / 208,
                                  f / 88, turn / 20, blocked, forest, water, profile 12 / 10 m
                    legal [C]     KEEP always; a candidate when valid
    StratNet1()  OpNet1's form (net/operative/heads.py): records of the rows (card 128 + place 3 + gframe 6 -> W), my
                 and the enemy's tokens, the seat as the council's summary (tactic/decide.Council); the candidate scorer
                 (summary, my token after the council, the candidate's F) -> hidden -> 1 with the last layer zero, the
                 zero-initialised per-candidate-type readout (heads.PICK_WHY; types KEEP / CREST) and a learned KEEP bias:
                 a fresh network = DEFAULT (always KEEP)
    choose(net, obs) -> [k]   greedy: 0 KEEP, k a candidate
    save(path, net, extra) / restore(path, device)   kind 'strategist_s1_v1'
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.strategist import crest as CR

KIND = "strategist_s1_v1"
SEAT = 5
TOP = 3
C = 1 + TOP
PROFILE = 12
FEATS = 2 + 1 + 4 + 1 + 3 + 3 + 3 + PROFILE
CAND_COLS = (("keep", "crest", "valid", "band", "band_p25", "share05", "band_slot_min", "delta_band", "rise60",
              "rise100", "back20", "s", "f", "turn", "blocked", "forest", "water")
             + tuple(f"profile{i}" for i in range(PROFILE)))
SEAT_COLS = ("att", "mine_n", "theirs_n", "front", "depth")
PLACE_COLS = ("s", "f", "height")
_COLS_WHY = "the columns by name: the label probe's ablation reads the card's names (cards/s0_1.PROBE_ABLATE), not indices"
assert len(CAND_COLS) == FEATS and len(SEAT_COLS) == SEAT
KEYS_HD = ("rows", "mask", "place", "gframe", "members", "targets", "tokens")
NET_KEYS = KEYS_HD + ("seat", "cand", "legal")
HIDDEN = 64
KEEP_BIAS = 2.0
_NET_WHY = "hidden 64 and the KEEP bias 2.0: net/config/operative.json council_extra (OpNet1's), the same form"


def cand_feats(t, centre, keep):
    if t is None:
        return [0.0] * FEATS
    prof = [v / 10.0 for v in t["profile"]]
    return ([1.0, 0.0] if keep else [0.0, 1.0]) + [1.0 if t["valid"] else 0.0, t["band"], t["band_p25"], t["share05"],
                                                    t["band_slot_min"], t["band"] - centre["band"], t["rise60"] / 10,
                                                    t["rise100"] / 10, t["back20"] / 10, t["s"] / 208.0, t["f"] / 88.0,
                                                    t["turn"] / 20.0, t["blocked"], t["forest"], t["water"]] + prof


def tokens_of(fd):
    """(cand [C, F], legal [C]) of the finder's output (KEEP first)."""
    rows = [cand_feats(fd["centre"], fd["centre"], True)]
    legal = [True]
    for j in range(TOP):
        t = fd["cands"][j] if j < len(fd["cands"]) else None
        rows.append(cand_feats(t, fd["centre"], False))
        legal.append(bool(t is not None and t["valid"]))
    return rows, legal


def seat_of(spec):
    return [float(spec["att"]), len(spec["mine"]) / 20.0, len(spec["theirs"]) / 20.0,
            float(spec["mine_zone"]["front"]) / 416.0, float(spec["mine_zone"]["depth"]) / 176.0]


def pre_battles(specs):
    """The battles before the deployment (module doc)."""
    from net.stage0 import battle as BA
    from net.stage0 import cfg as scfg
    from net.stage0 import specs as SP
    out = []
    for s in specs:
        mz, ez = s["mine_zone"], s["enemy_zone"]
        pm, pt = SP.rows(mz, len(s["mine"])), SP.rows(ez, len(s["theirs"]))
        mine = [BA._unit(k, *p) for k, p in zip(s["mine"], pm)]
        theirs = [BA._unit(k, *p) for k, p in zip(s["theirs"], pt)]
        l1 = BA._unit(scfg()["lords"][s["fac"][1]], *CR.lord_place(pm, mz["b"], mz), general=True)   # (crest.lord_place:
        l2 = BA._unit(scfg()["lords"][s["fac"][2]], *CR.lord_place(pt, ez["b"], ez), general=True)   # the bridge's rule)
        out.append({1: [l1] + mine, 2: [l2] + theirs, "factions": dict(s["factions"])})
    return out


@torch.no_grad()
def observe(specs, device="cpu", P=None, chunk=128):
    """[obs] (module doc), each a dict of NET_KEYS tensors on the cpu."""
    from net.matchup import duel as DU
    from net.operative import heads as HD
    from net.stage0 import battle as BA
    from net.strategist import s1_ground as SG
    from net.strategist import s1_scenes as SS
    from net.tactic import raw as TR
    from sim.params import load
    P = P or load()
    out = []
    for a in range(0, len(specs), chunk):
        part = specs[a:a + chunk]
        maps = [s["map"] for s in part]
        SG.register(maps)
        st, card = DU.build(BA.NAME, pre_battles(part), device, P, BA.PER_SIDE, maps=maps)
        raw = TR.Seer(P).raw(st)
        u = st.u
        side = u["side"]
        raw[..., TR.I["vis"]] = torch.where(side == 2, torch.ones_like(raw[..., TR.I["vis"]]), raw[..., TR.I["vis"]])
        for b, s in enumerate(part):
            mem = (side[b] == 1) & ~u["lord"][b] & (u["men"][b] > 0)
            E = (side[b] == 2) & ~u["lord"][b] & (u["men"][b] > 0)
            mz = s["mine_zone"]
            axes = HD.axes_of(raw[b:b + 1], P.map_half)
            o = HD.inputs_of(raw[b:b + 1], card[b:b + 1], mem, E, (float(mz["x"]), float(mz["z"]), float(mz["b"])), 0,
                             axes)
            cand, legal = tokens_of(SS.found(s))
            ob = {k: o[k].detach().cpu() for k in KEYS_HD}
            ob["seat"] = torch.tensor(seat_of(s), dtype=torch.float32)
            ob["cand"] = torch.tensor(cand, dtype=torch.float32)
            ob["legal"] = torch.tensor(legal, dtype=torch.bool)
            out.append(ob)
    return out


def stack(obs):
    return {k: torch.stack([o[k] for o in obs]) for k in NET_KEYS}


class StratNet1(nn.Module):
    def __init__(self, card=True):
        super().__init__()
        from net.operative import gframe as GF
        from net.operative import groups as GR
        from net.tactic import cfg as tcfg
        from net.tactic import decide as DC
        from net.tactic import net as NT
        W = int(tcfg()["council"]["width"])
        self.analysis = NT.UnitAnalysis()
        A = int(tcfg()["analysis"]["out"])
        self.record = nn.Linear(A + 3 + len(GF.COLS), W)
        self.group = nn.Linear(W + GR.NUMBERS, W)
        self.enemy = nn.Linear(W, W)
        self.seat = nn.Linear(SEAT, W)
        self.council = DC.Council()
        self.s1 = nn.Linear(2 * W + FEATS, HIDDEN)
        self.s2 = nn.Linear(HIDDEN, 1)
        self.pick = nn.Linear(2 * W, 2)                  # (heads.PICK_WHY) per candidate type: KEEP, CREST
        with torch.no_grad():
            for m in (self.s2, self.pick):
                m.weight.zero_()
                m.bias.zero_()
        self.keep_bias = nn.Parameter(torch.tensor(KEEP_BIAS))
        if card:
            ck = torch.load(DC.CARD_CKPT, map_location="cpu", weights_only=False)
            self.analysis.load_state_dict(ck["analysis"])
        for p in self.analysis.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        from net.unit.net import NEG
        mask = x["mask"]
        zero = torch.zeros((), device=mask.device)
        cards = self.analysis(x["rows"], mask)
        rec = torch.where(mask[..., None], F.relu(self.record(torch.cat([cards, x["place"], x["gframe"]], -1))), zero)

        def mean(sel):
            w = (sel & mask).float()
            return (rec * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp(min=1)
        g = F.relu(self.group(torch.cat([mean(x["members"]), x["tokens"]], -1)))
        e = F.relu(self.enemy(mean(x["targets"])))
        s = self.seat(x["seat"])
        M = mask.shape[0]
        seq = torch.cat([rec, g[:, None], e[:, None]], 1)
        mm = torch.cat([mask, torch.ones((M, 2), dtype=torch.bool, device=mask.device)], 1)
        after, sm = self.council(seq, s, mm)
        ga = after[:, -2]
        Cn = x["cand"].shape[1]
        z = torch.cat([sm[:, None].expand(M, Cn, sm.shape[-1]), ga[:, None].expand(M, Cn, ga.shape[-1]), x["cand"]], -1)
        logit = self.s2(F.relu(self.s1(z)))[..., 0]
        logit = logit + (self.pick(torch.cat([sm, ga], -1))[:, None, :] * x["cand"][..., :2]).sum(-1)
        logit = logit + F.one_hot(torch.zeros((), dtype=torch.long, device=mask.device), Cn).float() * self.keep_bias
        return torch.where(x["legal"], logit, torch.full_like(logit, NEG))


@torch.no_grad()
def choose(net, obs, device="cpu", chunk=256):
    net.eval()
    out = []
    for a in range(0, len(obs), chunk):
        x = {k: v.to(device) for k, v in stack(obs[a:a + chunk]).items()}
        out += net(x).argmax(1).tolist()
    return out


def save(path, net, extra=None):
    import os
    tmp = str(path) + ".part"
    with open(tmp, "wb") as fh:
        torch.save({"kind": KIND, "net": net.state_dict(), **(extra or {})}, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def restore(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck.get("kind") != KIND:
        raise ValueError(f"not a {KIND} checkpoint: {path}")
    net = StratNet1(card=False).to(device)
    net.load_state_dict(ck["net"])
    for p in net.analysis.parameters():
        p.requires_grad_(False)
    return net, ck

"""The exam of drill U3 'several enemies' (drill_multi.py): the fixed set drill_multi.exam_specs() (240: 40 per layout x
timing), the network sampling with a fixed seed, the branches (branch.KIT_MULTI) in every skirmish.

    bash tools/nn/dock.sh tools.nn.unit.exam_multi [--ckpt build/.../latest.pt] [--device cuda]

Metrics: chose the best of the 4 variants (attack A / attack B / stand facing A / facing B; the network's behaviour at
its first contact, branch.Kit.behaviour); which variant was best how often (attack vs stand, A vs B) and the margin;
melee damage taken to the front / the flank and rear (each step's loss split among the enemies in contact by the sector
each is in: within contact.front_deg 45 deg of our facing the front, beyond rear_deg 135 the rear); met the first
enemy head-on (the first contact's enemy in our front sector); the trade; out of the area (own choice / pushed); by
layout and timing.
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch

from tools.nn.sim.params import load
from tools.nn.train import reward
from tools.nn.unit import branch as Br
from tools.nn.unit import decode as Dc
from tools.nn.unit import drill_multi as DM
from tools.nn.unit import net as Nt

SEED = 4906


class Watch:
    def __init__(self, B, N, P, device):
        self.N, self.dt = N, P.dt
        cc = P.sim["contact"]
        self.front, self.rear = math.radians(cc["front_deg"]), math.radians(cc["rear_deg"])
        z = lambda: torch.zeros(B, device=device)       # noqa: E731
        self.first_sec = torch.full((B,), -1, dtype=torch.long, device=device)
        self.dmg_front, self.dmg_side, self.dmg_all = z(), z(), z()
        self.out_free, self.out_pushed, self.alive_s = z(), z(), z()

    def sectors(self, u):
        """[B, N] the sector (0 front, 1 flank, 2 rear) each slot's centre is in, seen from our unit."""
        dx, dz = u["x"] - u["x"][:, :1], u["z"] - u["z"][:, :1]
        rel = torch.remainder(torch.atan2(dx, dz) - torch.deg2rad(u["b"][:, :1]) + math.pi, 2 * math.pi) - math.pi
        a = rel.abs()
        return torch.where(a <= self.front, 0, torch.where(a >= self.rear, 2, 1))

    def __call__(self, st, hp0, was_done, area, step):
        u = st.u
        live = ~was_done
        hs = u["hit_sec"].view(st.B, self.N, self.N)[:, :, 0] > 0          # [B, j] j's contact on us
        hs = hs & (u["side"] == 2)
        sec = self.sectors(u)
        any_c = hs.any(1)
        first = live & any_c & (self.first_sec < 0)
        # the first contact's enemy: the one in contact (the nearer to our front's axis if two at once)
        j = torch.where(hs, sec, torch.full_like(sec, 9)).argmin(1)
        self.first_sec = torch.where(first, sec.gather(1, j[:, None])[:, 0], self.first_sec)
        lost = (hp0 - u["hp_abs"][:, 0]).clamp(min=0) * (live & any_c).float()
        n = hs.float().sum(1).clamp(min=1)
        front = ((sec == 0) & hs).float().sum(1) / n
        self.dmg_all += lost
        self.dmg_front += lost * front
        self.dmg_side += lost * (1 - front) * any_c.float()
        alive = live & (u["men"][:, 0] > 0)
        out = Dc.signed_dist(u["x"][:, 0], u["z"][:, 0], {k: a[:, 0] for k, a in area.items()}) < -0.5
        pushed = u["m"][:, 0] | u["r"][:, 0]
        self.alive_s += alive.float() * self.dt
        self.out_free += (alive & out & ~pushed).float() * self.dt
        self.out_pushed += (alive & out & pushed).float() * self.dt


def _mean(x, m):
    return round(float(x[m].float().mean()), 3) if bool(m.any()) else None


def run(net, P=None, device="cpu", duty=None, out_path=None, seed=SEED):
    from tools.nn.unit import train as T
    P = P or load()
    duty = duty or T.Duty(device, 1.0)
    sp = DM.exam_specs()
    cfg = {"battles": len(sp), "seconds": DM.LIMIT_S, "cell_m": 8.0, "bounds": None}
    torch.manual_seed(seed)
    setup = T.multi_setup(cfg, P, device, sp=sp)
    st = setup["st"]
    w = Watch(st.B, st.N, P, device)
    m0 = reward.measure(st)
    t0 = time.time()
    was = net.training
    net.eval()
    _, rs = T.rollout(net, None, cfg, P, device, duty, setup, watch=w,
                      branch={"share": 1.0, "weight": 0.0, "lead_s": Br.LEAD_S, "horizon_s": Br.HORIZON_S})
    net.train(was)
    m1 = reward.measure(st)
    bi = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in rs.pop("_branch").items()}
    trade = ((m0[:, 1, 3] - m1[:, 1, 3]) - (m0[:, 0, 3] - m1[:, 0, 3])).cpu()
    beh, best, brd = bi["behaviour"], bi["best"], bi["branched"]
    ok = brd & (bi["contact_d"] >= 0)
    names = bi["names"]
    _, margin = Br.best_of(torch.nan_to_num(bi["trades"]))
    fs = w.first_sec.cpu()

    def block(m):
        da = w.dmg_all.cpu()[m].sum().clamp(min=1e-6)
        return {"n": int(m.sum()), "chose_best": _mean(beh == best, m & ok),
                "best": {nm: _mean(best == v, m & brd) for v, nm in enumerate(names)},
                "behaviour": {nm: _mean(beh == v, m & ok) for v, nm in enumerate(names)},
                "best_attack": _mean(best <= 1, m & brd), "best_on_A": _mean((best == 0) | (best == 2), m & brd),
                "margin": _mean(margin, m & brd), "head_on_first": _mean(fs == 0, m & (fs >= 0)),
                "dmg_front": round(float(w.dmg_front.cpu()[m].sum() / da), 3),
                "dmg_flank_rear": round(float(w.dmg_side.cpu()[m].sum() / da), 3), "trade": _mean(trade, m)}
    allm = torch.ones(len(sp), dtype=torch.bool)
    alive = float(w.alive_s.sum())
    table = {"all": block(allm),
             "by_layout": {k: block(torch.tensor([s["layout"] == k for s in sp])) for k in DM.LAYOUTS},
             "by_timing": {k: block(torch.tensor([s["timing"] == k for s in sp])) for k in DM.TIMINGS},
             "out_of_area_free": round(float(w.out_free.sum()) / max(alive, 1e-6), 4),
             "out_of_area_pushed": round(float(w.out_pushed.sum()) / max(alive, 1e-6), 4),
             "end": rs.get("end"), "duration_s": round(float(st.t.mean()), 1), "entropy": rs["entropy"],
             "bans_broken": rs["bans_broken"], "stand_branch_braced": (rs.get("branch") or {}).get("stand_branch_braced"),
             "seconds": round(time.time() - t0, 1)}
    if out_path:
        Path(out_path).write_text(json.dumps(table, indent=1), encoding="utf-8")
    print(json.dumps({"exam_multi": table}), flush=True)
    return table


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="a unit_v1 checkpoint (none: an untrained network, seed 0)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--gpu-duty", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    from tools.nn.unit import train as T
    if a.ckpt:
        net, _, _ = T.restore(a.ckpt, a.device)
    else:
        torch.manual_seed(0)
        net = Nt.UnitNet().to(a.device)
    return run(net, load(), a.device, T.Duty(a.device, a.gpu_duty), a.out)


if __name__ == "__main__":
    main()

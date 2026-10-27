"""The exam of drill U2 'reach and hold' (drill_go.py): the fixed set drill_go.exam_specs() (360: every pair once x race
early / close / held x pace fast / slow), the network sampling with a fixed seed.

    bash tools/nn/dock.sh tools.nn.unit.exam_go [--ckpt build/.../latest.pt] [--device cuda]

Metrics: reached the area (share) and in how many seconds; before the enemy (our centre in the area first); the area
ours / the enemy's (drill_go.whose_area: shares of the skirmish's seconds; area_ours_old - our centre in it, standing -
the first measure, kept to compare); met head-on (the enemy's centre in our front
sector at the contact, as exam_hold); out of the area after the arrival, of its own choice / pushed; the trade; the
choice's entropy; all also by pace and by race.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from tools.nn.sim.params import load
from tools.nn.train import reward
from tools.nn.unit import decode as Dc
from tools.nn.unit import drill_go as DG
from tools.nn.unit import exam_hold as EH
from tools.nn.unit import net as Nt

SEED = 4904


class Watch(EH.Watch):
    """exam_hold's metrics, the out-of-area time counted only after the arrival, and the time the area is ours."""

    def __init__(self, B, N, P, device, ender):
        super().__init__(B, N, P, device)
        self.ender = ender
        self.ours_s = torch.zeros(B, device=device)
        self.own_s = torch.zeros(B, device=device)          # whose_area: ours
        self.theirs_s = torch.zeros(B, device=device)       # ... the enemy's
        self.after_s = torch.zeros(B, device=device)

    def __call__(self, st, hp0, was_done, area, step):
        u = st.u
        live = ~was_done
        hs = u["hit_sec"][:, self.H * self.N + 0]
        sec = self.sector(u)
        first = live & (hs > 0) & (self.contact_t < 0)
        self.contact_t = torch.where(first, st.t, self.contact_t)
        self.contact_sec = torch.where(first, sec, self.contact_sec)
        self.contact_game = torch.where(first, hs.long() - 1, self.contact_game)
        lost = (hp0 - u["hp_abs"][:, 0]).clamp(min=0) * (live & (hs > 0)).float()
        self.dmg_all += lost
        self.dmg_front += lost * (sec == 0).float()
        alive = live & (u["men"][:, 0] > 0)
        sd = Dc.signed_dist(u["x"][:, 0], u["z"][:, 0], {k: a[:, 0] for k, a in area.items()})
        after = alive & self.ender.arrived
        pushed = u["m"][:, 0] | u["r"][:, 0]
        self.alive_s += alive.float() * self.dt
        self.after_s += after.float() * self.dt
        self.out_free += (after & (sd < -0.5) & ~pushed).float() * self.dt
        self.out_pushed += (after & (sd < -0.5) & pushed).float() * self.dt
        self.ours_s += (alive & (sd >= 0) & ~u["r"][:, 0]).float() * self.dt
        wa = DG.whose_area(st, area)
        self.own_s += (live & (wa == DG.OURS)).float() * self.dt
        self.theirs_s += (live & (wa == DG.THEIRS)).float() * self.dt


def _mean(x, m):
    return round(float(x[m].float().mean()), 3) if bool(m.any()) else None


def run(net, P=None, device="cpu", duty=None, out_path=None, seed=SEED):
    from tools.nn.unit import train as T
    P = P or load()
    duty = duty or T.Duty(device, 1.0)
    sp = DG.exam_specs()
    cfg = {"battles": len(sp), "seconds": 700.0, "cell_m": 8.0, "bounds": None}
    torch.manual_seed(seed)
    setup = T.go_setup(cfg, P, device, sp=sp)
    st, ender = setup["st"], setup["ender"]
    w = Watch(st.B, st.N, P, device, ender)
    m0 = reward.measure(st)
    t0 = time.time()
    was = net.training
    net.eval()
    _, rs = T.rollout(net, None, cfg, P, device, duty, setup, watch=w)
    net.train(was)
    m1 = reward.measure(st)
    trade = ((m0[:, 1, 3] - m1[:, 1, 3]) - (m0[:, 0, 3] - m1[:, 0, 3])).cpu()
    arr, at, et = ender.arrived.cpu(), ender.arrive_t.cpu(), ender.enemy_t.cpu()
    first = arr & ((et < 0) | (at <= et))
    dur = st.t.cpu().clamp(min=1e-6)
    ours = w.ours_s.cpu() / dur
    own = w.own_s.cpu() / dur
    theirs = w.theirs_s.cpu() / dur
    ct, cs = w.contact_t.cpu(), w.contact_sec.cpu()
    met = ct >= 0
    after = w.after_s.cpu().clamp(min=1e-6)

    def block(m):
        return {"n": int(m.sum()), "reached": _mean(arr, m), "reach_s": _mean(at, m & arr), "before_enemy": _mean(first, m),
                "area_ours_old": _mean(ours, m), "area_ours": _mean(own, m), "area_theirs": _mean(theirs, m), "contact": _mean(met, m), "front_at_contact": _mean(cs == 0, m & met),
                "out_free": round(float(w.out_free.cpu()[m].sum() / after[m].sum()), 4),
                "trade": _mean(trade, m)}
    allm = torch.ones(len(sp), dtype=torch.bool)
    table = {"all": block(allm),
             "by_pace": {p: block(torch.tensor([s["pace"] == p for s in sp])) for p in DG.PACES},
             "by_race": {r: block(torch.tensor([s["race"] == r for s in sp])) for r in DG.RACES},
             "far_area": block(torch.tensor([s["edge"] > DG.NEAR_M[1] for s in sp])),
             "out_pushed": round(float(w.out_pushed.sum() / after.sum()), 4),
             "end": rs.get("end"), "duration_s": round(float(st.t.mean()), 1), "entropy": rs["entropy"],
             "bans_broken": rs["bans_broken"], "seconds": round(time.time() - t0, 1)}
    if out_path:
        Path(out_path).write_text(json.dumps(table, indent=1), encoding="utf-8")
    print(json.dumps({"exam_go": table}), flush=True)
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

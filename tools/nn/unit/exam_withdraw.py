"""The exam of drill U6 'withdraw from melee' (drill_withdraw.py): the fixed set drill_withdraw.exam_specs() (240: 60 per
direction behind / aside x pace fast / slow), the network sampling with a fixed seed.

    bash tools/nn/dock.sh tools.nn.unit.exam_withdraw [--ckpt build/.../latest.pt] [--device cuda]

Metrics: reached the new area (share) and in how many seconds from the order; the losses on the way (our health lost,
share of the start, from the order to the arrival - or to the end, not arrived); routed (the morale); damage in the back
(our health lost while the enemy in contact is in our rear sector, beyond contact.rear_deg 135 deg; share of the start;
and of all the losses); the trade; out of the new area after the arrival (own choice / pushed; ~0); by direction and pace.
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch

from tools.nn.sim.params import load
from tools.nn.train import reward
from tools.nn.unit import decode as Dc
from tools.nn.unit import drill_withdraw as DW
from tools.nn.unit import net as Nt

SEED = 4912


class Watch:
    def __init__(self, B, N, P, device, ender):
        self.H, self.N, self.dt, self.ender = N // 2, N, P.dt, ender
        self.rear = math.radians(P.sim["contact"]["rear_deg"])
        z = lambda: torch.zeros(B, device=device)       # noqa: E731
        self.lost_way, self.lost_all, self.lost_back = z(), z(), z()
        self.out_free, self.out_pushed, self.after_s = z(), z(), z()
        self.prev = None

    def __call__(self, st, hp0, was_done, area, step):
        u = st.u
        H, N = self.H, self.N
        live = ~was_done
        if self.prev is None:
            self.prev = u["hp_abs"][:, 0].clone()
        lost = (self.prev - u["hp_abs"][:, 0]).clamp(min=0) / u["hp0"][:, 0].clamp(min=1) * live.float()
        self.prev = u["hp_abs"][:, 0].clone()
        self.lost_all += lost
        self.lost_way += lost * (~self.ender.arrived).float()
        hs = u["hit_sec"].view(st.B, N, N)
        contact = hs[:, H, 0] > 0
        dx, dz = u["x"][:, H] - u["x"][:, 0], u["z"][:, H] - u["z"][:, 0]
        rel = torch.remainder(torch.atan2(dx, dz) - torch.deg2rad(u["b"][:, 0]) + math.pi, 2 * math.pi) - math.pi
        self.lost_back += lost * (contact & (rel.abs() >= self.rear)).float()
        after = live & self.ender.arrived & (u["men"][:, 0] > 0)
        sd = Dc.signed_dist(u["x"][:, 0], u["z"][:, 0], {k: a[:, 0] for k, a in area.items() if k != "on"})
        pushed = u["m"][:, 0] | u["r"][:, 0]
        self.after_s += after.float() * self.dt
        self.out_free += (after & (sd < -0.5) & ~pushed).float() * self.dt
        self.out_pushed += (after & (sd < -0.5) & pushed).float() * self.dt


def _mean(x, m):
    return round(float(x[m].float().mean()), 3) if bool(m.any()) else None


def run(net, P=None, device="cpu", duty=None, out_path=None, seed=SEED):
    from tools.nn.unit import train as T
    P = P or load()
    duty = duty or T.Duty(device, 1.0)
    sp = DW.exam_specs()
    cfg = {"battles": len(sp), "seconds": 400.0, "cell_m": 8.0, "bounds": None}
    torch.manual_seed(seed)
    setup = T.withdraw_setup(cfg, P, device, sp=sp)
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
    arr = ender.arrived.cpu()
    reach = (ender.arrive_t - ender.t0).cpu()
    routed = (ender.why == 1).cpu()
    c = {k: getattr(w, k).cpu() for k in ("lost_way", "lost_all", "lost_back", "out_free", "out_pushed", "after_s")}

    def block(m):
        la = c["lost_all"][m].sum().clamp(min=1e-6)
        return {"n": int(m.sum()), "reached": _mean(arr, m), "reach_s": _mean(reach, m & arr),
                "lost_on_way": _mean(c["lost_way"], m), "routed": _mean(routed, m),
                "back_damage": _mean(c["lost_back"], m), "back_share": round(float(c["lost_back"][m].sum() / la), 3),
                "trade": _mean(trade, m),
                "out_free": round(float(c["out_free"][m].sum() / c["after_s"][m].sum().clamp(min=1e-6)), 4)}
    allm = torch.ones(len(sp), dtype=torch.bool)
    table = {"all": block(allm),
             "by_way": {k: block(torch.tensor([s["way"] == k for s in sp])) for k in DW.DIRS},
             "by_pace": {k: block(torch.tensor([s["pace"] == k for s in sp])) for k in DW.PACES},
             "by_contact": {k: block(torch.tensor([s["contact"] == k for s in sp])) for k in DW.CONTACT},
             "out_pushed": round(float(c["out_pushed"].sum() / c["after_s"].sum().clamp(min=1e-6)), 4),
             "end": rs.get("end"), "entropy": rs["entropy"], "bans_broken": rs["bans_broken"],
             "seconds": round(time.time() - t0, 1)}
    table["terrain"] = rs.get("terrain")         # (realmap.Meter: on a map only)
    if out_path:
        Path(out_path).write_text(json.dumps(table, indent=1), encoding="utf-8")
    print(json.dumps({"exam_withdraw": table}), flush=True)
    return table


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="a unit_v1 checkpoint (none: an untrained network, seed 0)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--gpu-duty", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--map", default=None, help="lay the skirmishes on this map (build/maps/<name>, tools/nn/unit/realmap.py): the exam's fixed places")
    a = ap.parse_args(argv)
    if a.map:
        from tools.nn.unit import realmap
        realmap.activate(a.map, exam=True)
    from tools.nn.unit import train as T
    if a.ckpt:
        net, _, _ = T.restore(a.ckpt, a.device)
    else:
        torch.manual_seed(0)
        net = Nt.UnitNet().to(a.device)
    return run(net, load(), a.device, T.Duty(a.device, a.gpu_duty), a.out)


if __name__ == "__main__":
    main()

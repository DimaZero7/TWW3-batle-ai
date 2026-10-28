"""The exam of drill U7 'a routing enemy' (drill_rout.py): the fixed set drill_rout.exam_specs() (240, breaking and
fleeing in turn), the network sampling with a fixed seed.

    bash tools/nn/dock.sh tools.nn.unit.exam_rout [--ckpt build/.../latest.pt] [--device cuda]

Metrics: routers cut down in the area - the enemy's health lost (share of its start) while it routs with its centre in
our area and our unit touches it (not the third unit's blows - crossing), and of all it lost so to us while routing the
share inside the area; routed at all (the breaking ones); the
leash's seconds a skirmish (standing at the edge facing a router beyond it); out of the area (own choice ~0; pushed);
the trade; by kind and by the area's size (6-7 / 8-10 cells).
"""
import argparse
import json
import time
from pathlib import Path

import torch

from tools.nn.sim.params import load
from tools.nn.train import reward
from tools.nn.unit import decode as Dc
from tools.nn.unit import drill_rout as DR
from tools.nn.unit import net as Nt

SEED = 4916


class Watch:
    def __init__(self, B, N, P, device):
        self.H, self.dt = N // 2, P.dt
        z = lambda: torch.zeros(B, device=device)       # noqa: E731
        self.rout_in, self.rout_all = z(), z()
        self.routed = torch.zeros(B, dtype=torch.bool, device=device)
        self.out_free, self.out_pushed, self.alive_s = z(), z(), z()
        self.prev = None

    def __call__(self, st, hp0, was_done, area, step):
        u = st.u
        H = self.H
        live = ~was_done
        if self.prev is None:
            self.prev = u["hp_abs"][:, H].clone()
        lost = (self.prev - u["hp_abs"][:, H]).clamp(min=0) / u["hp0"][:, H].clamp(min=1) * live.float()
        self.prev = u["hp_abs"][:, H].clone()
        a0 = {k: a[:, 0] for k, a in area.items() if k != "on"}
        routing = u["r"][:, H]
        self.routed = self.routed | (live & routing)
        in_area = Dc.signed_dist(u["x"][:, H], u["z"][:, H], a0) >= 0
        # ours only: the router's loss while our unit touches it (the pair's contact, state hit_sec either way); the
        # third unit's (crossing: it chases the router) is not ours
        N = st.N
        hs = u["hit_sec"].view(st.B, N, N)
        ours = (hs[:, 0, H] > 0) | (hs[:, H, 0] > 0)
        self.rout_all += lost * (routing & ours).float()
        self.rout_in += lost * (routing & in_area & ours).float()
        alive = live & (u["men"][:, 0] > 0)
        sd = Dc.signed_dist(u["x"][:, 0], u["z"][:, 0], a0)
        pushed = u["m"][:, 0] | u["r"][:, 0]
        self.alive_s += alive.float() * self.dt
        self.out_free += (alive & (sd < -0.5) & ~pushed).float() * self.dt
        self.out_pushed += (alive & (sd < -0.5) & pushed).float() * self.dt


def _mean(x, m):
    return round(float(x[m].float().mean()), 3) if bool(m.any()) else None


def run(net, P=None, device="cpu", duty=None, out_path=None, seed=SEED):
    from tools.nn.unit import train as T
    P = P or load()
    duty = duty or T.Duty(device, 1.0)
    sp = DR.exam_specs()
    cfg = {"battles": len(sp), "seconds": DR.LIMIT_S, "cell_m": 8.0, "bounds": None}
    torch.manual_seed(seed)
    setup = T.rout_setup(cfg, P, device, sp=sp)
    st = setup["st"]
    w = Watch(st.B, st.N, P, device)
    m0 = reward.measure(st)
    t0 = time.time()
    was = net.training
    net.eval()
    _, rs = T.rollout(net, None, cfg, P, device, duty, setup, watch=w)
    net.train(was)
    m1 = reward.measure(st)
    trade = ((m0[:, 1, 3] - m1[:, 1, 3]) - (m0[:, 0, 3] - m1[:, 0, 3])).cpu()
    c = {k: getattr(w, k).cpu() for k in ("rout_in", "rout_all", "routed", "out_free", "out_pushed", "alive_s")}

    def block(m):
        return {"n": int(m.sum()), "routed": _mean(c["routed"], m), "routers_cut_in_area": _mean(c["rout_in"], m),
                "routers_cut_all": _mean(c["rout_all"], m),
                "in_area_share": round(float(c["rout_in"][m].sum() / c["rout_all"][m].sum().clamp(min=1e-6)), 3),
                "trade": _mean(trade, m),
                "out_free": round(float(c["out_free"][m].sum() / c["alive_s"][m].sum().clamp(min=1e-6)), 4)}
    small = torch.tensor([min(s["w"], s["h"]) <= 7 for s in sp])
    table = {"all": block(torch.ones(len(sp), dtype=torch.bool)),
             "by_kind": {k: block(torch.tensor([s["kind"] == k for s in sp])) for k in DR.KINDS},
             "by_area": {"6-7": block(small), "8-10": block(~small)},
             "leash_s_per_skirmish": round(rs["leash_s"] / len(sp), 2),
             "out_pushed": round(float(c["out_pushed"].sum() / c["alive_s"].sum().clamp(min=1e-6)), 4),
             "end": rs.get("end"), "entropy": rs["entropy"], "bans_broken": rs["bans_broken"],
             "order_changes_per_skirmish": round(rs["order_changes"] / len(sp), 1), "seconds": round(time.time() - t0, 1)}
    if out_path:
        Path(out_path).write_text(json.dumps(table, indent=1), encoding="utf-8")
    print(json.dumps({"exam_rout": table}), flush=True)
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

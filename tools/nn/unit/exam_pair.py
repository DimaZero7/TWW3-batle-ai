"""The exam of drill U4 'friends beside' (drill_pair.py): the fixed set drill_pair.exam_specs() (240: 30 per layout x
enemy set), the network (both our units) sampling with a fixed seed.

    bash tools/nn/dock.sh tools.nn.unit.exam_pair [--ckpt build/.../latest.pt] [--device cuda]

Metrics: jostling - the share of the seconds (both standing) our two formations touch or overlap (geometry.rect_gap
<= 0); melee damage our units take to the front / the flank and rear (each step's loss of a unit split among the
enemies in contact with it by the sector each is in: within contact.front_deg 45 deg of its facing the front, beyond
rear_deg 135 the rear); met head-on (at each unit's first contact the enemy in its front sector); the pair's trade; out
of the area (own choice / pushed; should be ~0); by layout and enemy set.
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch

from tools.nn.sim import geometry
from tools.nn.sim.params import load
from tools.nn.train import reward
from tools.nn.unit import decode as Dc
from tools.nn.unit import drill_pair as DP
from tools.nn.unit import net as Nt

SEED = 4908


class Watch:
    def __init__(self, B, N, P, device):
        self.N, self.dt = N, P.dt
        self.spacing = P.sim["formation"]["spacing_m"]
        cc = P.sim["contact"]
        self.front, self.rear = math.radians(cc["front_deg"]), math.radians(cc["rear_deg"])
        z = lambda: torch.zeros(B, device=device)       # noqa: E731
        self.prev = None
        self.first_sec = torch.full((B, 2), -1, dtype=torch.long, device=device)
        self.dmg_front, self.dmg_side, self.dmg_all = z(), z(), z()
        self.touch_s, self.both_s = z(), z()
        self.out_free, self.out_pushed, self.alive_s = z(), z(), z()

    def __call__(self, st, hp0, was_done, area, step):
        u = st.u
        N = self.N
        live = ~was_done
        if self.prev is None:
            self.prev = u["hp_abs"].clone()
        lost = (self.prev - u["hp_abs"]).clamp(min=0)
        self.prev = u["hp_abs"].clone()
        hs = u["hit_sec"].view(st.B, N, N)                    # [B, j, i]: j's contact on i
        dx = u["x"][:, None, :] - u["x"][:, :, None]          # [B, i, j]: j seen from i
        dz = u["z"][:, None, :] - u["z"][:, :, None]
        rel = torch.remainder(torch.atan2(dx, dz) - torch.deg2rad(u["b"])[:, :, None] + math.pi, 2 * math.pi) - math.pi
        sec = torch.where(rel.abs() <= self.front, 0, torch.where(rel.abs() >= self.rear, 2, 1))   # [B, i, j]
        foe = (u["side"] == 2)[:, None, :]
        for i in (0, 1):
            on = (hs[:, :, i] > 0) & foe[:, 0]                # [B, j] enemies in contact with our unit i
            n = on.float().sum(1)
            fr = ((sec[:, i] == 0) & on).float().sum(1) / n.clamp(min=1)
            li = lost[:, i] * (live & (n > 0)).float()
            self.dmg_all += li
            self.dmg_front += li * fr
            self.dmg_side += li * (1 - fr)
            j = torch.where(on, sec[:, i], torch.full_like(sec[:, i], 9)).argmin(1)
            first = live & (n > 0) & (self.first_sec[:, i] < 0)
            self.first_sec[:, i] = torch.where(first, sec[:, i].gather(1, j[:, None])[:, 0], self.first_sec[:, i])
        up = (u["men"][:, :2] > 0) & ~u["gone"][:, :2] & ~u["r"][:, :2]
        both = live & up.all(1)
        r = geometry.rect_gap(u["x"][:, :2], u["z"][:, :2], u["b"][:, :2], *_dims(u, self.spacing))
        self.both_s += both.float() * self.dt
        self.touch_s += (both & (r[:, 0, 1] <= 0)).float() * self.dt
        sd = Dc.signed_dist(u["x"][:, :2], u["z"][:, :2], {k: a[:, :2] for k, a in area.items() if k != "on"})
        alive = live[:, None] & (u["men"][:, :2] > 0)
        pushed = u["m"][:, :2] | u["r"][:, :2]
        self.alive_s += alive.float().sum(1) * self.dt
        self.out_free += (alive & (sd < -0.5) & ~pushed).float().sum(1) * self.dt
        self.out_pushed += (alive & (sd < -0.5) & pushed).float().sum(1) * self.dt


def _dims(u, spacing):
    f, d = geometry.dims({k: v[:, :2] for k, v in u.items() if v.dim() == 2 and v.shape[1] == u["x"].shape[1]}, spacing)
    return f, d


def _mean(x, m):
    return round(float(x[m].float().mean()), 3) if bool(m.any()) else None


def run(net, P=None, device="cpu", duty=None, out_path=None, seed=SEED):
    from tools.nn.unit import train as T
    P = P or load()
    duty = duty or T.Duty(device, 1.0)
    sp = DP.exam_specs()
    cfg = {"battles": len(sp), "seconds": DP.LIMIT_S, "cell_m": 8.0, "bounds": None}
    torch.manual_seed(seed)
    setup = T.pair_setup(cfg, P, device, sp=sp)
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
    fs = w.first_sec.cpu()

    def block(m):
        da = w.dmg_all.cpu()[m].sum().clamp(min=1e-6)
        met = fs[m] >= 0
        return {"n": int(m.sum()),
                "jostle": round(float(w.touch_s.cpu()[m].sum() / w.both_s.cpu()[m].sum().clamp(min=1e-6)), 3),
                "dmg_front": round(float(w.dmg_front.cpu()[m].sum() / da), 3),
                "dmg_flank_rear": round(float(w.dmg_side.cpu()[m].sum() / da), 3),
                "head_on": round(float((fs[m][met] == 0).float().mean()), 3) if bool(met.any()) else None,
                "out_free": round(float(w.out_free.cpu()[m].sum() / w.alive_s.cpu()[m].sum().clamp(min=1e-6)), 4),
                "trade": _mean(trade, m)}
    allm = torch.ones(len(sp), dtype=torch.bool)
    table = {"all": block(allm),
             "by_layout": {k: block(torch.tensor([s["layout"] == k for s in sp])) for k in DP.LAYOUTS},
             "by_enemies": {k: block(torch.tensor([s["enemies"] == k for s in sp])) for k in DP.ENEMY_SETS},
             "out_pushed": round(float(w.out_pushed.sum() / w.alive_s.sum().clamp(min=1e-6)), 4),
             "end": rs.get("end"), "duration_s": round(float(st.t.mean()), 1), "entropy": rs["entropy"],
             "bans_broken": rs["bans_broken"], "seconds": round(time.time() - t0, 1)}
    table["terrain"] = rs.get("terrain")         # (realmap.Meter: on a map only)
    if out_path:
        Path(out_path).write_text(json.dumps(table, indent=1), encoding="utf-8")
    print(json.dumps({"exam_pair": table}), flush=True)
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

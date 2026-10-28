"""The exam of drill U5 'shooters - fire at one target' (drill_shoot.py): the fixed set drill_shoot.exam_specs() (240,
'at' and 'past' in turn), the network sampling with a fixed seed.

    bash tools/nn/dock.sh tools.nn.unit.exam_shoot [--ckpt build/.../latest.pt] [--device cuda]

Metrics: fire time - the share of the seconds the enemy is within our range (centre to centre, standing, seen) that
our unit fires (state fire); the enemy in the sector - the share of the seconds the enemy is up that its centre lies in
our fire sector; damage by fire - the enemy's health lost (share) while out of melee with us; of it before the first
contact; our health lost (share) in melee; the trade; out of the position (own choice / pushed; ~0); by mode at / past.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from tools.nn.sim.params import load
from tools.nn.train import reward
from tools.nn.unit import decode as Dc
from tools.nn.unit import drill_shoot as DS
from tools.nn.unit import net as Nt

SEED = 4910


class Watch:
    def __init__(self, B, N, P, device):
        self.H, self.N, self.dt = N // 2, N, P.dt
        z = lambda: torch.zeros(B, device=device)       # noqa: E731
        self.in_range_s, self.fire_s, self.up_s, self.sector_s = z(), z(), z(), z()
        self.fire_dmg, self.fire_dmg_pre, self.melee_lost = z(), z(), z()
        self.contacted = torch.zeros(B, dtype=torch.bool, device=device)
        self.out_free, self.out_pushed, self.alive_s = z(), z(), z()
        self.prev = None

    def __call__(self, st, hp0, was_done, area, step):
        u = st.u
        H, N = self.H, self.N
        live = ~was_done
        if self.prev is None:
            self.prev = u["hp_abs"].clone()
        lost = (self.prev - u["hp_abs"]).clamp(min=0) / u["hp0"].clamp(min=1)
        self.prev = u["hp_abs"].clone()
        hs = u["hit_sec"].view(st.B, N, N)
        contact = (hs[:, H, 0] > 0) | (hs[:, 0, H] > 0)
        self.contacted = self.contacted | (live & contact)
        up = live & (u["men"][:, H] > 0) & ~u["gone"][:, H] & ~u["r"][:, H]
        dx, dz = u["x"][:, H] - u["x"][:, 0], u["z"][:, H] - u["z"][:, 0]
        dist = torch.sqrt(dx * dx + dz * dz)
        in_range = up & u["vis"][:, H] & (dist <= u["range"][:, 0]) & (u["men"][:, 0] > 0) & ~u["r"][:, 0]
        way = torch.rad2deg(torch.atan2(dx, dz))
        off = (torch.remainder(way - area["sb"][:, 0] + 180, 360) - 180).abs()
        self.up_s += up.float() * self.dt
        self.sector_s += (up & (off <= area["sw"][:, 0] / 2)).float() * self.dt
        self.in_range_s += in_range.float() * self.dt
        self.fire_s += (in_range & u["fire"][:, 0]).float() * self.dt
        fd = lost[:, H] * (live & ~contact).float()
        self.fire_dmg += fd
        self.fire_dmg_pre += fd * (~self.contacted).float()
        self.melee_lost += lost[:, 0] * (live & contact).float()
        alive = live & (u["men"][:, 0] > 0)
        sd = Dc.signed_dist(u["x"][:, 0], u["z"][:, 0], {k: a[:, 0] for k, a in area.items() if k != "on"})
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
    sp = DS.exam_specs()
    cfg = {"battles": len(sp), "seconds": DS.LIMIT_S, "cell_m": 8.0, "bounds": None}
    torch.manual_seed(seed)
    setup = T.shoot_setup(cfg, P, device, sp=sp)
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
    c = {k: getattr(w, k).cpu() for k in ("in_range_s", "fire_s", "up_s", "sector_s", "fire_dmg", "fire_dmg_pre",
                                          "melee_lost", "out_free", "out_pushed", "alive_s")}
    contacted = w.contacted.cpu()

    def block(m):
        return {"n": int(m.sum()),
                "fire_time": round(float(c["fire_s"][m].sum() / c["in_range_s"][m].sum().clamp(min=1e-6)), 3),
                "enemy_in_sector": round(float(c["sector_s"][m].sum() / c["up_s"][m].sum().clamp(min=1e-6)), 3),
                "fire_damage": _mean(c["fire_dmg"], m), "fire_damage_before_contact": _mean(c["fire_dmg_pre"], m),
                "contact": _mean(contacted, m), "melee_lost": _mean(c["melee_lost"], m), "trade": _mean(trade, m),
                "out_free": round(float(c["out_free"][m].sum() / c["alive_s"][m].sum().clamp(min=1e-6)), 4)}
    allm = torch.ones(len(sp), dtype=torch.bool)
    table = {"all": block(allm),
             "by_mode": {k: block(torch.tensor([s["mode"] == k for s in sp])) for k in DS.MODES},
             "by_side": {k: block(torch.tensor([s["ours"] == k for s in sp])) for k in ("emp", "skv")},
             "out_pushed": round(float(c["out_pushed"].sum() / c["alive_s"].sum().clamp(min=1e-6)), 4),
             "leash_s": rs["leash_s"], "end": rs.get("end"), "duration_s": round(float(st.t.mean()), 1),
             "entropy": rs["entropy"], "bans_broken": rs["bans_broken"], "seconds": round(time.time() - t0, 1)}
    if out_path:
        Path(out_path).write_text(json.dumps(table, indent=1), encoding="utf-8")
    print(json.dumps({"exam_shoot": table}), flush=True)
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

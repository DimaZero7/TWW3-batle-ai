"""The exam of drill U5b 'a clean line of fire' (drill_shoot2.py): the fixed set drill_shoot2.exam_specs() (240, the two
layouts in turn), the network (both our units) sampling with a fixed seed.

    bash tools/nn/dock.sh tools.nn.unit.exam_shoot2 [--ckpt build/.../latest.pt] [--device cuda]

Metrics (our shooters - the units with a fire sector): fire time - the share of the seconds an enemy stands within a
shooter's range (centre to centre, seen) that it fires; fire lost to friends - of the fire it could have had at its
target, the share friends took away (the simulator's bookkeeping fc_skip / fc_cap: held fire for a closed line, the men
whose path a friend blocks); damage by fire - the enemies' health lost (share) while none of ours touches them; our
losses out of melee while our shooters fire - the friendly fire's upper bound (the shots that catch friends between;
friendly fire into a melee is not told apart from the melee); the side's trade; out of the position / area (own choice,
~0; apart: walking back in - a move whose point lies in the area, after a fight carried the unit out); by layout and by
the shooters' kind (direct / arcing fire).
"""
import argparse
import json
import time
from pathlib import Path

import torch

from tools.nn.sim.params import load
from tools.nn.train import reward
from tools.nn.unit import decode as Dc
from tools.nn.unit import drill_shoot2 as D2
from tools.nn.unit import net as Nt

SEED = 4914


class Watch:
    def __init__(self, B, N, P, device):
        self.N, self.dt = N, P.dt
        z = lambda: torch.zeros(B, device=device)       # noqa: E731
        self.in_range_s, self.fire_s, self.cap, self.skip = z(), z(), z(), z()
        self.fire_dmg, self.ff_lost = z(), z()
        self.out_free, self.out_back, self.alive_s = z(), z(), z()
        self.prev = None

    def __call__(self, st, hp0, was_done, area, step):
        u = st.u
        N = self.N
        live = ~was_done
        if self.prev is None:
            self.prev = u["hp_abs"].clone()
        lost = (self.prev - u["hp_abs"]).clamp(min=0) / u["hp0"].clamp(min=1) * live[:, None].float()
        self.prev = u["hp_abs"].clone()
        hs = u["hit_sec"].view(st.B, N, N)
        touch = (hs > 0) | (hs.transpose(1, 2) > 0)                     # [B, i, j] in contact
        ours = torch.zeros_like(u["side"], dtype=torch.bool)
        ours[:, :2] = True
        foe = u["side"] == 2
        shooter = ours & (area["sw"] > 0)
        dx = u["x"][:, None, :] - u["x"][:, :, None]
        dz = u["z"][:, None, :] - u["z"][:, :, None]
        dist = torch.sqrt(dx * dx + dz * dz)                             # [B, i, j]
        foe_up = foe & (u["men"] > 0) & ~u["gone"] & ~u["r"] & u["vis"]
        in_range = shooter & (u["men"] > 0) & ~u["r"] & ((dist <= u["range"][:, :, None]) & foe_up[:, None, :]).any(2)
        self.in_range_s += (in_range & live[:, None]).float().sum(1) * self.dt
        self.fire_s += (in_range & u["fire"] & live[:, None]).float().sum(1) * self.dt
        self.cap += (u["fc_cap"] * shooter).sum(1) * live.float()
        self.skip += (u["fc_skip"] * shooter).sum(1) * live.float()
        foe_free = foe & ~(touch & ours[:, :, None]).any(1)
        self.fire_dmg += (lost * foe_free).sum(1)
        firing = (shooter & u["fire"]).any(1)
        ours_free = ours & ~(touch & foe[:, None, :]).any(2)
        self.ff_lost += (lost * ours_free).sum(1) * firing.float()
        a2 = {k: a[:, :2] for k, a in area.items() if k in ("x", "z", "b", "front", "depth")}
        sd = Dc.signed_dist(u["x"][:, :2], u["z"][:, :2], a2)
        alive = live[:, None] & (u["men"][:, :2] > 0)
        pushed = u["m"][:, :2] | u["r"][:, :2]
        # walking back in: a move order whose point lies in its area (after a fight pushed or carried it out)
        back = (u["order_kind"][:, :2] == 1) & (Dc.signed_dist(u["ox"][:, :2], u["oz"][:, :2], a2) >= 0)
        self.alive_s += alive.float().sum(1) * self.dt
        self.out_free += (alive & (sd < -0.5) & ~pushed & ~back).float().sum(1) * self.dt
        self.out_back += (alive & (sd < -0.5) & ~pushed & back).float().sum(1) * self.dt


def _mean(x, m):
    return round(float(x[m].float().mean()), 3) if bool(m.any()) else None


def run(net, P=None, device="cpu", duty=None, out_path=None, seed=SEED):
    from tools.nn.unit import train as T
    P = P or load()
    duty = duty or T.Duty(device, 1.0)
    sp = D2.exam_specs()
    cfg = {"battles": len(sp), "seconds": D2.LIMIT_S, "cell_m": 8.0, "bounds": None}
    torch.manual_seed(seed)
    setup = T.shoot2_setup(cfg, P, device, sp=sp)
    st = setup["st"]
    direct = (st.u["direct"][:, :2] & (setup["area"]["sw"][:, :2] > 0)).any(1).cpu()
    w = Watch(st.B, st.N, P, device)
    m0 = reward.measure(st)
    t0 = time.time()
    was = net.training
    net.eval()
    _, rs = T.rollout(net, None, cfg, P, device, duty, setup, watch=w)
    net.train(was)
    m1 = reward.measure(st)
    trade = ((m0[:, 1, 3] - m1[:, 1, 3]) - (m0[:, 0, 3] - m1[:, 0, 3])).cpu()
    c = {k: getattr(w, k).cpu() for k in ("in_range_s", "fire_s", "cap", "skip", "fire_dmg", "ff_lost", "out_free",
                                          "out_back", "alive_s")}

    def block(m):
        return {"n": int(m.sum()),
                "fire_time": round(float(c["fire_s"][m].sum() / c["in_range_s"][m].sum().clamp(min=1e-6)), 3),
                "fire_lost_to_friends": round(float(c["skip"][m].sum() / c["cap"][m].sum().clamp(min=1e-6)), 3),
                "fire_damage": _mean(c["fire_dmg"], m), "own_lost_out_of_melee_while_firing": _mean(c["ff_lost"], m),
                "trade": _mean(trade, m),
                "out_free": round(float(c["out_free"][m].sum() / c["alive_s"][m].sum().clamp(min=1e-6)), 4),
                "out_walking_back": round(float(c["out_back"][m].sum() / c["alive_s"][m].sum().clamp(min=1e-6)), 4)}
    allm = torch.ones(len(sp), dtype=torch.bool)
    table = {"all": block(allm),
             "by_sub": {k: block(torch.tensor([s["sub"] == k for s in sp])) for k in ("side", "file", "screen")},
             "by_fire": {"direct": block(direct), "arcing": block(~direct)},
             "end": rs.get("end"), "entropy": rs["entropy"], "bans_broken": rs["bans_broken"],
             "order_changes_per_skirmish": round(rs["order_changes"] / len(sp), 1), "seconds": round(time.time() - t0, 1)}
    table["terrain"] = rs.get("terrain")         # (realmap.Meter: on a map only)
    if out_path:
        Path(out_path).write_text(json.dumps(table, indent=1), encoding="utf-8")
    print(json.dumps({"exam_shoot2": table}), flush=True)
    return table


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="a unit_v1 checkpoint (none: an untrained network, seed 0)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--gpu-duty", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--map", default=None, help="lay the skirmishes on these maps (a name, a,b or all - build/maps/<name>, tools/nn/unit/realmap.py; several: in turn): the exam's fixed places")
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

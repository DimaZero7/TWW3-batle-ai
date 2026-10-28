"""The exam of drill U8 'a priority target' (drill_prio.py): the fixed set drill_prio.exam_specs() (240: 40 per layout x
priority - duel: the enemy shooter / the infantry / none; two: A / B / none), the network sampling with a fixed seed.

    bash tools/nn/dock.sh tools.nn.unit.exam_prio [--ckpt build/.../latest.pt] [--device cuda]

Metrics: obedience - the share of the decisions (one a second) while the named target is allowed (its row's
target_ok: infantry - its centre in our area; a shooter - in our sector and range, the shooter itself not in melee and
the target not routing: train.obedience_parts) that the network's target is it, also split by our unit in melee / out
of it; the main one (goal >= 0.9) obedience_contested - the same over the decisions with another enemy allowed too (a
real choice; with the named one the only target allowed it is taken ~always: 0.99 in the duel); obedience_old - the
rule before (no melee and rout exceptions), for the comparison; hits at the priority - the
share of the seconds our unit fights or fires (in melee or firing) that its current target (the simulator's `target`)
is the priority; the priority down (dead, gone or routing) - the share and the time (s, of those down); its health
lost; each enemy's time down (the slots H, H + 1: the duel's shooter / infantry, U3's A / B) - against the cells with
no priority; our health lost; the trade; out of the area (own choice; ~0). By cell (layout x priority) and over all
skirmishes with a priority / without one ('none': the network chooses itself - its trade should not fall).
"""
import argparse
import json
import time
from pathlib import Path

import torch

from tools.nn.sim.params import load
from tools.nn.train import reward
from tools.nn.unit import decode as Dc
from tools.nn.unit import drill_prio as DQ
from tools.nn.unit import net as Nt

SEED = 4916


class Watch:
    def __init__(self, B, N, P, device, prio):
        self.H, self.N, self.dt = N // 2, N, P.dt
        self.p = prio[:, 0]                                                    # [B] the priority's slot, -1 none
        z = lambda: torch.zeros(B, device=device)       # noqa: E731
        self.busy_s, self.at_prio_s, self.out_free, self.alive_s = z(), z(), z(), z()
        self.busy_on = torch.zeros((B, 2), device=device)                      # busy with H / H + 1
        self.down_t = torch.full((B, 2), -1.0, device=device)                  # H / H + 1 first down (s)

    def __call__(self, st, hp0, was_done, area, step):
        u = st.u
        H = self.H
        live = ~was_done
        foe = slice(H, H + 2)
        down = (u["men"][:, foe] <= 0) | u["gone"][:, foe] | u["r"][:, foe]
        self.down_t = torch.where(live[:, None] & down & (self.down_t < 0), st.t[:, None].expand(-1, 2), self.down_t)
        me_up = live & (u["men"][:, 0] > 0) & ~u["r"][:, 0]
        busy = me_up & (u["m"][:, 0] | u["fire"][:, 0])
        tg = u["target"][:, 0]
        for k in range(2):
            self.busy_on[:, k] += (busy & (tg == H + k)).float() * self.dt
        p = self.p.clamp(min=0)
        p_up = (self.p >= 0) & (u["men"].gather(1, p[:, None])[:, 0] > 0) & ~u["gone"].gather(1, p[:, None])[:, 0] & \
            ~u["r"].gather(1, p[:, None])[:, 0]
        self.busy_s += (busy & p_up).float() * self.dt
        self.at_prio_s += (busy & p_up & (tg == self.p)).float() * self.dt
        alive = live & (u["men"][:, 0] > 0)
        sd = Dc.signed_dist(u["x"][:, 0], u["z"][:, 0], {k: a[:, 0] for k, a in area.items()
                                                          if k in ("x", "z", "b", "front", "depth")})
        pushed = u["m"][:, 0] | u["r"][:, 0]
        self.alive_s += alive.float() * self.dt
        self.out_free += (alive & (sd < -0.5) & ~pushed).float() * self.dt


def _mean(x, m):
    return round(float(x[m].float().mean()), 3) if bool(m.any()) else None


def _share(a, b, m):
    s = float(b[m].sum())
    return round(float(a[m].sum()) / s, 3) if s > 0 else None


def run(net, P=None, device="cpu", duty=None, out_path=None, seed=SEED):
    from tools.nn.unit import train as T
    P = P or load()
    duty = duty or T.Duty(device, 1.0)
    sp = DQ.exam_specs()
    cfg = {"battles": len(sp), "seconds": DQ.LIMIT_S, "cell_m": 8.0, "bounds": None}
    torch.manual_seed(seed)
    setup = T.prio_setup(cfg, P, device, sp=sp)
    st = setup["st"]
    H = st.N // 2
    w = Watch(st.B, st.N, P, device, setup["prio"])
    m0 = reward.measure(st)
    hp_start = st.u["hp_abs"].clone()
    t0 = time.time()
    was = net.training
    net.eval()
    _, rs = T.rollout(net, None, cfg, P, device, duty, setup, watch=w)
    net.train(was)
    m1 = reward.measure(st)
    trade = ((m0[:, 1, 3] - m1[:, 1, 3]) - (m0[:, 0, 3] - m1[:, 0, 3])).cpu()
    lost = ((hp_start - st.u["hp_abs"]).clamp(min=0) / st.u["hp0"].clamp(min=1)).cpu()          # [B, N]
    pr = {k: v.cpu() for k, v in rs.pop("_prio").items()}
    p = setup["prio"][:, 0].cpu()
    pk = (p - H).clamp(min=0)
    down_t = w.down_t.cpu()
    p_down_t = down_t.gather(1, pk[:, None])[:, 0]
    p_lost = lost.gather(1, p.clamp(min=0)[:, None])[:, 0]
    c = {k: getattr(w, k).cpu() for k in ("busy_s", "at_prio_s", "out_free", "alive_s")}
    busy_on = w.busy_on.cpu()
    given = p >= 0
    names = lambda lay: DQ.PRIOS[lay]                                         # noqa: E731

    def block(m, lay=None):
        g = m & given
        dn = g & (p_down_t >= 0)
        out = {"n": int(m.sum()),
               "obedience_contested": _share(pr["obeyed_contested_s"], pr["contested_s"], g),
               "obedience": _share(pr["obeyed_s"], pr["allowed_s"], g),
               "obedience_old": _share(pr["obeyed_old_s"], pr["allowed_old_s"], g),
               "contested_s": _mean(pr["contested_s"], g),
               "obedience_melee": _share(pr["obeyed_melee_s"], pr["allowed_melee_s"], g),
               "obedience_free": _share(pr["obeyed_s"] - pr["obeyed_melee_s"], pr["allowed_s"] - pr["allowed_melee_s"], g),
               "allowed_s": _mean(pr["allowed_s"], g),
               "hits_at_prio": _share(c["at_prio_s"], c["busy_s"], g),
               "prio_down": _mean(p_down_t >= 0, g), "prio_down_t": _mean(p_down_t, dn), "prio_hp_lost": _mean(p_lost, g),
               "our_hp_lost": _mean(lost[:, 0], m), "trade": _mean(trade, m),
               "out_free": round(float(c["out_free"][m].sum() / c["alive_s"][m].sum().clamp(min=1e-6)), 4)}
        if lay is not None:
            # each enemy: down (share, time of those down) and the share of our busy time on it
            for k, nm in enumerate(names(lay)):
                out[f"{nm}_down"] = _mean(down_t[:, k] >= 0, m)
                out[f"{nm}_down_t"] = _mean(down_t[:, k], m & (down_t[:, k] >= 0))
                out[f"{nm}_hp_lost"] = _mean(lost[:, H + k], m)
                out[f"busy_on_{nm}"] = _share(busy_on[:, k], busy_on.sum(1), m)
        return out
    lay_of = [s["layout"] for s in sp]
    cells = {}
    for lay, pq in DQ.EXAM_CELLS:
        m = torch.tensor([lay_of[i] == lay and sp[i]["prio"] == pq for i in range(len(sp))])
        cells[f"{lay}/{pq or 'none'}"] = block(m, lay)
    table = {"given": block(given), "none": block(~given), "cells": cells,
             "end": rs.get("end"), "duration_s": round(float(st.t.mean()), 1), "entropy": rs["entropy"],
             "bans_broken": rs["bans_broken"], "order_changes_per_skirmish": round(rs["order_changes"] / len(sp), 1),
             "seconds": round(time.time() - t0, 1)}
    table["terrain"] = rs.get("terrain")         # (realmap.Meter: on a map only)
    if out_path:
        Path(out_path).write_text(json.dumps(table, indent=1), encoding="utf-8")
    print(json.dumps({"exam_prio": table}), flush=True)
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

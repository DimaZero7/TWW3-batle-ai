"""The exam of drill U1 'hold the line' (drill_hold.py): the fixed set drill_hold.exam_specs() (every Empire x Skaven
melee infantry pair, both ways, the enemy coming from the front / the flank / the rear), the network sampling with a
fixed seed - the same skirmishes for every measurement.

    bash tools/nn/dock.sh tools.nn.unit.exam_hold [--ckpt build/.../latest.pt] [--device cuda --gpu-duty 0.7]

Metrics (more is better unless said):
* met head-on: at the contact (the first step the enemy's contact with us is on, state hit_sec) the enemy's centre lies
  in our front sector (within contact.front_deg 45 deg of our facing; the flank to rear_deg 135, the rear beyond - the
  simulator's melee sectors, geometry.sector); also the game's label (the side the contact began on, hit_sec); by the
  start (face / flank / rear) and by 'had time to turn' (contact later than the facing order's re-form + turn: 3.9 s +
  the angle at 80 deg/s; no turn needed from the front);
* melee damage to the front: our health lost in contact while the enemy's centre is in our front sector / all of it;
* out of the area (should be ~0): seconds our centre is over 0.5 m out of its area, of its own choice (not in melee, not
  routing) and pushed (in melee or routing) - shares of the alive seconds;
* trade: (enemy gold lost - ours) / budget per skirmish, by pair (the safety: should not fall);
* the choice's entropy per head; how the skirmishes ended (we routed / the enemy left / the 3 min limit);
* the branches (branch.py, every skirmish): met the charge standing braced (HOLD, still, the enemy within 80 deg at the
  contact), attacked, chose the better branch; the share of skirmishes / pairs where standing is better.
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
from tools.nn.unit import drill_hold as DH
from tools.nn.unit import net as Nt

SEED = 4902


class Watch:
    def __init__(self, B, N, P, device):
        self.H = N // 2
        self.N = N
        self.dt = P.dt
        cc = P.sim["contact"]
        self.front, self.rear = math.radians(cc["front_deg"]), math.radians(cc["rear_deg"])
        z = lambda: torch.zeros(B, device=device)       # noqa: E731
        self.contact_t = torch.full((B,), -1.0, device=device)
        self.contact_sec = torch.full((B,), -1, dtype=torch.long, device=device)
        self.contact_game = torch.full((B,), -1, dtype=torch.long, device=device)
        self.dmg_front, self.dmg_all = z(), z()
        self.out_free, self.out_pushed, self.alive_s = z(), z(), z()

    def sector(self, u):
        dx, dz = u["x"][:, self.H] - u["x"][:, 0], u["z"][:, self.H] - u["z"][:, 0]
        rel = torch.remainder(torch.atan2(dx, dz) - torch.deg2rad(u["b"][:, 0]) + math.pi, 2 * math.pi) - math.pi
        a = rel.abs()
        return torch.where(a <= self.front, 0, torch.where(a >= self.rear, 2, 1))

    def __call__(self, st, hp0, was_done, area, step):
        u = st.u
        live = ~was_done
        hs = u["hit_sec"][:, self.H * self.N + 0]                 # the enemy's contact on us: the side + 1, 0 none
        sec = self.sector(u)
        first = live & (hs > 0) & (self.contact_t < 0)
        self.contact_t = torch.where(first, st.t, self.contact_t)
        self.contact_sec = torch.where(first, sec, self.contact_sec)
        self.contact_game = torch.where(first, hs.long() - 1, self.contact_game)
        lost = (hp0 - u["hp_abs"][:, 0]).clamp(min=0) * (live & (hs > 0)).float()
        self.dmg_all += lost
        self.dmg_front += lost * (sec == 0).float()
        alive = live & (u["men"][:, 0] > 0)
        out = Dc.signed_dist(u["x"][:, 0], u["z"][:, 0], {k: a[:, 0] for k, a in area.items()}) < -0.5
        pushed = u["m"][:, 0] | u["r"][:, 0]
        self.alive_s += alive.float() * self.dt
        self.out_free += (alive & out & ~pushed).float() * self.dt
        self.out_pushed += (alive & out & pushed).float() * self.dt


def _share(x, m):
    n = int(m.sum())
    return (round(float(x[m].float().mean()), 3) if n else None), n


def run(net, P=None, device="cpu", duty=None, out_path=None, seed=SEED):
    from tools.nn.unit import train as T
    P = P or load()
    duty = duty or T.Duty(device, 1.0)
    sp = DH.exam_specs()
    cfg = {"battles": len(sp), "seconds": DH.LIMIT_S, "cell_m": 8.0, "bounds": None}
    torch.manual_seed(seed)
    setup = T.hold_setup(cfg, P, device, sp=sp)
    st = setup["st"]
    w = Watch(st.B, st.N, P, device)
    m0 = reward.measure(st)
    t0 = time.time()
    was = net.training
    net.eval()
    from tools.nn.unit import branch as Br
    _, rs = T.rollout(net, None, cfg, P, device, duty, setup, watch=w,
                      branch={"share": 1.0, "weight": 0.0, "lead_s": Br.LEAD_S, "horizon_s": Br.HORIZON_S})
    bi = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in rs.pop("_branch").items()}
    stand_braced = (rs.get("branch") or {}).get("stand_branch_braced")
    net.train(was)
    m1 = reward.measure(st)
    trade = ((m0[:, 1, 3] - m1[:, 1, 3]) - (m0[:, 0, 3] - m1[:, 0, 3])).cpu()
    starts = [s["start"] for s in sp]
    rel = torch.tensor([min(s["rel"], 360 - s["rel"]) for s in sp])
    need = torch.where(rel <= 45, torch.zeros_like(rel), 3.9 + rel / 80.0)
    ct, cs, cg = w.contact_t.cpu(), w.contact_sec.cpu(), w.contact_game.cpu()
    met = ct >= 0
    had = met & (ct >= need)
    table = {"by_start": {}, "by_time": {}}
    beh, best, br = bi["behaviour"], bi.get("best", torch.full((len(sp),), -1)), bi["branched"]
    okb = br & (bi["contact_d"] >= 0)
    for name in DH.STARTS:
        m = torch.tensor([s == name for s in starts])
        table["by_start"][name] = {"n": int(m.sum()), "contact": _share(met, m)[0],
                                   "front_at_contact": _share(cs == 0, m & met)[0],
                                   "front_game_label": _share(cg == 0, m & met)[0],
                                   "had_time": _share(had, m & met)[0],
                                   "braced_at_contact": _share(beh == Br.STAND, m & okb)[0],
                                   "attacked": _share(beh == Br.ATTACK, m & okb)[0],
                                   "chose_best": _share(beh == best, m & okb)[0],
                                   "stand_better": _share(best == Br.STAND, m & br)[0],
                                   "melee_dmg_front": round(float(w.dmg_front.cpu()[m].sum() / w.dmg_all.cpu()[m].sum().clamp(min=1e-6)), 3)}
    for name, m in (("had_time", had), ("no_time", met & ~had)):
        table["by_time"][name] = {"n": int(m.sum()), "front_at_contact": _share(cs == 0, m)[0]}
    alive = float(w.alive_s.sum())
    pairs = {}
    for s, t in zip(sp, trade.tolist()):
        pairs.setdefault(f'{s["me"]} vs {s["foe"]}', []).append(t)
    pairs = {k: round(sum(v) / len(v), 3) for k, v in pairs.items()}
    stand_pairs = {}
    for s_, b_, ok_ in zip(sp, best.tolist(), br.tolist()):
        if ok_:
            stand_pairs.setdefault(f'{s_["me"]} vs {s_["foe"]}', []).append(float(b_ == Br.STAND))
    stand_pairs = {k: round(sum(v) / len(v), 3) for k, v in stand_pairs.items()}
    table.update({
        "skirmishes": len(sp), "contact": round(float(met.float().mean()), 3),
        "melee_dmg_front": round(float(w.dmg_front.sum() / w.dmg_all.sum().clamp(min=1e-6)), 3),
        "out_of_area_free": round(float(w.out_free.sum()) / max(alive, 1e-6), 4),
        "out_of_area_pushed": round(float(w.out_pushed.sum()) / max(alive, 1e-6), 4),
        "trade_mean": round(float(trade.mean()), 3),
        "trade_emp": round(float(trade[torch.tensor([s["ours"] == "emp" for s in sp])].mean()), 3),
        "trade_skv": round(float(trade[torch.tensor([s["ours"] == "skv" for s in sp])].mean()), 3),
        "end": rs.get("end"), "duration_s": round(float(st.t.mean()), 1),
        "contact_t_s": round(float(ct[met].mean()), 1) if bool(met.any()) else None,
        "entropy": rs["entropy"], "bans_broken": rs["bans_broken"], "leash_s": rs["leash_s"],
        "branch": {"branched": round(float(br.float().mean()), 3),
                   "braced_at_contact": _share(beh == Br.STAND, okb)[0], "attacked": _share(beh == Br.ATTACK, okb)[0],
                   "chose_best": _share(beh == best, okb)[0], "stand_better": _share(best == Br.STAND, br)[0],
                   "pairs_stand_mostly_better": round(sum(v > 0.5 for v in stand_pairs.values()) / max(len(stand_pairs), 1), 3),
                   "stand_branch_braced": stand_braced, "diff_mean": round(float((bi["t_stand"] - bi["t_attack"])[br].abs().mean()), 4) if bool(br.any()) else None},
        "seconds": round(time.time() - t0, 1), "pairs": pairs, "pairs_stand_better": stand_pairs})
    table["terrain"] = rs.get("terrain")         # (realmap.Meter: on a map only)
    if out_path:
        Path(out_path).write_text(json.dumps(table, indent=1), encoding="utf-8")
    short = {k: v for k, v in table.items() if k not in ("pairs", "pairs_stand_better")}
    print(json.dumps({"exam": short}), flush=True)
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

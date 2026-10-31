"""The exam of drill U5c 'the shooter's width' (drill_shootw.py): the fixed set drill_shootw.exam_specs() (240: gap /
angle / open in turn), the network sampling with a fixed seed - or a script in its place (--script), the yardsticks:

    keep    hold the position with the start's width (no width order: fire at will)
    gap     a width order first - the gap between our infantry (gap / angle layouts; the open: the position's front),
            within our limits - then hold and fire at will

    bash tools/nn/dock.sh tools.nn.unit.exam_shootw [--ckpt build/.../latest.pt | --script all] [--device cuda] [--map ..]

Metrics (our shooter, slot 0): men firing - of the fire it could have had at its target, the share it had (the
simulator's bookkeeping: 1 - fc_skip / fc_cap; friends closing the line take the rest); fire lost to friends (fc_skip
/ fc_cap); fire time - the share of the seconds an enemy stands within its range (centre to centre, seen) that it
fires; damage by fire - the enemies' health lost (share of their starting health, summed) while none of ours touches
them; the trade; width orders a skirmish (re-forms begun); its front at its first volley against the gap; out of the
position (own choice; ~0). By layout.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from tools.nn.sim import battle, geometry
from tools.nn.sim import orders as O
from tools.nn.sim.params import load
from tools.nn.train import reward
from tools.nn.unit import decode as Dc
from tools.nn.unit import drill_shootw as DW5
from tools.nn.unit import net as Nt

SEED = 4932
SCRIPTS = ("keep", "gap")
START_S = 1.0          # the script's width order: given this long before the hold


class Watch:
    def __init__(self, B, N, P, device):
        self.N, self.dt = N, P.dt
        self.spacing = P.sim["formation"]["spacing_m"]
        z = lambda: torch.zeros(B, device=device)       # noqa: E731
        self.in_range_s, self.fire_s, self.cap, self.skip, self.fire_dmg = z(), z(), z(), z(), z()
        self.out_free, self.alive_s, self.reforms = z(), z(), z()
        self.front_fire = torch.full((B,), -1.0, device=device)
        self.prev = None
        self.prev_wnew = None

    def __call__(self, st, hp0, was_done, area, step):
        u = st.u
        N = self.N
        live = ~was_done
        if self.prev is None:
            self.prev = u["hp_abs"].clone()
        lost = (self.prev - u["hp_abs"]).clamp(min=0) / u["hp0"].clamp(min=1) * live[:, None].float()
        self.prev = u["hp_abs"].clone()
        hs = u["hit_sec"].view(st.B, N, N)
        touch = (hs > 0) | (hs.transpose(1, 2) > 0)
        ours = u["side"] == 1
        foe = u["side"] == 2
        dx, dz = u["x"] - u["x"][:, :1], u["z"] - u["z"][:, :1]
        dist = torch.sqrt(dx * dx + dz * dz)                             # [B, j] from our shooter
        foe_up = foe & (u["men"] > 0) & ~u["gone"] & ~u["r"] & u["vis"]
        me_ok = live & (u["men"][:, 0] > 0) & ~u["r"][:, 0]
        in_range = me_ok & (foe_up & (dist <= u["range"][:, :1])).any(1)
        self.in_range_s += in_range.float() * self.dt
        self.fire_s += (in_range & u["fire"][:, 0]).float() * self.dt
        self.cap += u["fc_cap"][:, 0] * live.float()
        self.skip += u["fc_skip"][:, 0] * live.float()
        foe_free = foe & ~(touch & ours[:, :, None]).any(1)
        self.fire_dmg += (lost * foe_free).sum(1)
        front, _ = geometry.dims(u, self.spacing)
        shot = live & (u["fc_cap"][:, 0] > u["fc_skip"][:, 0] + 1e-9)
        self.front_fire = torch.where(shot & (self.front_fire < 0), front[:, 0], self.front_fire)
        wn = u["wnew"][:, 0]
        if self.prev_wnew is None:
            self.prev_wnew = torch.full_like(wn, -1.0)
        self.reforms += (live & (wn >= 0) & ((wn - self.prev_wnew).abs() > 1e-3)).float()
        self.prev_wnew = wn.clone()
        a0 = {k: a[:, 0] for k, a in area.items() if k in ("x", "z", "b", "front", "depth")}
        sd = Dc.signed_dist(u["x"][:, 0], u["z"][:, 0], a0)
        alive = live & (u["men"][:, 0] > 0)
        pushed = u["m"][:, 0] | u["r"][:, 0]
        self.alive_s += alive.float() * self.dt
        self.out_free += (alive & (sd < -0.5) & ~pushed).float() * self.dt


def _mean(x, m):
    return round(float(x[m].float().mean()), 3) if bool(m.any()) else None


def _share(a, b, m):
    s = float(b[m].sum())
    return round(float(a[m].sum()) / s, 3) if s > 0 else None


def gap_width(sp, area):
    """[B] the script 'gap''s width: the gap between our infantry (the open layout: the position's front)."""
    g = torch.tensor([s["gap"] if s["layout"] != "open" else s["pos_front"] for s in sp], device=area["front"].device)
    return torch.minimum(g, area["front"][:, 0])


def script_orders(st, name, want, wlim, t0):
    """Our shooter's orders (slot 0) of a script: hold where it stands (fire at will); 'gap' with a width order want [B]
    (within our limits) in its first START_S."""
    o = O.hold(st.B, st.N, st.device)
    o.kind[:, 0] = O.HOLD
    o.x[:, 0], o.z[:, 0] = st.u["x"][:, 0], st.u["z"][:, 0]
    if name == "gap" and float(st.t.max()) - t0 < START_S - 1e-6:
        o.width[:, 0] = torch.minimum(torch.maximum(want, wlim[0][:, 0]), wlim[1][:, 0])
    return o


def table_of(sp, st, w, m0, m1, rs, out_path, who, t0):
    trade = ((m0[:, 1, 3] - m1[:, 1, 3]) - (m0[:, 0, 3] - m1[:, 0, 3])).cpu()
    c = {k: getattr(w, k).cpu() for k in ("in_range_s", "fire_s", "cap", "skip", "fire_dmg", "out_free", "alive_s",
                                           "reforms", "front_fire")}
    gap = torch.tensor([s["gap"] for s in sp])

    def block(m):
        met = m & (c["front_fire"] > 0)
        return {"n": int(m.sum()), "men_firing": _share(c["cap"] - c["skip"], c["cap"], m),
                "fire_lost_to_friends": _share(c["skip"], c["cap"], m),
                "fire_time": _share(c["fire_s"], c["in_range_s"], m), "fire_damage": _mean(c["fire_dmg"], m),
                "trade": _mean(trade, m), "width_orders": _mean(c["reforms"], m),
                "front_at_fire_m": _mean(c["front_fire"], met), "front_at_fire_v_gap": _mean(c["front_fire"] / gap, met),
                "out_free": _share(c["out_free"], c["alive_s"], m)}
    allm = torch.ones(len(sp), dtype=torch.bool)
    lay = lambda k: torch.tensor([s["layout"] == k for s in sp])              # noqa: E731
    table = {"who": who, "all": block(allm), "by_layout": {k: block(lay(k)) for k in DW5.LAYOUTS},
             "duration_s": round(float(st.t.mean()), 1), "seconds": round(time.time() - t0, 1)}
    table.update({k: v for k, v in rs.items() if k in ("end", "entropy", "bans_broken")})
    if "order_changes" in rs:
        table["order_changes_per_skirmish"] = round(rs["order_changes"] / len(sp), 1)
    table["terrain"] = rs.get("terrain")         # (realmap.Meter: on a map only)
    if out_path:
        Path(out_path).write_text(json.dumps(table, indent=1), encoding="utf-8")
    print(json.dumps({"exam_shootw": table}), flush=True)
    return table


def run(net, P=None, device="cpu", duty=None, out_path=None, seed=SEED):
    from tools.nn.unit import train as T
    P = P or load()
    duty = duty or T.Duty(device, 1.0)
    sp = DW5.exam_specs()
    cfg = {"battles": len(sp), "seconds": DW5.LIMIT_S, "cell_m": 8.0, "bounds": None}
    torch.manual_seed(seed)
    setup = T.shootw_setup(cfg, P, device, sp=sp)
    st = setup["st"]
    w = Watch(st.B, st.N, P, device)
    m0 = reward.measure(st)
    t0 = time.time()
    was = net.training
    net.eval()
    _, rs = T.rollout(net, None, cfg, P, device, duty, setup, watch=w)
    net.train(was)
    m1 = reward.measure(st)
    return table_of(sp, st, w, m0, m1, rs, out_path, "network", t0)


def run_script(name, P=None, device="cpu", out_path=None):
    """The exam played by a script in the network's place (no latency; it holds, so no leash)."""
    from tools.nn.unit import train as T
    P = P or load()
    sp = DW5.exam_specs()
    cfg = {"battles": len(sp), "seconds": DW5.LIMIT_S, "cell_m": 8.0, "bounds": None}
    setup = T.shootw_setup(cfg, P, device, sp=sp)            # (laid on the active map, as the network's)
    st, own, area, ender = setup["st"], setup["own"], setup["area"], setup["ender"]
    wlim = Dc.width_limits(st, P)
    want = gap_width(sp, area)
    spacing = P.sim["formation"]["spacing_m"]
    t0 = float(st.t.max())
    m0 = reward.measure(st)
    step = battle.stepper(device)
    tm = time.time()
    from tools.nn.unit import realmap
    meter = realmap.Meter(st.B, device) if realmap.active() else None
    bi, ii = own.nonzero(as_tuple=True)
    w = Watch(st.B, st.N, P, device)
    for _ in range(int(round(DW5.LIMIT_S / P.dt)) + 1):
        was = st.done.clone()
        o = O.merge(setup["script"](st), script_orders(st, name, want, wlim, t0), own)
        hp0 = st.u["hp_abs"][:, 0].clone()
        step(st, o, P)
        ender.after_step(st, spacing, area)
        w(st, hp0, was, area, 0)
        if meter is not None:
            meter.step(st, bi, ii, P, was)
        if bool(st.done.all()):
            break
    m1 = reward.measure(st)
    rs = {"end": {k: float((ender.why == i).float().mean()) for i, k in
                  ((1, "we_routed"), (2, "enemy_left"), (3, "limit"), (0, "open"))}}
    if meter is not None:
        rs["terrain"] = meter.result()
    return table_of(sp, st, w, m0, m1, rs, out_path, f"script {name}", tm)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="a unit_v1 checkpoint (none: an untrained network, seed 0)")
    ap.add_argument("--script", default=None, choices=SCRIPTS + ("all",), help="a script in the network's place "
                    "(all: each in turn)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--gpu-duty", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--map", default=None, help="lay the skirmishes on these maps (a name, a,b or all - build/maps/<name>, "
                    "tools/nn/unit/realmap.py; several: in turn): the exam's fixed places")
    a = ap.parse_args(argv)
    if a.map:
        from tools.nn.unit import realmap
        realmap.activate(a.map, exam=True)
    from tools.nn.unit import train as T
    P = load()
    if a.script:
        names = SCRIPTS if a.script == "all" else (a.script,)
        return {nm: run_script(nm, P, a.device, None if a.out is None else
                               str(Path(a.out).with_name(f"{Path(a.out).stem}_{nm}.json"))) for nm in names}
    if a.ckpt:
        net, _, _ = T.restore(a.ckpt, a.device)
    else:
        torch.manual_seed(0)
        net = Nt.UnitNet().to(a.device)
    return run(net, P, a.device, T.Duty(a.device, a.gpu_duty), a.out)


if __name__ == "__main__":
    main()

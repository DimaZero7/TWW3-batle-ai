"""The exam of drill U10 'the height in my area' (drill_height.py; a real map only - --map, default ksl_plains_01): the
fixed set drill_height.exam_specs() (240: hold / shoot x enemy from above / below / across, in turn) at the drill's fixed
places, the network sampling with a fixed seed - or a script in its place (--script), the yardsticks:

    stay    hold where it was put (no order)
    high    walk to the highest of the area's sample points (its box less half the unit's front and depth), face the
            enemy, hold

    bash tools/nn/dock.sh tools.nn.unit.exam_height [--ckpt build/.../latest.pt | --script all] [--device cuda]

Metrics: infantry - our ground over the nearest enemy's while in melee (dh_melee_m: + we stand higher, better; the
centres' ground), the share of the skirmishes whose first contact found us higher (above_at_contact), the trade;
shooters - our ground over the target's while we fire (dh_fire_m), damage by fire (the enemy's health lost, share, while
none of ours touches it), the trade; by the enemy's side (from above / below / across); the places' rise (relief_m).
"""
import argparse
import json
import time
from pathlib import Path

import torch

from tools.nn.sim import battle, geometry, terrain
from tools.nn.sim import orders as O
from tools.nn.sim.params import load
from tools.nn.train import reward
from tools.nn.unit import drill_height as DT
from tools.nn.unit import net as Nt
from tools.nn.unit import realmap

SEED = 4941
SCRIPTS = ("stay", "high")
MAP = "ksl_plains_01"
ARRIVE_M = 3.0


class Watch:
    def __init__(self, B, N, P, device):
        self.N, self.H, self.dt = N, N // 2, P.dt
        z = lambda: torch.zeros(B, device=device)       # noqa: E731
        self.dh_melee, self.melee_s, self.dh_fire, self.fire_s, self.fire_dmg = z(), z(), z(), z(), z()
        self.dh_contact = torch.full((B,), float("nan"), device=device)
        self.prev = None

    def __call__(self, st, hp0, was_done, area, step):
        u = st.u
        H, N = self.H, self.N
        live = ~was_done & (u["men"][:, 0] > 0) & ~u["r"][:, 0]
        if self.prev is None:
            self.prev = u["hp_abs"].clone()
        lost = (self.prev - u["hp_abs"]).clamp(min=0) / u["hp0"].clamp(min=1)
        self.prev = u["hp_abs"].clone()
        g = terrain.ground(u)
        dh = g[:, 0] - g[:, H]
        melee = live & u["m"][:, 0]
        self.dh_melee += torch.where(melee, dh, torch.zeros_like(dh)) * self.dt
        self.melee_s += melee.float() * self.dt
        self.dh_contact = torch.where(melee & torch.isnan(self.dh_contact), dh, self.dh_contact)
        fire = live & u["fire"][:, 0] & ~u["m"][:, 0]
        self.dh_fire += torch.where(fire, dh, torch.zeros_like(dh)) * self.dt
        self.fire_s += fire.float() * self.dt
        touch = u["hit_sec"].view(st.B, N, N)
        free = ~((touch[:, 0, H] > 0) | (touch[:, H, 0] > 0))
        self.fire_dmg += lost[:, H] * (~was_done & free).float()


def _share(a, b, m):
    s = float(b[m].sum())
    return round(float(a[m].sum()) / s, 3) if s > 0 else None


def _mean(x, m):
    m = m & ~torch.isnan(x.float())
    return round(float(x[m].float().mean()), 3) if bool(m.any()) else None


def high_point(st, area, P):
    """[B] x, z of the highest of each area's sample points (drill_height.area_points' grid, its box less half the unit's
    front and depth on each side), on the battle's map."""
    front, depth = geometry.dims(st.u, P.sim["formation"]["spacing_m"])
    n = DT.GRID
    a = torch.linspace(-0.5, 0.5, n, device=st.device)
    fr = (area["front"][:, 0] - front[:, 0]).clamp(min=0)
    dp = (area["depth"][:, 0] - depth[:, 0]).clamp(min=0)
    S = a[None, :, None] * fr[:, None, None]
    F = a[None, None, :] * dp[:, None, None]
    S, F = S.expand(-1, n, n).reshape(st.B, -1), F.expand(-1, n, n).reshape(st.B, -1)
    t = area["b"][:, 0:1] * (torch.pi / 180)
    x = area["x"][:, 0:1] + S * torch.cos(t) + F * torch.sin(t)
    z = area["z"][:, 0:1] - S * torch.sin(t) + F * torch.cos(t)
    h = terrain.height(st.u, x, z)
    k = h.argmax(1, keepdim=True)
    return x.gather(1, k)[:, 0], z.gather(1, k)[:, 0]


def script_orders(st, name, goal_xz):
    """Our unit's orders (slot 0) of a script: hold where it stands (stay), or walk to goal_xz and hold there facing
    the enemy (high)."""
    u = st.u
    H = st.N // 2
    o = O.hold(st.B, st.N, st.device)
    o.x[:, 0], o.z[:, 0] = u["x"][:, 0], u["z"][:, 0]
    if name == "high":
        gx, gz = goal_xz
        far = (torch.sqrt((gx - u["x"][:, 0]) ** 2 + (gz - u["z"][:, 0]) ** 2) > ARRIVE_M) & ~u["m"][:, 0]
        o.kind[:, 0] = torch.where(far, torch.full_like(o.kind[:, 0], O.MOVE), o.kind[:, 0])
        o.x[:, 0] = torch.where(far, gx, u["x"][:, 0])
        o.z[:, 0] = torch.where(far, gz, u["z"][:, 0])
        face = torch.rad2deg(torch.atan2(u["x"][:, H] - o.x[:, 0], u["z"][:, H] - o.z[:, 0])) % 360
        o.face[:, 0] = face
    return o


def table_of(sp, st, w, m0, m1, rs, places, out_path, who, t0):
    trade = ((m0[:, 1, 3] - m1[:, 1, 3]) - (m0[:, 0, 3] - m1[:, 0, 3])).cpu()
    c = {k: getattr(w, k).cpu() for k in ("dh_melee", "melee_s", "dh_fire", "fire_s", "fire_dmg", "dh_contact")}
    relief = torch.tensor(places["relief"], dtype=torch.float32)

    def block(m):
        hold = m & torch.tensor([s["kind"] == "hold" for s in sp])
        shoot = m & torch.tensor([s["kind"] == "shoot" for s in sp])
        met = hold & ~torch.isnan(c["dh_contact"])
        return {"n": int(m.sum()), "relief_m": _mean(relief, m),
                "hold": {"n": int(hold.sum()), "dh_melee_m": _share(c["dh_melee"], c["melee_s"], hold),
                         "above_at_contact": _mean((c["dh_contact"] > 0).float(), met),
                         "dh_at_contact_m": _mean(c["dh_contact"], met), "trade": _mean(trade, hold)},
                "shoot": {"n": int(shoot.sum()), "dh_fire_m": _share(c["dh_fire"], c["fire_s"], shoot),
                          "fire_damage": _mean(c["fire_dmg"], shoot), "trade": _mean(trade, shoot)}}
    allm = torch.ones(len(sp), dtype=torch.bool)
    table = {"who": who, "all": block(allm),
             "by_from": {f: block(torch.tensor([s["frm"] == f for s in sp])) for f in DT.FROM},
             "places_below_relief": round(float((relief < DT.RELIEF_M).float().mean()), 3),
             "duration_s": round(float(st.t.mean()), 1), "seconds": round(time.time() - t0, 1)}
    table.update({k: v for k, v in rs.items() if k in ("end", "entropy", "bans_broken", "terrain")})
    if "order_changes" in rs:
        table["order_changes_per_skirmish"] = round(rs["order_changes"] / len(sp), 1)
    if out_path:
        Path(out_path).write_text(json.dumps(table, indent=1), encoding="utf-8")
    print(json.dumps({"exam_height": table}), flush=True)
    return table


def _ensure_map():
    if not realmap.active():
        realmap.activate(MAP, exam=True)


def run(net, P=None, device="cpu", duty=None, out_path=None, seed=SEED):
    from tools.nn.unit import train as T
    _ensure_map()
    P = P or load()
    duty = duty or T.Duty(device, 1.0)
    sp = DT.exam_specs()
    cfg = {"battles": len(sp), "seconds": DT.LIMIT_S, "cell_m": 8.0, "bounds": None}
    torch.manual_seed(seed)
    setup = T.height_setup(cfg, P, device, sp=sp)
    st = setup["st"]
    w = Watch(st.B, st.N, P, device)
    m0 = reward.measure(st)
    t0 = time.time()
    was = net.training
    net.eval()
    _, rs = T.rollout(net, None, cfg, P, device, duty, setup, watch=w)
    net.train(was)
    m1 = reward.measure(st)
    return table_of(sp, st, w, m0, m1, rs, setup["map_place"], out_path, "network", t0)


def run_script(name, P=None, device="cpu", out_path=None):
    """The exam played by a script in the network's place (no latency; it stays in its area, no leash)."""
    from tools.nn.unit import train as T
    _ensure_map()
    P = P or load()
    sp = DT.exam_specs()
    cfg = {"battles": len(sp), "seconds": DT.LIMIT_S, "cell_m": 8.0, "bounds": None}
    setup = T.height_setup(cfg, P, device, sp=sp)
    st, own, area, ender = setup["st"], setup["own"], setup["area"], setup["ender"]
    goal_xz = high_point(st, area, P)
    spacing = P.sim["formation"]["spacing_m"]
    m0 = reward.measure(st)
    step = battle.stepper(device)
    tm = time.time()
    meter = realmap.Meter(st.B, device)
    bi, ii = own.nonzero(as_tuple=True)
    w = Watch(st.B, st.N, P, device)
    for _ in range(int(round(DT.LIMIT_S / P.dt)) + 1):
        was = st.done.clone()
        o = O.merge(setup["script"](st), script_orders(st, name, goal_xz), own)
        step(st, o, P)
        ender.after_step(st, spacing, area)
        w(st, None, was, area, 0)
        meter.step(st, bi, ii, P, was)
        if bool(st.done.all()):
            break
    m1 = reward.measure(st)
    rs = {"end": {k: float((ender.why == i).float().mean()) for i, k in
                  ((1, "we_routed"), (2, "enemy_left"), (3, "limit"), (0, "open"))}, "terrain": meter.result()}
    return table_of(sp, st, w, m0, m1, rs, setup["map_place"], out_path, f"script {name}", tm)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="a unit_v1 checkpoint (none: an untrained network, seed 0)")
    ap.add_argument("--script", default=None, choices=SCRIPTS + ("all",), help="a script in the network's place "
                    "(all: each in turn)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--gpu-duty", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--map", default=MAP, help="the real map (build/maps/<name>): the drill runs on a map only")
    a = ap.parse_args(argv)
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

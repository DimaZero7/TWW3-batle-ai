"""The exam of drill U9 'the width for the task' (drill_width.py): the fixed set drill_width.exam_specs() (240: one enemy
/ a line of two in turn), the network sampling with a fixed seed - or a script in its place (--script), the yardsticks:

    keep        attack the nearest at a run from the start, the start's width (no width order: the network's old way)
    match       a width order first - the enemies' line at the start (drill_width.enemy_span), within our limits - then
                attack the nearest
    area        the same with the area's front for the width
    match_area  the enemies' line, no wider than the area's front

    bash tools/nn/dock.sh tools.nn.unit.exam_width [--ckpt build/.../latest.pt | --script keep] [--device cuda]

Metrics: melee damage taken to the flank and rear (exam_multi.Watch: each step's loss split among the enemies in contact
by the sector each is in, seen from our facing); our men striking (the simulator's own men striking, melee.strikes' F,
summed over our targets) a man of ours, and the enemies' men striking us against ours, over our seconds in melee; the
trade; our rout; out of the area (own choice; ~0); the line beyond the area's sides (the share of the time an end of our
front sticks out more than 1 m past the area's side); width orders a skirmish (re-forms begun); our front at the first
contact against the enemies' line and against the area's front. By kind, by our start against the enemies' line
(narrower / wider), by the area against the line.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from tools.nn.sim import battle, geometry, melee
from tools.nn.sim import orders as O
from tools.nn.sim.params import load
from tools.nn.train import opponents, reward
from tools.nn.unit import decode as Dc
from tools.nn.unit import drill_multi as DM
from tools.nn.unit import drill_width as DW
from tools.nn.unit import exam_multi as EM
from tools.nn.unit import net as Nt
from tools.nn.unit import zmap as Z

SEED = 4926
SCRIPTS = ("keep", "match", "area", "match_area")
START_S = 1.0          # the scripts' width order: given this long (HOLD) before the attack


class Strikers:
    """Records the men striking of the simulator's last melee.strikes (F [B, i, j]: i's men striking j) while on - for
    the exam only (it changes nothing). Under a compiled step the record may stay empty: the metric is then None."""

    def __init__(self):
        self.F = None

    def __enter__(self):
        self.orig = melee.strikes

        def wrap(*a, **k):
            out = self.orig(*a, **k)
            self.F = out[3]
            return out
        melee.strikes = wrap
        return self

    def __exit__(self, *exc):
        melee.strikes = self.orig


class Watch(EM.Watch):
    def __init__(self, B, N, P, device, strikers=None):
        super().__init__(B, N, P, device)
        self.spacing = P.sim["formation"]["spacing_m"]
        self.H = N // 2
        self.strikers = strikers
        z = lambda: torch.zeros(B, device=device)       # noqa: E731
        self.melee_s, self.ours_strike, self.theirs_strike = z(), z(), z()
        self.beyond_s, self.reforms = z(), z()
        self.front_contact = torch.full((B,), -1.0, device=device)
        self.prev_wnew = None

    def __call__(self, st, hp0, was_done, area, step):
        super().__call__(st, hp0, was_done, area, step)
        u = st.u
        H = self.H
        live = ~was_done
        alive = live & (u["men"][:, 0] > 0)
        front, _ = geometry.dims(u, self.spacing)
        f0 = front[:, 0]
        # the line beyond the area's sides (in the area's frame: our centre across + half our front)
        s, _ = Z.local(u["x"][:, 0], u["z"][:, 0], area["x"][:, 0], area["z"][:, 0], area["b"][:, 0])
        beyond = s.abs() + f0 / 2 > area["front"][:, 0] / 2 + 1.0
        self.beyond_s += (alive & beyond).float() * self.dt
        # re-forms begun (a new width being formed)
        wn = u["wnew"][:, 0]
        if self.prev_wnew is None:
            self.prev_wnew = torch.full_like(wn, -1.0)
        self.reforms += (live & (wn >= 0) & ((wn - self.prev_wnew).abs() > 1e-3)).float()
        self.prev_wnew = wn.clone()
        hs = u["hit_sec"].view(st.B, self.N, self.N)[:, :, 0] > 0
        in_c = live & (hs & (u["side"] == 2)).any(1)
        self.front_contact = torch.where(in_c & (self.front_contact < 0), f0, self.front_contact)
        F = self.strikers.F if self.strikers is not None else None
        if F is not None:
            m = alive & u["m"][:, 0]
            self.melee_s += m.float() * self.dt
            ours = F[:, 0, H:].sum(1) / u["men"][:, 0].clamp(min=1)
            theirs = F[:, H:, 0].sum(1) / F[:, 0, H:].sum(1).clamp(min=1e-6)
            self.ours_strike += torch.where(m, ours, torch.zeros_like(ours)) * self.dt
            self.theirs_strike += torch.where(m & (F[:, 0, H:].sum(1) > 0), theirs, torch.zeros_like(ours)) * self.dt


def script_orders(st, name, span, area, wlim, t0):
    """Our unit's orders (slot 0) of a script: before START_S a hold with its width (keep: none), then attack the
    nearest at a run (replay.nearest_attack). span [B]: the enemies' line at the start."""
    o = opponents.nearest(st)
    lo, hi = wlim[0][:, 0], wlim[1][:, 0]
    if name == "keep":
        return o
    want = {"match": span, "area": area["front"][:, 0],
            "match_area": torch.minimum(span, area["front"][:, 0])}[name]
    w = torch.minimum(torch.maximum(want, lo), hi)
    if float(st.t.max()) - t0 < START_S - 1e-6:
        o.kind[:, 0] = O.HOLD
        o.target[:, 0] = -1
        o.width[:, 0] = w
    return o


def _mean(x, m):
    return round(float(x[m].float().mean()), 3) if bool(m.any()) else None


def _share(a, b, m):
    s = float(b[m].sum())
    return round(float(a[m].sum()) / s, 3) if s > 0 else None


def run_script(name, P=None, device="cpu", out_path=None):
    """The exam played by a script in the network's place (no latency, the leash as the network's)."""
    P = P or load()
    sp = DW.exam_specs()
    st, own, area = DW.build(sp, P, device)
    ender = DM.Ender(st.B, device, DW.LIMIT_S)
    leash = Dc.Leash(st.B, st.N, device)
    span = torch.tensor([DW.enemy_span(s) for s in sp], device=device)
    wlim = Dc.width_limits(st, P)
    spacing = P.sim["formation"]["spacing_m"]
    t0 = float(st.t.max())
    m0 = reward.measure(st)
    step = battle.stepper(device)
    tm = time.time()
    with Strikers() as sk:
        w = Watch(st.B, st.N, P, device, sk)
        for _ in range(int(round(DW.LIMIT_S / P.dt)) + 1):
            was = st.done.clone()
            mine = script_orders(st, name, span, area, wlim, t0)
            o = O.merge(opponents.nearest(st), leash.apply(st, mine, area, own), own)
            hp0 = st.u["hp_abs"][:, 0].clone()
            step(st, o, P)
            ender.after_step(st, spacing, area)
            w(st, hp0, was, area, 0)
            if bool(st.done.all()):
                break
    m1 = reward.measure(st)
    return table_of(sp, st, w, m0, m1, {"end": {k: float((ender.why == i).float().mean()) for i, k in
                                                 ((1, "we_routed"), (2, "enemy_left"), (3, "limit"), (0, "open"))}},
                    out_path, f"script {name}", tm)


def table_of(sp, st, w, m0, m1, rs, out_path, who, t0):
    trade = ((m0[:, 1, 3] - m1[:, 1, 3]) - (m0[:, 0, 3] - m1[:, 0, 3])).cpu()
    routed = (st.u["r"][:, 0] | (st.u["men"][:, 0] <= 0)).cpu()
    span = torch.tensor([DW.enemy_span(s) for s in sp])
    width0 = torch.tensor([s["width"] for s in sp])
    afront = torch.tensor([s["area_front"] for s in sp])
    fc = w.front_contact.cpu()
    c = {k: getattr(w, k).cpu() for k in ("dmg_all", "dmg_side", "out_free", "alive_s", "beyond_s", "reforms",
                                           "melee_s", "ours_strike", "theirs_strike")}
    got_f = bool((c["melee_s"] > 0).any())

    def block(m):
        met = m & (fc > 0)
        return {"n": int(m.sum()), "dmg_flank_rear": _share(c["dmg_side"], c["dmg_all"], m),
                "ours_striking": _share(c["ours_strike"], c["melee_s"], m) if got_f else None,
                "theirs_v_ours_striking": _share(c["theirs_strike"], c["melee_s"], m) if got_f else None,
                "trade": _mean(trade, m), "we_routed": _mean(routed, m),
                "out_free": _share(c["out_free"], c["alive_s"], m), "beyond_area": _share(c["beyond_s"], c["alive_s"], m),
                "width_orders": _mean(c["reforms"], m),
                "front_at_contact_v_line": _mean(fc / span, met), "front_at_contact_v_area": _mean(fc / afront, met),
                "front_at_contact_m": _mean(fc, met)}
    allm = torch.ones(len(sp), dtype=torch.bool)
    kind = lambda k: torch.tensor([s["kind"] == k for s in sp])                # noqa: E731
    table = {"who": who, "all": block(allm), "by_kind": {k: block(kind(k)) for k in DW.KINDS},
             "by_start": {"narrower_than_line": block(width0 < span), "wider_than_line": block(width0 >= span)},
             "by_area": {"narrower_than_line": block(afront < span), "wider_than_line": block(afront >= span)},
             "duration_s": round(float(st.t.mean()), 1), "seconds": round(time.time() - t0, 1)}
    table.update({k: v for k, v in rs.items() if k in ("end", "entropy", "bans_broken")})
    if "order_changes" in rs:
        table["order_changes_per_skirmish"] = round(rs["order_changes"] / len(sp), 1)
    table["terrain"] = rs.get("terrain")         # (realmap.Meter: on a map only)
    if out_path:
        Path(out_path).write_text(json.dumps(table, indent=1), encoding="utf-8")
    print(json.dumps({"exam_width": table}), flush=True)
    return table


def run(net, P=None, device="cpu", duty=None, out_path=None, seed=SEED):
    from tools.nn.unit import train as T
    P = P or load()
    duty = duty or T.Duty(device, 1.0)
    sp = DW.exam_specs()
    cfg = {"battles": len(sp), "seconds": DW.LIMIT_S, "cell_m": 8.0, "bounds": None}
    torch.manual_seed(seed)
    setup = T.width_setup(cfg, P, device, sp=sp)
    st = setup["st"]
    m0 = reward.measure(st)
    t0 = time.time()
    was = net.training
    net.eval()
    with Strikers() as sk:
        w = Watch(st.B, st.N, P, device, sk)
        _, rs = T.rollout(net, None, cfg, P, device, duty, setup, watch=w)
    net.train(was)
    m1 = reward.measure(st)
    return table_of(sp, st, w, m0, m1, rs, out_path, "network", t0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="a unit_v1 checkpoint (none: an untrained network, seed 0)")
    ap.add_argument("--script", default=None, choices=SCRIPTS + ("all",), help="a script in the network's place "
                    "(all: each in turn)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--gpu-duty", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--map", default=None, help="lay the skirmishes on this map (build/maps/<name>, tools/nn/unit/realmap.py): the exam's fixed places")
    a = ap.parse_args(argv)
    if a.map:
        from tools.nn.unit import realmap
        realmap.activate(a.map, exam=True)
    from tools.nn.unit import train as T
    P = load()
    if a.script:
        names = SCRIPTS if a.script == "all" else (a.script,)
        out = {}
        for nm in names:
            o = None if a.out is None else str(Path(a.out).with_name(f"{Path(a.out).stem}_{nm}.json"))
            out[nm] = run_script(nm, P, a.device, o)
        return out
    if a.ckpt:
        net, _, _ = T.restore(a.ckpt, a.device)
    else:
        torch.manual_seed(0)
        net = Nt.UnitNet().to(a.device)
    return run(net, P, a.device, T.Duty(a.device, a.gpu_duty), a.out)


if __name__ == "__main__":
    main()

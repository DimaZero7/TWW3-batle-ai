"""The exam of the tactician's first drill (design/tactic "Экзамен"): no teacher, the fixed battles drill1.exam_specs(),
each start apart. The main measure: the share of battles where at the contact the infantry stands between the enemy and
the shooters. The yardstick: the same battles with the teacher's formula driving; the network no worse. The safeguards:
how often the shooters got into melee, whether they fired, frame changes a battle, the share of 'keep', the last resort's uses (fell),
the trade. For information: the
infantry's front to its frame's width at the contact (does a unit fill a wide zone).

    bash ops/dock.sh net.tactic.exam1 [--ckpt tmp/old-build/.../latest.pt | --formula | --random] [--device cuda]
"""
import argparse
import json
import time
from pathlib import Path

import torch

from sim.params import load
from net.tactic import decide as DC
from net.tactic import drill1 as DR
from net.tactic import env as EN


def _mean(x, m):
    return round(float(x[m].float().mean()), 3) if bool(m.any()) else None


def table(sp, rd, st):
    """{start: {...}, all: {...}} of a batch's readings."""
    def block(m):
        contact = rd["contact"] & m
        out = {"n": int(m.sum()), "between": _mean(rd["between"], m), "contact": _mean(rd["contact"], m),
               "hit": _mean(rd["hit"], m), "shooters_in_melee": _mean(rd["melee"], m),
               "shooters_fired_s": _mean(rd["fired_s"], m), "ammo_used": _mean(rd["ammo_used"], m),
               "trade": _mean(rd["trade"], m), "contact_t": _mean(rd["t_contact"], contact),
               "fill": _mean(rd["fill"], contact & ~torch.isnan(rd["fill"]))}
        return out
    dev = rd["contact"].device
    out = {"all": block(torch.ones(len(sp), dtype=torch.bool, device=dev))}
    for s in DR.STARTS:
        out[s] = block(torch.tensor([x["start"] == s for x in sp], device=dev))
    return out


def run(unit_net, P=None, device="cpu", net=None, formula=False, greedy=False, n=None, seed=None, out_path=None, who=None):
    """The exam played by a TacticNet (sampling or its best) or the teacher's formula. Returns the table."""
    P = P or load()
    sp = DR.exam_specs(n=n)
    torch.manual_seed(int(seed if seed is not None else 4953))
    t0 = time.time()
    setup = EN.Setup(sp, P, device)
    was = None
    if net is not None:
        was = net.training
        net.eval()
    _, rs, rd = EN.play(setup, unit_net, net=net, formula=formula, greedy=greedy, keep=False)
    if net is not None:
        net.train(was)
    t = table(sp, rd, setup.st)
    t["who"] = who or ("formula" if formula else "network")
    t["changes_per_battle"] = rs["changes_per_battle"]
    t["fell"] = rs["fell"]
    t["keep_share"] = rs["keep_share"]
    t["forced"] = rs["forced"]
    t["small_moves_per_battle"] = rs["small_moves_per_battle"]
    t["frames_overlap"], t["frames_out"] = rs["frames_overlap"], rs["frames_out"]
    t["seconds"] = round(time.time() - t0, 1)
    t["why"] = {k: round(float((rd["why"] == i).float().mean()), 3) for i, k in
                ((1, "we_out"), (2, "enemy_out"), (3, "limit"), (0, "open"))}
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(t, indent=1), encoding="utf-8")
    print(json.dumps({"exam_tactic1": t}), flush=True)
    return t


def main(argv=None):
    from net.tactic import train as TT
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="a tactic_v1 checkpoint")
    ap.add_argument("--formula", action="store_true", help="the teacher's formula drives (the yardstick)")
    ap.add_argument("--random", action="store_true", help="a network with fresh random weights (seed 0)")
    ap.add_argument("--greedy", action="store_true", help="the network's best choices, not sampling")
    ap.add_argument("--battles", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    P = load()
    unit_net = TT.unit_network(a.device)
    net = None
    if a.ckpt:
        net, _, _ = TT.restore(a.ckpt, a.device)
    elif a.random:
        torch.manual_seed(0)
        net = DC.TacticNet().to(a.device)
    return run(unit_net, P, a.device, net, a.formula or net is None, a.greedy, a.battles, out_path=a.out)


if __name__ == "__main__":
    main()

"""The strategist's exam (plan.md "Обучение"): the fixed battles cases.exam_specs() (config exam.cases: 192 attack + 192 hold,
random armies 3-8 with a lord; --arena: both armies the arena's, 7 with a lord), no teacher. The measure of each case (battle.measures); the yardstick - the same battles under the
formula; the plain strategies (teacher.MODES) show the measure tells the right from the wrong. The safeguards: the
deployment's errors (deploy.errors + no open cell), the parcels' errors at second 0 and after, the operative's re-forms.

    bash ops/dock.sh net.strategist.exam [--ckpt <strategist ckpt> | --mode formula|blob|front|hold|attack]
        [--greedy] [--op formula|net] [--tactic-net] [--device cuda] [--out <json>]
"""
import argparse
import json
import time
from pathlib import Path

import torch

from net import ROOT
from sim.params import load
from net.strategist import battle as BT
from net.strategist import cases as CS
from net.strategist import cfg


def _m(x, sel):
    return round(float(x[sel].float().mean()), 3) if bool(sel.any()) else None


def table(pre, rd, rs, out):
    m = BT.measures(pre, rd)
    t = {}
    groups = out["plan"]["on"].sum(1).float()
    destroy = out["plan"]["verb"] == 0
    for i, case in enumerate(CS.CASES):
        sel = pre.case == i
        if not bool(sel.any()):
            continue
        row = {"n": int(sel.sum()), "measure": _m(m[case], sel), "contact": _m(rd["contact"], sel),
               "trade": _m(rd["trade"], sel), "groups": _m(groups, sel), "said_destroy": _m(destroy, sel)}
        if case == "wide":
            row["threes_met"] = _m(((rd["t3"] >= 0) & rd["has3"]).float().sum(1) / 2, sel)
            row["threes_hit"] = _m((rd["hit3"] & rd["has3"]).float().sum(1) / 2, sel)
            row["threes_flank"] = _m((rd["flank3"] & rd["has3"]).float().sum(1) / 2, sel)
        else:
            row["between"] = _m(rd["between"], sel)
            row["lost"] = _m(rd["lost"], sel)
            row["advanced"] = _m(rd["advanced"], sel)
            att = sel & rd["contact"]
            row["t_contact"] = _m(rd["t_contact"], att)
            row["lost_at_contact"] = _m(rd["lost_contact"], att)
        t[case] = row
    for k in ("deploy_errors", "deploy_error_battles", "parcels", "parcel_errors", "errors", "op_reforms_per_battle",
              "trade", "tactic_fell", "tactic_fell_share"):
        t[k] = rs[k]
    return t


def run(unit_net, P=None, device="cpu", net=None, mode="formula", greedy=False, n=None, seed=None, out_path=None,
        who=None, op_net=None, op_mode="formula", tactic_net=None, arena=False):
    P = P or load()
    sp = CS.exam_specs(n=n, seed=seed, arena=arena)
    torch.manual_seed(int(cfg()["exam"]["seed"]))
    t0 = time.time()
    pre = CS.Pre(sp, P, device)
    was = None
    if net is not None:
        was = net.training
        net.eval()
    out, inp, ctx = BT.decide(pre, net, mode, greedy)
    if net is not None:
        net.train(was)
    rd, rs = BT.play(pre, out, inp, unit_net, op_net=op_net, op_mode=op_mode, tactic_net=tactic_net)
    t = table(pre, rd, rs, out)
    t["who"] = who or (("network" + (" greedy" if greedy else "")) if net is not None else mode)
    t["operative"] = cfg()["operative_ckpt"] if op_net is not None else "formula"
    t["tactician"] = "run4 network" if tactic_net is not None else "formula"
    t["armies"] = "arena 7 v 7" if arena else "random 3-8 with a lord"
    own_n = pre.own.sum(1).float()
    t["own_units"] = round(float(own_n.mean()), 2)
    t["groups_by_size"] = {int(k): _m(out["plan"]["on"].sum(1).float(), own_n == k) for k in own_n.unique().tolist()}
    t["measure_by_size"] = {int(k): _m(BT.measure_of(pre, rd), own_n == k) for k in own_n.unique().tolist()}
    t["seconds"] = round(time.time() - t0, 1)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(t, indent=1), encoding="utf-8")
    print(json.dumps({"exam_strategist": t}), flush=True)
    return t


def operative_net(device):
    from net.operative import train as OT
    net, _, _ = OT.restore(str(ROOT / cfg()["operative_ckpt"]), device)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def main(argv=None):
    from net.tactic import train as TT
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--mode", default=None, help="comma-separated: the formula and the plain strategies (no --ckpt and "
                                                 "no --mode: config default)")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--op", default=None, help="the operative: net (config operative_ckpt) or formula (default: config operative)")
    ap.add_argument("--tactic-net", action="store_true")
    ap.add_argument("--battles", type=int, default=None)
    ap.add_argument("--arena", action="store_true", help="both armies the arena's (7 v 7 Empire with a lord)")
    ap.add_argument("--drill-cap", type=int, default=None, help="override config teacher.drill_cap (0 / 1)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="a json path; with several modes <out>_<mode>.json")
    a = ap.parse_args(argv)
    if a.drill_cap is not None:
        cfg()["teacher"]["drill_cap"] = bool(a.drill_cap)
    P = load()
    unit_net = TT.unit_network(a.device)
    op_net = operative_net(a.device) if (a.op or cfg()["operative"]) == "net" else None
    tnet = None
    if a.tactic_net:
        tnet, _, _ = TT.restore(str(ROOT / cfg()["tactic_ckpt"]), a.device)
        tnet.eval()
    if a.ckpt is None and a.mode is None:
        from net.strategist import default
        who = default()
        a.ckpt, a.mode = (None, "formula") if who == "formula" else (who, None)
    if a.ckpt:
        from net.strategist import train as ST
        net, _, _ = ST.restore(a.ckpt, a.device)
        return run(unit_net, P, a.device, net, greedy=a.greedy, n=a.battles, out_path=a.out, op_net=op_net,
                   tactic_net=tnet, arena=a.arena)
    modes = a.mode.split(",")
    for mode in modes:
        path = a.out if (a.out is None or len(modes) == 1) else a.out.replace(".json", f"_{mode}.json")
        run(unit_net, P, a.device, None, mode, n=a.battles, out_path=path, op_net=op_net, tactic_net=tnet,
            arena=a.arena)


if __name__ == "__main__":
    main()

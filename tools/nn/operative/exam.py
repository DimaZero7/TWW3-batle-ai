"""The operative's exam (plan.md "Обучение и страховка"): the fixed battles cases.exam_specs() (128 a case; the go case
64 a command), no teacher. The main measure of each case (env.measures): gather - the share of my units within 150 m of
the first contact; split - the share of the clumps with the infantry between at their contact; go - the share of the
battles done as the command says ('destroy': a contact with the infantry between; 'hold': no contact, my centre within
100 m of the start). The yardstick: the same battles under the formula; the plain operators (teacher.MODES) show the
measure tells the right from the wrong. The safeguards: the parcels' errors (parcel.check on every parcel sent), the
re-forms and moves a battle, the groups, the trade, the tactician's last resort.

    bash tools/nn/dock.sh tools.nn.operative.exam [--ckpt build/.../latest.pt | --mode formula|each|blob|stay|go]
        [--greedy] [--tactic-net]  (the tactician's run4 network under it: the end-to-end run)
"""
import argparse
import json
import time
from pathlib import Path

import torch

from tools.nn.operative import cfg
from tools.nn.operative import cases as CS
from tools.nn.operative import env as EN
from tools.nn.sim.params import load


def _m(x, sel):
    return round(float(x[sel].float().mean()), 3) if bool(sel.any()) else None


def table(setup, rd, rs):
    m = EN.measures(setup, rd)
    out = {}
    for i, case in enumerate(CS.CASES):
        sel = setup.case == i
        if not bool(sel.any()):
            continue
        row = {"n": int(sel.sum()), "measure": _m(m[case], sel), "contact": _m(rd["contact"], sel),
               "between": _m(rd["between"], sel), "hit": _m(rd["hit"], sel), "trade": _m(rd["trade"], sel),
               "contact_t": _m(rd["t_contact"], sel & rd["contact"])}
        if CS.base(case) == "gather":
            row["gathered"] = _m(torch.nan_to_num(rd["gathered"], nan=0.0), sel)
        if case == "split_a":
            row["stayed"] = _m(rd["stayed"], sel)
        if CS.base(case) == "go":
            row["destroy"] = _m(m[case], sel & setup.destroy)
            row["hold"] = _m(m[case], sel & ~setup.destroy)
            row["advanced_on_hold"] = _m(rd["advanced"], sel & ~setup.destroy)
        out[case] = row
    groups = rd["groups"]
    out["groups_end"] = {cs: _m(groups.on.sum(1).float(), setup.case == i) for i, cs in enumerate(CS.CASES)
                         if bool((setup.case == i).any())}
    own_n = setup.own.sum(1)
    if any(s.get("army") for s in setup.sp):
        m = EN.measures(setup, rd)
        mine = torch.stack([m[CS.CASES[int(c)]][b] for b, c in enumerate(setup.case.tolist())])
        out["measure_by_size"] = {int(k): _m(mine, own_n == k) for k in own_n.unique().tolist()}
        out["groups_by_size"] = {int(k): _m(groups.on.sum(1).float(), own_n == k) for k in own_n.unique().tolist()}
    for k in ("reforms_per_battle", "moves_per_battle", "groups_mean", "parcels", "parcel_errors", "errors",
              "tactic_fell_share", "trade"):
        out[k] = rs[k]
    out["keeps_per_decision"] = round(rs["keeps"] / max(rs["decisions"], 1), 3)    # (groups kept a battle's decision)
    return out


def run(unit_net, P=None, device="cpu", net=None, mode="formula", greedy=False, n=None, seed=None, out_path=None,
        who=None, tactic_net=None, army=False):
    P = P or load()
    sp = CS.exam_specs(n=n, seed=seed, cases=CS.ARMY if army else CS.OLD)
    torch.manual_seed(int(cfg()["exam"]["seed"]))
    t0 = time.time()
    setup = CS.Setup(sp, P, device)
    was = None
    if net is not None:
        was = net.training
        net.eval()
    _, rs, rd = EN.play(setup, unit_net, net=net, mode=mode, greedy=greedy, keep=False, tactic_net=tactic_net)
    if net is not None:
        net.train(was)
    t = table(setup, rd, rs)
    t["who"] = who or (("network" + (" greedy" if greedy else "")) if net is not None else mode)
    t["tactician"] = "run4 network" if tactic_net is not None else "formula"
    t["armies"] = "3-8 with a lord" if army else "the old cases (3 and 6, no lord)"
    t["seconds"] = round(time.time() - t0, 1)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(t, indent=1), encoding="utf-8")
    print(json.dumps({"exam_operative": t}), flush=True)
    return t


def main(argv=None):
    from tools.nn.operative import train as OT
    from tools.nn.tactic import train as TT
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="an operative_v1 checkpoint")
    ap.add_argument("--mode", default="formula", help="no checkpoint: the formula or a plain operator (teacher.MODES); "
                    "comma-separated: several, <out>_<mode>.json")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--tactic-net", action="store_true", help="the tactician's network (config tactic_ckpt) under it")
    ap.add_argument("--battles", type=int, default=None)
    ap.add_argument("--army", action="store_true", help="the armies' exam: 3-8 a side with a lord (cases.ARMY)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    P = load()
    unit_net = TT.unit_network(a.device)
    net = None
    if a.ckpt:
        net, _, _ = OT.restore(a.ckpt, a.device)
    tnet = None
    if a.tactic_net:
        from tools import config as project
        tnet, _, _ = TT.restore(str(project.ROOT / cfg()["tactic_ckpt"]), a.device)
        tnet.eval()
    if net is not None:
        return run(unit_net, P, a.device, net, a.mode, a.greedy, a.battles, out_path=a.out, tactic_net=tnet, army=a.army)
    modes = a.mode.split(",")
    for mode in modes:
        path = a.out if (a.out is None or len(modes) == 1) else a.out.replace(".json", f"_{mode}.json")
        run(unit_net, P, a.device, None, mode, a.greedy, a.battles, out_path=path, tactic_net=tnet, army=a.army)


if __name__ == "__main__":
    main()

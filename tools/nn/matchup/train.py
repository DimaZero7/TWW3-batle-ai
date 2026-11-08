"""The operative's training on the matchups (build/diag_m1/night.md): our side's operative network in the chain
(duel.Chain: the tacticians' formula with the group's target to the units, the frozen unit network) against a script,
the decisions stored and rebuilt as at the cases (operative/env.rebuild), the update operative/train.update: PPO plus
the teacher's term (operative/teacher.loss - the army's formula for a battle of 9 own units and more; every first
decision of a battle with first_weight, the rest a share with a falling weight). The reward at a battle's end: the
gold trade (the enemy's share lost less mine) + WIN_BONUS for a win, - WIN_BONUS for a loss; the teacher is the main signal
(imitation first: the user's 'skips are learned by imitation, not by a cost').

    DOCK_NAME=agent-operative bash tools/nn/dock.sh tools.nn.matchup.train --device cuda --minutes 35 --out build/diag_m1/run1
"""
import argparse
import contextlib
import json
import random
import time
from pathlib import Path

import torch

from tools.nn.matchup import armies as AR
from tools.nn.matchup import cfg
from tools.nn.matchup import duel as DU
from tools.nn.operative import cfg as ocfg
from tools.nn.operative import groups as GR
from tools.nn.operative import net as NN
from tools.nn.operative import teacher as TC
from tools.nn.operative import train as OT
from tools.nn.sim.params import load
from tools.nn.tactic import train as TT
from tools.nn.unit import train as UT

WIN_BONUS = 0.5


def rollout(name, enemy, B, net, critic, unit_net, P, device, rnd, limit=None):
    """(batch for operative/train.update, readings, stats)."""
    from tools.nn.operative import frame as FR
    from tools.nn.operative import inputs as OIN
    from tools.nn.operative import env as OEN
    battles = [AR.augment(AR.base(name), rnd) for _ in range(B)]
    chain = DU.Chain(1, op=net if net is not None else "formula", command="hold", op_greedy=False, targets=True)
    rec = {"raw": [], "groups": [], "act": [], "lp": [], "value": [], "valid": []}
    orig = chain.decide

    def decide(st, raw, K):
        if chain.axes is None:
            chain.axes = FR.Axes.of(raw, chain.sidev, None)
        chain.groups.drop_dead(raw)
        g0 = chain.groups.clone()
        inp = OIN.build(raw, chain.setup.card, chain.sidev, chain.axes, g0, chain.command, K=K)
        with torch.no_grad():
            v = critic(inp) if critic is not None else torch.zeros(st.B, device=raw.device)
        res = orig(st, raw, K)
        _, _, out = res
        if "lp" not in out:
            out["lp"] = torch.zeros(st.B, device=raw.device)
        rec["raw"].append(raw)
        rec["groups"].append(g0)
        rec["act"].append(out["act"])
        rec["lp"].append(out["lp"])
        rec["value"].append(v)
        rec["valid"].append(~st.done & inp["own"].any(1))
        return res
    chain.decide = decide
    rd, stats = DU.play(name, battles, {1: chain, 2: DU.enemy(enemy)}, unit_net, P, device, limit_s=limit)
    D = len(rec["raw"])
    valid = torch.stack(rec["valid"])
    reward = torch.zeros((D, B), device=device)
    last = (valid.float() * torch.arange(1, D + 1, device=device)[:, None]).argmax(0)          # [B]
    trade = rd["lost"][:, 1] - rd["lost"][:, 0] + WIN_BONUS * (rd["winner"] == 1).float() \
        - WIN_BONUS * (rd["winner"] == 2).float()
    reward[last, torch.arange(B, device=device)] = trade
    done = torch.zeros((D, B), dtype=torch.bool, device=device)
    done[last, torch.arange(B, device=device)] = True
    batch = {"raw": torch.stack(rec["raw"]), "groups": GR.Groups.stack(rec["groups"]),
             "act": {k: torch.stack([a[k] for a in rec["act"]]) for k in rec["act"][0]},
             "lp": torch.stack(rec["lp"]), "value": torch.stack(rec["value"]), "reward": reward, "done": done,
             "valid": valid, "last_value": torch.zeros(B, device=device), "card": chain.setup.card,
             "side": chain.sidev, "command": chain.command, "destroy": chain.destroy, "axes": chain.axes,
             "bounds": None, "K": OEN.rows_of(chain.setup.st.N)}
    return batch, rd, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    tc = ocfg()["train"]
    ap.add_argument("--battles", type=int, default=32)
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--init", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="build/diag_m1/run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--teach-updates", type=int, default=60)
    ap.add_argument("--enemies", default="nearest,nearest,ai_like")
    ap.add_argument("--matchups", default="m1,m2,m3")
    ap.add_argument("--no-lock", action="store_true")
    ap.add_argument("--limit", type=int, default=600, help="a battle's limit, s (the matchups' 900 for the exam)")
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)
    rnd = random.Random(a.seed)
    P = load()
    dev = a.device
    net, critic = NN.OperativeNet().to(dev), NN.OperativeCritic().to(dev)
    done = 0
    pa, pc = OT.params_of(net, critic)
    opt = torch.optim.Adam(pa + pc, lr=float(tc["lr"]))
    if a.init:
        net, critic, ck = OT.restore(a.init, dev)
        pa, pc = OT.params_of(net, critic)
        opt = torch.optim.Adam(pa + pc, lr=float(tc["lr"]))
        if not ck["padded"]:
            try:
                opt.load_state_dict(ck["opt"])
            except (ValueError, KeyError):        # (an imitation's checkpoint: its optimizer had no critic)
                pass
        done = int(ck.get("updates", 0))
    teach_from = done
    unit_net = TT.unit_network(dev)
    duty = UT.Duty(dev, 1.0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = a.matchups.split(",")
    enemies = a.enemies.split(",")
    start = time.time()
    with (TT.lock(dev, f"matchup {out.name}") if not a.no_lock else contextlib.nullcontext()):
        with (out / "log.jsonl").open("a", encoding="utf-8") as logf:
            k = 0
            while time.time() - start < a.minutes * 60:
                name, enemy = names[k % len(names)], enemies[(k // len(names)) % len(enemies)]
                k += 1
                t0 = time.time()
                net.eval()
                batch, rd, st = rollout(name, enemy, a.battles, net, critic, unit_net, P, dev, rnd, limit=a.limit)
                net.train()
                t1 = time.time()
                us = OT.update(net, critic, opt, batch, tc, dev, duty, TC.weight_at(done - teach_from, a.teach_updates))
                if torch.device(dev).type == "cuda":
                    torch.cuda.empty_cache()
                done += 1
                row = {"update": done, "matchup": name, "enemy": enemy, **DU.summary(rd),
                       "side1": st.get("side1"), **{k2: us[k2] for k2 in ("policy_loss", "value_loss", "kl", "teach_loss",
                                                                        "teach_weight", "agree", "samples")},
                       "rollout_s": round(t1 - t0, 1), "update_s": round(time.time() - t1, 1),
                       "minutes": round((time.time() - start) / 60, 2)}
                row["side1"] = {kk: vv for kk, vv in (row["side1"] or {}).items() if kk != "errors"}
                print(json.dumps(row), flush=True)
                logf.write(json.dumps(row) + "\n")
                logf.flush()
                OT.save(out / "latest.pt", net, critic, opt, tc, done)
    print(json.dumps({"updates": done, "minutes": round((time.time() - start) / 60, 1)}), flush=True)


if __name__ == "__main__":
    main()

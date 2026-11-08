"""The operative's imitation on the matchups (build/diag_m1/night.md): states of battles the teacher (the army's formula)
or the network played, each decision labelled by the teacher's formula (operative/teacher.loss: the cross-entropies of
every head against the formula's choices, 'keep' a full class), many passes over the stored states - no simulator in
the update. Rounds: the formula drives (behaviour cloning), then the network drives and the teacher labels (DAgger).

    DOCK_NAME=agent-operative bash tools/nn/dock.sh tools.nn.matchup.bc --device cuda --rounds 3 --out build/diag_m1/bc1
"""
import argparse
import contextlib
import json
import random
import time
from pathlib import Path

import torch

from tools.nn.matchup import train as MT
from tools.nn.matchup import duel as DU
from tools.nn.operative import cfg as ocfg
from tools.nn.operative import env as OEN
from tools.nn.operative import net as NN
from tools.nn.operative import teacher as TC
from tools.nn.operative import train as OT
from tools.nn.sim.params import load
from tools.nn.tactic import train as TT


def collect(names, enemies, B, net, unit_net, P, dev, rnd, limit):
    """[batch] of rollouts: each matchup against each enemy once (net None: the formula drives)."""
    out = []
    for name in names:
        for enemy in enemies:
            t0 = time.time()
            batch, rd, st = MT.rollout(name, enemy, B, net, None, unit_net, P, dev, rnd, limit=limit)
            batch["name"], batch["enemy"] = name, enemy
            out.append(batch)
            print(json.dumps({"collect": name, "enemy": enemy, "driver": "formula" if net is None else "network",
                              **DU.summary(rd), "side1": {k: v for k, v in (st.get("side1") or {}).items()
                                                          if k != "errors"}, "s": round(time.time() - t0, 1)}), flush=True)
    return out


def samples(batches):
    """[(batch index, di, bi)] of the valid decisions."""
    out = []
    for k, b in enumerate(batches):
        d, bb = b["valid"].nonzero(as_tuple=True)
        out += [(k, int(x), int(y)) for x, y in zip(d.tolist(), bb.tolist())]
    return out


def train_pass(net, opt, batches, idx, mb, dev):
    """One pass: minibatches drawn each from one rollout (one forward a step - the walk over groups and rows costs the
    same for 30 samples as for 500), in a random order."""
    rows = []
    by = {}
    for k, d, b in idx:
        by.setdefault(k, []).append((d, b))
    chunks = []
    for k, lst in by.items():
        random.shuffle(lst)
        chunks += [(k, lst[i:i + mb]) for i in range(0, len(lst), mb)]
    random.shuffle(chunks)
    pa = [p for p in net.parameters() if p.requires_grad]
    for k, chunk in chunks:
        batch = batches[k]
        di = torch.tensor([d for d, _ in chunk], device=dev)
        bi = torch.tensor([b for _, b in chunk], device=dev)
        opt.zero_grad()
        inp, ctx, _ = OEN.rebuild(batch, di, bi)
        term, st = TC.loss(net, inp, ctx, batch["destroy"][bi])
        st.pop("keep_part", None)
        term.backward()
        torch.nn.utils.clip_grad_norm_(pa, 1.0)
        opt.step()
        rows.append({"loss": float(term.detach()), **{h: v for h, v in st.items() if v is not None}})
    keys = rows[0].keys()
    return {k: round(sum(r.get(k, 0.0) for r in rows) / len(rows), 4) for k in keys}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--battles", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--matchups", default="m1,m2,m3")
    ap.add_argument("--enemies", default="nearest,ai_like")
    ap.add_argument("--init", default=None)
    ap.add_argument("--minutes", type=float, default=0.0, help="stop after this many minutes (0: the rounds)")
    ap.add_argument("--out", default="build/diag_m1/bc")
    ap.add_argument("--data", default="", help="comma-separated data_*.pt of earlier runs (the states, loaded first)")
    ap.add_argument("--first-passes", type=int, default=0, help="passes over the loaded data before the rounds")
    ap.add_argument("--formula-first", action="store_true", help="round 0 driven by the formula even with --init")
    ap.add_argument("--no-lock", action="store_true")
    a = ap.parse_args(argv)
    torch.manual_seed(0)
    rnd = random.Random(0)
    P = load()
    dev = a.device
    net, critic = NN.OperativeNet().to(dev), NN.OperativeCritic().to(dev)
    if a.init:
        net, critic, _ = OT.restore(a.init, dev)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=a.lr)
    unit_net = TT.unit_network(dev)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names, enemies = a.matchups.split(","), a.enemies.split(",")
    start = time.time()
    data = []
    for f in [x for x in a.data.split(",") if x]:
        data += [{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()} for b in torch.load(f, weights_only=False)]
    for b in data:
        for k in ("groups", "axes"):
            if hasattr(b[k], "__dataclass_fields__"):
                b[k] = type(b[k])(*(getattr(b[k], f).to(dev) for f in b[k].__dataclass_fields__))
    with (TT.lock(dev, f"matchup bc {out.name}") if not a.no_lock else contextlib.nullcontext()):
        with (out / "log.jsonl").open("a", encoding="utf-8") as logf:
            for p_ in range(a.first_passes if data else 0):
                res = train_pass(net, opt, data, samples(data), a.minibatch, dev)
                row = {"round": -1, "pass": p_, **res, "minutes": round((time.time() - start) / 60, 1)}
                print(json.dumps(row), flush=True)
                logf.write(json.dumps(row) + "\n")
                OT.save(out / "latest.pt", net, critic, opt, ocfg()["train"], p_ + 1)
            for r in range(a.rounds):
                driver = None if (r == 0 and (a.formula_first or (not a.init and not data))) else net
                net.eval()
                new = collect(names, enemies, a.battles, driver, unit_net, P, dev, rnd, a.limit)
                torch.save([{k: v for k, v in b.items()} for b in new], out / f"data_r{r}.pt")
                data += new
                net.train()
                idx = samples(data)
                for p_ in range(a.passes):
                    t0 = time.time()
                    res = train_pass(net, opt, data, list(idx), a.minibatch, dev)
                    row = {"round": r, "pass": p_, "samples": len(idx), **res, "s": round(time.time() - t0, 1),
                           "minutes": round((time.time() - start) / 60, 1)}
                    print(json.dumps(row), flush=True)
                    logf.write(json.dumps(row) + "\n")
                    logf.flush()
                    OT.save(out / "latest.pt", net, critic, opt, ocfg()["train"], r * a.passes + p_ + 1)
                    if torch.device(dev).type == "cuda":
                        torch.cuda.empty_cache()
                    if a.minutes and time.time() - start > a.minutes * 60:
                        break
                if a.minutes and time.time() - start > a.minutes * 60:
                    break
    print(json.dumps({"done": True, "minutes": round((time.time() - start) / 60, 1)}), flush=True)


if __name__ == "__main__":
    main()

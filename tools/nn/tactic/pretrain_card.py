"""Is a card of 8 numbers on the map enough? (design/tactic "Карточка на карте": the squeeze 128 -> 8 is trained on its
own to predict ~10 facts of the unit; class >= 95 % and the shares' error <= 10 % - enough; else 16.)

The rows come from simulator battles of the v3 army generator (tools/nn/armies/generate.py; both sides 'nearest'), a
snapshot every few seconds, random correct parcels (parcel.random) - the rows exactly as the tactician gets them
(rows.build). The facts of a row: the class (caste, 13), the body size (5), the share of men left, fatigue, running,
ammunition, in melee, hidden, the run speed, the armour. The model: 'Анализ отряда' (185 -> 256 -> 128) -> the squeeze
(128 -> k) -> a small reader (k -> 64 -> the facts), trained together (the analysis is not trained yet: what is tested
is whether k numbers carry the facts); scored on rows of other battles.

    python -m tools.nn.tactic.pretrain_card [--card 8 16] [--battles 64] [--seconds 90] [--steps 3000]
Writes build/diag_tactic/card_pretrain/result.json and card<k>.pt (the analysis and the squeeze).
"""
import argparse
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools import config as project
from tools.nn.tactic import net as NT
from tools.nn.tactic import parcel as PC
from tools.nn.tactic import raw as TR
from tools.nn.tactic import rows as RW
from tools.nn.tactic import zone as ZN

OUT = project.ROOT / "build" / "diag_tactic" / "card_pretrain"
CASTE = (36, 49)          # the passport's caste positions (passport.features)
SIZE = (49, 54)           # the body size positions
SHARES = ("men", "fat", "ammo", "speed", "armour")
FLAGS = ("f", "m", "hidden")
LIMITS = {"class": 0.95, "size": 0.95, "share_mae": 0.10, "flag_err": 0.10}


def facts(rows):
    """{class [M], size [M] long, shares [M, 5], flags [M, 3]} of rows [M, 185]."""
    ix = RW.RI
    return {"class": rows[:, CASTE[0]:CASTE[1]].argmax(1), "size": rows[:, SIZE[0]:SIZE[1]].argmax(1),
            "shares": torch.stack([rows[:, ix["men"]], rows[:, ix["fat"]], rows[:, ix["ammo"]], rows[:, 5],
                                   rows[:, 16]], 1).clamp(0, 1),
            "flags": torch.stack([rows[:, ix["f"]], rows[:, ix["m"]], rows[:, ix["hidden_on"]]], 1)}


def collect(seeds, seconds, every, parcels, device, gen):
    """[M, 185] rows of the tactician (mask True) from battles of the seeds."""
    from tools.nn.armies import generate
    from tools.nn.sim import battle, scenario
    from tools.nn.sim.params import load
    from tools.nn.train import opponents
    from tools.nn.unit import train as T
    P = load()
    arenas = [generate.battle(s, armies="v3") for s in seeds]
    armies = [scenario.from_arena(a, "attack" if i % 2 == 0 else "defend") for i, a in enumerate(arenas)]
    st = scenario.build(armies, P, device=device)
    card = T.cards(st, device).float()
    seer = TR.Seer(P)
    out = []
    for s in range(seconds):
        for _ in range(round(1.0 / P.dt)):
            battle.step(st, opponents.nearest(st), P)
        raw = seer.raw(st)
        if (s + 1) % every:
            continue
        rc = raw.cpu()
        ps = []
        for b in range(st.B):
            for _ in range(parcels):
                p = PC.random(rc, gen, b=b)
                if p is not None:
                    ps.append(p)
        if not ps:
            continue
        p = PC.Parcel.cat(ps)
        r = RW.build(rc, card.cpu(), p, ZN.Tiles.of(p))
        out.append(r["rows"][r["mask"]])
    return torch.cat(out)


class Reader(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.analysis = NT.UnitAnalysis()
        self.squeeze = NT.CardSqueeze(out=k)
        self.read = nn.Sequential(nn.Linear(k, 64), nn.ReLU(), nn.Linear(64, 13 + 5 + len(SHARES) + len(FLAGS)))

    def forward(self, rows):
        h = self.read(self.squeeze(self.analysis(rows)))
        return h[:, :13], h[:, 13:18], torch.sigmoid(h[:, 18:18 + len(SHARES)]), h[:, 18 + len(SHARES):]


def loss_of(model, rows):
    f = facts(rows)
    c, s, sh, fl = model(rows)
    return (F.cross_entropy(c, f["class"]) + F.cross_entropy(s, f["size"]) + 10 * F.l1_loss(sh, f["shares"])
            + F.binary_cross_entropy_with_logits(fl, f["flags"]))


@torch.no_grad()
def score(model, rows):
    f = facts(rows)
    c, s, sh, fl = model(rows)
    mae = (sh - f["shares"]).abs().mean(0)
    err = ((fl > 0).float() != f["flags"]).float().mean(0)
    out = {"class": float((c.argmax(1) == f["class"]).float().mean()), "size": float((s.argmax(1) == f["size"]).float().mean()),
           **{f"mae_{n}": float(v) for n, v in zip(SHARES, mae)}, **{f"err_{n}": float(v) for n, v in zip(FLAGS, err)},
           **{f"base_{n}": float(v) for n, v in zip(FLAGS, f["flags"].mean(0))}}
    out["enough"] = bool(out["class"] >= LIMITS["class"] and out["size"] >= LIMITS["size"]
                         and float(mae.max()) <= LIMITS["share_mae"] and float(err.max()) <= LIMITS["flag_err"])
    return out


def train(k, data, test, steps, device, seed=0, batch=4096, lr=1e-3):
    torch.manual_seed(seed)
    model = Reader(k).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator(device=device).manual_seed(seed)
    for step in range(steps):
        idx = torch.randint(data.shape[0], (batch,), device=device, generator=g)
        loss = loss_of(model, data[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model, score(model, test)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--card", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--battles", type=int, default=64)
    ap.add_argument("--seconds", type=int, default=90)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--parcels", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    t0 = time.time()
    gen = torch.Generator().manual_seed(7)
    data = collect(range(100, 100 + a.battles), a.seconds, a.every, a.parcels, a.device, gen).to(a.device)
    test = collect(range(900_000, 900_000 + a.battles // 2), a.seconds, a.every, a.parcels, a.device, gen).to(a.device)
    t1 = time.time()
    res = {"rows_train": int(data.shape[0]), "rows_test": int(test.shape[0]), "collect_s": round(t1 - t0, 1),
           "limits": LIMITS, "cards": {}}
    OUT.mkdir(parents=True, exist_ok=True)
    for k in a.card:
        model, sc = train(k, data, test, a.steps, a.device)
        res["cards"][str(k)] = sc
        torch.save({"k": k, "analysis": model.analysis.state_dict(), "squeeze": model.squeeze.state_dict(),
                    "score": sc}, OUT / f"card{k}.pt")
        print(k, json.dumps(sc), flush=True)
    res["train_s"] = round(time.time() - t1, 1)
    (OUT / "result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "cards"}))


if __name__ == "__main__":
    main()

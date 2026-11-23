"""S0-1's label probe on the network's REAL input (tmp/drill_s1/design_opus.md 6 'label probe', design_merged 5,
tmp/drill_s1/brief_C3.md X3): can StratNet1 see the cause, and only the cause? The engine's probe (net/drill/probe.py,
O0-1's) reused for its pieces: its share(); the fit loop is the engine's own (net/drill/fit.passes) on this level
(net/drill/levels/strategist.py).

    collect    probe_train (n cause scenes + n twins + the general share, s1_scenes; its own ground pool) and probe_held
               (the same form, its own pool and seeds): inputs at tick 0 and the teacher's labels, one store.Store a set,
               keyed by the specs' identities and the code: a repeated collect of the same key is kept, never appended;
               another key in the same folder is refused (a new --out)
    fit        a fresh StratNet1 a seed and arm fitted on probe_train (levels/strategist.balanced), read after every pass
               on probe_held's UNFORCED rows (>= 1 valid candidate: the cause scenes and the twins 'role', 'low', 'shift'
               - the low and shift twins keep the cause scene's candidate geometry, s1_scenes; the general battles with a
               candidate); the forced rows (KEEP the only legal choice) are printed apart and not gated:
        intact     CREST labels followed (the teacher's candidate) >= INTACT and KEEP labels kept >= INTACT, each n >= N_MIN
        ablated    refitted with the card's columns zeroed (cards/s0_1.PROBE_ABLATE: the heights, the candidates' band and
                   profile, the mission): its accuracy <= the majority class's share (unforced, probe_held) + BAND
               the gate holds on EVERY seed at the last pass AND at the pass of the lowest training loss; exit 1 if not
    python -m net.strategist.probe_s1 collect --out DIR [--n 512 --n-held 256]
    python -m net.strategist.probe_s1 fit --dir DIR [--passes 12 --lr 1e-3 --mb 256 --seeds 5,6,7 --device cuda]
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

INTACT, BAND, N_MIN = 0.90, 0.05, 50
_WHY = ("design_opus 6 + brief_C3 X3: on probe_held's unforced rows intact >= 0.9; the card's columns erased: not above "
        "the majority share + 0.05; n >= 50 a class")


def _cols(names, table):
    out = []
    for nm in names:
        if nm.endswith("*"):
            out += [i for i, c in enumerate(table) if c.startswith(nm[:-1])]
        else:
            out.append(table.index(nm))
    return out


def ablate(obs, spec=None):
    """obs with the card's PROBE_ABLATE columns zeroed (by name)."""
    from net.drill.cards import s0_1 as CARD
    from net.strategist import s1_net as SN
    from net.tactic import rows as RW
    spec = CARD.PROBE_ABLATE if spec is None else spec
    tables = {"rows": list(RW.ROW), "place": SN.PLACE_COLS, "cand": SN.CAND_COLS, "seat": SN.SEAT_COLS}
    o = dict(obs)
    for key, names in spec.items():
        t = o[key].clone()
        t[..., _cols(names, tables[key])] = 0.0
        o[key] = t
    return o


def specs_of(name, n):
    from net.drill.cards import s0_1 as CARD
    from net.strategist import s1_scenes as SS
    return SS.scenes_at(CARD.SETS[name], n, name) + [SS.general(CARD.SETS[name + "_general"] + i, name)
                                                      for i in range(n // 2)]


def _key(specs):
    from net.strategist import exam_s1 as EX
    from net.strategist import s1_scenes as SS
    h = hashlib.sha256(json.dumps([SS.identity(s) for s in specs], default=str).encode()).hexdigest()[:16]
    return f"{h}-{EX._tree_sha(EX.KEY_TREES)}"


def collect(a):
    from net.drill import store as ST
    from net.drill.levels import strategist as LV
    from net.strategist import s1_net as SN
    from net.strategist import s1_scenes as SS
    D = Path(a.out)
    for name, n in (("probe_train", a.n), ("probe_held", a.n_held)):
        t0 = time.time()
        specs = specs_of(name, n)
        key = _key(specs)
        st = ST.Store(D / name, run_key=key)                     # (another key here: StoreRefused)
        if st.manifest.get("shards"):
            print(name, "kept (same key)", key, flush=True)
            continue
        obs = SN.observe(specs, a.device)
        labels = [SS.label(s) for s in specs]
        rows = LV.rows_of(specs, obs, labels)
        st.put(name, rows, [{"seed": s["seed"], "start": LV.kind_of(s, k)} for s, k in zip(specs, labels)])
        print(name, len(specs), "battles", LV.census(rows), round(time.time() - t0), "s", flush=True)


def unforced(rows):
    """[M] bool: a row with >= 1 legal candidate besides KEEP."""
    return rows["legal"][:, 0, 1:].any(1)


def read(net, rows, abl, device):
    from net.drill import probe as PB
    k = _choices(net, rows, abl, device)
    te = rows["teacher"]
    out = {}
    for tag, sel in (("unforced", unforced(rows)), ("forced", ~unforced(rows))):
        kk, tt = k[sel], te[sel]
        n = int(sel.sum())
        maj = max(float((tt == 0).float().mean()), float((tt > 0).float().mean())) if n else None
        out[tag] = {"followed": PB.share(int(((kk == tt) & (tt > 0)).sum()), int((tt > 0).sum())),
                    "kept": PB.share(int(((kk == 0) & (tt == 0)).sum()), int((tt == 0).sum())),
                    "accuracy": round(float((kk == tt).float().mean()), 4) if n else None,
                    "majority": None if maj is None else round(maj, 4), "n": n}
    return out


@torch.no_grad()
def _choices(net, rows, abl, device):
    from net.drill.levels import strategist as LV
    net.eval()
    out = []
    for i in range(0, rows["b"].shape[0], 512):
        idx = torch.arange(i, min(i + 512, rows["b"].shape[0]))
        b = LV.to_batch(rows, idx)
        if abl:
            b.obs = ablate(b.obs)
        out.append(net(LV._obs(b.to(device))).argmax(1).cpu())
    return torch.cat(out)


def fit_arm(tr, held, abl, a, seed, log):
    from net.drill import fit as FT
    from net.drill.levels import strategist as LV
    from net.strategist import s1_net as SN
    torch.manual_seed(seed)
    net = SN.StratNet1().to(a.device)
    opt = torch.optim.Adam([q for q in net.parameters() if q.requires_grad], lr=a.lr)
    gen = torch.Generator().manual_seed(seed + 1)
    w = LV.balanced(tr)

    def batch_of(rows, cards, idx):
        b = LV.to_batch(rows, idx)
        if abl:
            b.obs = ablate(b.obs)
        return b
    curve = []
    for p in range(a.passes):
        hist, _ = FT.passes(LV, net, opt, tr, None, w, a.device, lambda x: None, gen, 1, a.mb, 1.0, batch_of=batch_of)
        r = {"arm": "ablated" if abl else "intact", "seed": seed, "pass": p, "loss": hist[-1],
             **read(net, held, abl, a.device)}
        curve.append(r)
        log(r)
    return curve


def _gate(r, abl):
    u = r["unforced"]
    if abl:
        return u["accuracy"] is not None and u["accuracy"] <= u["majority"] + BAND
    f, k = u["followed"], u["kept"]
    return f["n"] >= N_MIN and k["n"] >= N_MIN and f["p"] >= INTACT and k["p"] >= INTACT


def verdict(runs):
    out, ok = {}, True
    for sd, r in runs.items():
        v = {}
        for arm in ("intact", "ablated"):
            c = r[arm]
            best = min(range(len(c)), key=lambda k: c[k]["loss"])
            for tag, k in (("last", len(c) - 1), ("best_loss", best)):
                g = _gate(c[k], arm == "ablated")
                v[f"{arm}_{tag}"] = {**{x: c[k][x] for x in ("pass", "loss", "unforced", "forced")}, "gate": g}
                ok = ok and g
        out[str(sd)] = v
    n_ok = all(c["unforced"]["followed"]["n"] >= N_MIN and c["unforced"]["kept"]["n"] >= N_MIN
               for r in runs.values() for c in r["intact"][-1:])
    return {"seeds": out, "n_ok": n_ok, "n_min": N_MIN, "pass": ok and n_ok, "lines_why": _WHY}


def fit(a):
    from net.drill import store as ST
    D = Path(a.dir)
    tr, _, _ = ST.Store(D / "probe_train").mixture()
    held, _, _ = ST.Store(D / "probe_held").mixture()
    logf = open(D / f"{a.name}.log", "a", encoding="utf-8")

    def log(x):
        s = json.dumps(x, default=str)
        print(s, flush=True)
        logf.write(s + "\n")
    runs = {}
    for sd in [int(x) for x in a.seeds.split(",")]:
        runs[sd] = {"intact": fit_arm(tr, held, False, a, sd, log), "ablated": fit_arm(tr, held, True, a, sd, log)}
    v = verdict(runs)
    (D / f"{a.name}.json").write_text(json.dumps({"verdict": v, "runs": runs, "args": vars(a)}, indent=1, default=str),
                                      encoding="utf-8")
    print("VERDICT", json.dumps(v, default=str))
    return 0 if v["pass"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("collect", "fit"))
    ap.add_argument("--out", default="tmp/drill_s1/c/probe")
    ap.add_argument("--dir", default="tmp/drill_s1/c/probe")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--n-held", type=int, default=256)
    ap.add_argument("--passes", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mb", type=int, default=256)
    ap.add_argument("--seeds", default="5,6,7")
    ap.add_argument("--name", default="probe")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    if a.what == "collect":
        collect(a)
        return
    sys.exit(fit(a))


if __name__ == "__main__":
    main()

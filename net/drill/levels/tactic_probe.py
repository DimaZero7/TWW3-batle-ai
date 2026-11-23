"""The tactic's label probe through the engine (METHOD step 4; the card's PROBE): can a small model on the network's real
input repeat the teacher's approach, and does it lose it when the cause's columns are erased?

    python -m net.drill.levels.tactic_probe --card t0_2 --out DIR [--parent PATH] [--smoke N] [--chunk 256]
        DIR/probe.json  {"card", "verdict": {pass, n_ok, fails, intact, erased, n}, ...} - the file the M2 verdict reads
                        (cards/t0_2.probe_piece: pass -> pass; fail or n_ok false -> NOT JUDGED, BLOCKED)

    starts      the card's PROBE.train / PROBE.held registry entries (probe_train / probe_held: S2 + S2N twins of each
                seed, the held axes PROBE.held_axis_n a set); seeds.check_specs and assert_starts on every start
    path        the teacher (card.teacher) driving, recorded by tactic_v's VBuffer into validate.TeacherCache (played
                once a key, its first chunk replayed and compared on reuse)
    labels      masks.of("tactic_approach", batch).eligible - decided, an attack with an approach, not at the rule's
                threshold, its row open; the label the teacher's approach
    the model   the parent network's encoder FROZEN (cards after the council on the rebuilt input, the order in force,
                cols, own, g_row, g_on) and a fresh heads.SideHead on (card_g, own_g, card_k, pair(g, k), x_k) - the
                side head's own input; trained by the legacy recipe R3 (learn.side_probe_train: Adam 1e-3 cosine to
                1e-5, 160 passes, minibatch 128, the last state)
    intact      held: followed per class (front / left / right) >= PROBE.followed, S2N false flank <= PROBE.false_flank
    erased      the cause's columns (PROBE.cause) zeroed in training and reading; on the complete twin pairs (S2 and
                its S2N both with an eligible label) whose FIRST labels differ, the share repeated <= erased_chance +
                erased_over; the intact model's share on the same pairs printed beside
    n_ok        every class >= cards.MARK_MIN_N held labels and >= MARK_MIN_N differing pairs
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from net.drill import masks as MK
from net.drill import record as RC
from net.drill.levels import tactic as LT

ROOT = Path(__file__).resolve().parents[3]


def specs_of(card_mod, smoke=0):
    """{split: [spec]} of the card's probe ranges (registry), twins S2 / S2N of each seed."""
    from net.drill import seeds as SD
    from net.drill import run as RN
    from net.tactic import imitate as IM
    from net.tactic import scenes as SN
    pr = card_mod.PROBE
    ent = {e["name"]: e for e in SD.load(RN.REGISTRY)}
    tr = [ent[n] for n in pr["train"]]
    hd = [ent[n] for n in pr["held"]]
    n_tr = [e["hi"] - e["lo"] for e in tr]
    out = {"probe_train": [s for e, n in zip(tr, n_tr) for s in IM.s2_specs(e["lo"], smoke or n)]}
    held = []
    for e in hd:
        if e["name"] == "probe held axes":
            held += SN.held_specs_side(seed0=e["lo"], n=2 if smoke else int(pr["held_axis_n"]))
        else:
            held += IM.s2_specs(e["lo"], max(2, smoke // 2) if smoke else e["hi"] - e["lo"])
    out["probe_held"] = held
    SD.check_specs(out)
    entries = SD.load(RN.REGISTRY)
    for split, sp in out.items():
        SD.assert_starts(entries, {"t0_2"}, split, [(SD._gen_of(s), None, int(s["seed"])) for s in sp])
    return out


def parts_of(specs, chunk, base):
    from net.tactic import learn as LN
    ch = min(chunk, len(specs))
    return [("scene", specs, True, LN._seeds(base, len(specs), ch), ch)]


def cause_cols():
    from net.tactic import cols as CL
    names = set(CL.SIDE_PAIR) | set(CL.SIDE_ROW)
    return [i for i, n in enumerate(CL.D2) if n in names]


def erase(batch, cc):
    o = dict(batch.obs)
    c = o["cols"].clone()
    c[..., cc] = 0
    o["cols"] = c
    if "s2" in o:
        o["s2"] = torch.zeros_like(o["s2"])
    aux = dict(batch.aux)
    aux["x_k"] = torch.zeros_like(aux["x_k"])
    return RC.Batch(**{**batch.as_dict(), "obs": o, "aux": aux})


def feats(net, rows, cards, erased, dev, mb=256):
    """{v [N, F] the side head's input at each eligible label, ap [N], b [N], sec [N], g [N]}; F = (cg, own, ck,
    pair, x) widths in 'dims'."""
    from net.tactic import learn as LN
    cc = cause_cols()
    M = int(rows["b"].shape[0])
    out = {k: [] for k in ("v", "ap", "b", "sec", "g")}
    dims = None
    net.eval()
    for i in range(0, M, mb):
        idx = torch.arange(i, min(M, i + mb))
        batch = RC.from_netbuffer({**rows, "cards": cards}, idx).to(dev)
        el = MK.of("tactic_approach", batch).eligible
        if not bool(el.any()):
            continue
        if erased:
            batch = erase(batch, cc)
        with torch.no_grad():
            enc = LT._enc(net, batch, LT.rebuild(batch))
            cs = enc["cards"]
            W = cs.shape[-1]
            o = batch.obs
            g_row = o["g_row"].long()
            lab = RC.label_of(batch)
            cg = cs.gather(1, g_row[..., None].expand(-1, -1, W))
            ck = cs.gather(1, lab.clamp(min=0)[..., None].expand(-1, -1, W))
            G = g_row.shape[1]
            parts = [cg, o["own"][:, :G].float(), ck, LN.at_label(o["pair"].float(), lab), batch.aux["x_k"].float()]
        dims = [int(p.shape[-1]) for p in parts]
        r, g = el.nonzero(as_tuple=True)
        out["v"].append(torch.cat([p[r, g] for p in parts], -1).cpu())
        out["ap"].append(batch.aux["ap"].long()[r, g].cpu())
        out["b"].append(batch.b[r].cpu())
        sec = rows["v_sec32"][idx] if "v_sec32" in rows else rows["sec"][idx]
        out["sec"].append(sec.to(dev)[r].long().cpu())
        out["g"].append(g.cpu())
    res = {k: torch.cat(v, 0) if v else torch.zeros(0) for k, v in out.items()}
    res["dims"] = dims
    return res


def head_of(dims):
    from net.tactic import heads as HD
    W, own, _, pair, x = dims
    return HD.SideHead(W, own=own, pair=pair, extra=x)


def split_v(h, v, dims):
    cut = torch.tensor(dims).cumsum(0).tolist()
    a = [0] + cut[:-1]
    return h(*[v[:, s:e] for s, e in zip(a, cut)])


def train(d, dev, log, seed=17, passes=160, mb=128):
    """learn.side_probe_train's recipe R3 on the precomputed inputs (every eligible label, weight 1)."""
    from net.tactic import heads as HD
    torch.manual_seed(seed)
    h = head_of(d["dims"]).to(dev)
    opt = torch.optim.Adam(h.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=passes, eta_min=1e-5)
    gen = torch.Generator().manual_seed(seed)
    v, ap = d["v"].float(), d["ap"]
    N = int(ap.shape[0])
    for p in range(passes):
        h.train()
        perm = torch.randperm(N, generator=gen)
        tot, nb = 0.0, 0
        for i in range(0, N, mb):
            j = perm[i:i + mb]
            lg = split_v(h, v[j].to(dev), d["dims"])
            loss = HD.side_loss(lg, ap[j].to(dev), torch.ones(j.numel(), device=dev))[0]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(h.parameters(), 1.0)
            opt.step()
            tot, nb = tot + float(loss.detach()), nb + 1
        sched.step()
        if p % 20 == 0 or p == passes - 1:
            log({"probe_pass": p, "loss": round(tot / max(nb, 1), 5)})
    return h.eval()


def picks(h, d, dev, mb=4096):
    out = []
    with torch.no_grad():
        for i in range(0, int(d["ap"].shape[0]), mb):
            out.append(split_v(h, d["v"][i:i + mb].float().to(dev), d["dims"]).argmax(-1).cpu())
    return torch.cat(out) if out else torch.zeros(0, dtype=torch.long)


def _share(k, n):
    return {"share": None if n == 0 else round(k / n, 4), "n": int(n)}


def intact_table(d, pk, meta):
    rec = [m.get("recipe") for m in meta]
    s2n = torch.tensor([rec[int(b)] == "S2N" for b in d["b"]], dtype=torch.bool)
    ap = d["ap"]
    t = {c: _share(int(((ap == i) & (pk == i)).sum()), int((ap == i).sum())) for i, c in enumerate(("front", "left", "right"))}
    f = s2n & (ap == 0)
    t["S2N false flank"] = _share(int((f & (pk > 0)).sum()), int(f.sum()))
    return t


def first_labels(d, meta, specs):
    """{(seed, held): {recipe: (ap, index)}} the earliest eligible label of each battle (sec, then g)."""
    best = {}
    for i in range(int(d["ap"].shape[0])):
        b = int(d["b"][i])
        key = (int(d["sec"][i]), int(d["g"][i]))
        if b not in best or key < best[b][0]:
            best[b] = (key, i)
    out = {}
    for b, (_, i) in best.items():
        s = specs[int(meta[b]["spec"])]
        out.setdefault((int(s["seed"]), s.get("held")), {})[s["recipe"]] = i
    return out


def pair_table(d, pk, meta, specs):
    """On the twin pairs MATCHED in time: the S2's first label against its S2N twin's label of the same unit at the same
    second (both eligible) - all of them, and those whose labels differ: the share of the two labels repeated, with two
    errors by pairs. (tmp/engine/a/r7/probe_ablate*.json, 23.11: the twins' own first labels were 28 s apart on average
    and at different seconds in 88 of 143 differing pairs; there the erased model read the moment from my unit's and the
    target's rows and their pair - 0.671 - against 0.545 at the same second: the test compared different moments.)"""
    fl = first_labels(d, meta, specs)
    at = {(int(d["b"][i]), int(d["sec"][i]), int(d["g"][i])): i for i in range(int(d["ap"].shape[0]))}
    bof = {}
    for b, m in enumerate(meta):
        s = specs[int(m["spec"])]
        bof[(int(s["seed"]), s.get("held"), s["recipe"])] = b
    full = []
    for (seed, held), x in fl.items():
        if "S2" in x and (seed, held, "S2N") in bof:
            a = x["S2"]
            j = at.get((bof[(seed, held, "S2N")], int(d["sec"][a]), int(d["g"][a])))
            if j is not None:
                full.append((a, j))
    dif = [(a, b) for a, b in full if int(d["ap"][a]) != int(d["ap"][b])]

    def rep(prs):
        sc = [(int(int(pk[a]) == int(d["ap"][a])) + int(int(pk[b]) == int(d["ap"][b]))) / 2 for a, b in prs]
        out = _share(int(round(2 * sum(sc))), 2 * len(prs))
        n = len(sc)
        if n >= 2:                                     # (two errors by pairs: the pairs are the independent draws)
            m = sum(sc) / n
            out["two_errors"] = round(2 * (sum((x - m) ** 2 for x in sc) / (n - 1)) ** 0.5 / n ** 0.5, 4)
        return out
    return {"complete_pairs": len(full), "differing_pairs": len(dif), "repeated_all": rep(full),
            "repeated_differing": rep(dif)}


def verdict(card_mod, it, er_pairs, in_pairs):
    from net.drill.cards import MARK_MIN_N
    pr = card_mod.PROBE
    fails = []
    for c in ("front", "left", "right"):
        x = it[c]
        if x["share"] is None or x["share"] < float(pr["followed"]):
            fails.append(f"followed {c} {x['share']} (n {x['n']})")
    x = it["S2N false flank"]
    if x["share"] is None or x["share"] > float(pr["false_flank"]):
        fails.append(f"S2N false flank {x['share']} (n {x['n']})")
    line = float(pr["erased_chance"]) + float(pr["erased_over"])
    x = er_pairs["repeated_differing"]
    if x["share"] is None or x["share"] > line:
        fails.append(f"erased: repeated on the differing twin pairs {x['share']} > {line} (pairs "
                     f"{er_pairs['differing_pairs']})")
    n_ok = all(it[c]["n"] >= MARK_MIN_N for c in ("front", "left", "right")) and \
        er_pairs["differing_pairs"] >= MARK_MIN_N
    return {"pass": not fails, "n_ok": n_ok, "fails": fails, "intact": it,
            "erased": {"line": line, **er_pairs, "intact_on_the_same_pairs": in_pairs},
            "n": {c: it[c]["n"] for c in ("front", "left", "right")}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parent", default="weights/tactic/attack1/final.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sim", default="")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--passes", type=int, default=160)
    a = ap.parse_args(argv)
    from net.drill import cards as CD
    from net.drill.levels import tactic_v as TV
    from net.tactic import ckpt as TK
    from net.tactic import exam_pin as EP
    from net.tactic import learn as LN
    from sim.params import load
    t0 = time.time()
    import importlib
    card = CD.get(a.card)
    cm = importlib.import_module(f"net.drill.cards.{a.card}")
    D = Path(a.out)
    D.mkdir(parents=True, exist_ok=True)
    logf = open(D / "probe.log", "a", encoding="utf-8")

    def log(x):
        s = x if isinstance(x, str) else json.dumps(x, default=str)
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    P = load()
    rt = SimpleNamespace(device=a.device, sim=a.sim)
    sp = specs_of(cm, a.smoke)
    if a.smoke:
        for x in sp.values():
            for s in x:
                s["limit"] = min(int(s["limit"]), 150)
    tseed = int(LN._c()["probe"]["seeds"]["tseed"]) + 7000
    net, _ = TK.from_v2(str(ROOT / a.parent), a.device)
    psha = hashlib.sha256((ROOT / a.parent).read_bytes()).hexdigest()
    log({"probe": card.id, "battles": {k: len(v) for k, v in sp.items()}, "parent": a.parent, "parent_sha": psha,
         "sim_key": EP.sim_key(), "cause_cols": cause_cols(), "start": time.strftime("%H:%M")})
    data = {}
    for j, (split, specs) in enumerate(sp.items()):
        parts = parts_of(specs, a.chunk, tseed + 500 * j)
        key = TV.v_key(True, a.smoke, a.chunk, a.device, parts, card.teacher) + f"-{split}"
        rows, cards, meta, info, how = TV.teacher_path(D / "cache", key, parts, rt, P, card.teacher, True, log)
        log({"split": split, "teacher_path": how, "rows": int(rows["b"].shape[0]), "battles": len(meta),
             "min": round((time.time() - t0) / 60, 1)})
        data[split] = {erased: feats(net, rows, cards, erased, a.device) for erased in (False, True)}
        data[split]["meta"], data[split]["specs"] = meta, specs
        log({"split": split, "labels": {c: int((data[split][False]["ap"] == i).sum())
                                        for i, c in enumerate(("front", "left", "right"))}})
    res = {"card": card.id, "parent_sha": psha, "sim_key": EP.sim_key(), "smoke": a.smoke, "lines": cm.PROBE}
    hs = {}
    for erased in (False, True):
        h = train(data["probe_train"][erased], a.device, log, passes=a.passes)
        hs[erased] = h
        torch.save(h.state_dict(), D / f"probe_head_{'erased' if erased else 'intact'}.pt")
    H = data["probe_held"]
    pk_i, pk_e = picks(hs[False], H[False], a.device), picks(hs[True], H[True], a.device)
    it = intact_table(H[False], pk_i, H["meta"])
    res["training_draw"] = intact_table(data["probe_train"][False], picks(hs[False], data["probe_train"][False],
                                                                          a.device), data["probe_train"]["meta"])
    res["erased_table"] = intact_table(H[True], pk_e, H["meta"])
    res["verdict"] = verdict(cm, it, pair_table(H[True], pk_e, H["meta"], H["specs"]),
                             pair_table(H[False], pk_i, H["meta"], H["specs"]))
    res["minutes"] = round((time.time() - t0) / 60, 1)
    (D / "probe.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    log({"PROBE": {k: res["verdict"][k] for k in ("pass", "n_ok", "fails")}, "minutes": res["minutes"]})
    logf.close()
    return 0 if res["verdict"]["pass"] and res["verdict"]["n_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

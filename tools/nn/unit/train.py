"""The unit level's own small training loop (build/diag_g49/plan_unit_net.md step 6): decisions -> simulator steps ->
reward -> PPO (tools/nn/train/ppo.py gae / policy_loss / explained_variance). Not run.py: none of its chain of heads,
general, ability branches or pools is needed here.

    bash tools/nn/dock.sh tools.nn.unit.train --battles 256 --seconds 120 --updates 5 [--device cuda --gpu-duty 0.7]

The smoke battle (the code works; not a drill): Empire spearmen (the network, one unit) against clanrat spearmen under
the script `nearest`, 150 m apart face to face; the spearmen's area a 100 x 100 m square round their start; the order:
any pace, in place. A decision a second (two simulator steps of 0.5 s), the orders at once (no latency here), the leash
every step. The reward: the side's reward.step (gold trade, the end) - no term for the area (a ban, not a penalty).
Stored per decision: only the raw state (view.raw) and the choices; the views are rebuilt in the update (a stored view
would not fit, plan section 3).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tools.nn.model import passport
from tools.nn.sim import battle, scenario
from tools.nn.sim import orders as O
from tools.nn.sim.params import load
from tools.nn.train import opponents, ppo, reward
from tools.nn.unit import decode as Dc
from tools.nn.unit import net as Nt
from tools.nn.unit import view as V

ME, FOE = "wh_main_emp_inf_spearmen_0", "wh2_main_skv_inf_clanrat_spearmen_0"
KIND = "unit_v1"


class Duty:
    """Keep the GPU busy at most `duty` of the time (the user may play on it): after each piece of work, sync and sleep
    (1 - duty) / duty of its length (as --gpu-duty in the main training)."""

    def __init__(self, device, duty=1.0):
        self.cuda = torch.device(device).type == "cuda"
        self.duty = duty
        self.t0 = time.time()
        self.slept = 0.0

    def start(self):
        self.t0 = time.time()

    def end(self):
        if self.cuda:
            torch.cuda.synchronize()
        if self.cuda and self.duty < 1.0:
            s = (time.time() - self.t0) * (1 - self.duty) / self.duty
            time.sleep(s)
            self.slept += s


def smoke_battles(B, P, device, gap_m=150.0, area_m=100.0):
    """B copies of the smoke battle: the state, the network's slots (own [B, N]) and their areas."""
    desc = {"attacker": 1, "sides": {1: {"faction": "wh_main_emp_empire", "units": [{"key": ME, "x": 0.0, "z": -gap_m / 2,
                                                                                       "b": 0.0}]},
                                     2: {"faction": "wh2_main_skv_skaven", "units": [{"key": FOE, "x": 0.0, "z": gap_m / 2,
                                                                                       "b": 180.0}]}}}
    st = scenario.build([desc] * B, P, device=device)
    own = torch.zeros((B, st.N), dtype=torch.bool, device=device)
    own[:, 0] = True
    area = {"x": torch.zeros((B, st.N), device=device), "z": torch.full((B, st.N), -gap_m / 2, device=device),
            "b": torch.zeros((B, st.N), device=device), "front": torch.full((B, st.N), area_m, device=device),
            "depth": torch.full((B, st.N), area_m, device=device), "on": own.clone()}
    return st, own, area


def cards(st, device):
    return torch.from_numpy(np.stack([passport.table(r) for r in st.keys])).to(device)


def history(raws, d, T=V.T):
    """[T, ..., RAW] the T raw snapshots before decision d of a stored sequence raws [D, ...] (newest first; before the
    start: empty, alive 0)."""
    out = torch.zeros((T,) + raws.shape[1:], device=raws.device)
    for k in range(T):
        if d - 1 - k >= 0:
            out[k] = raws[d - 1 - k]
    return out


def views(raws, d_idx, b_idx, card, own, area, cfg, everyone=False):
    """The views of the samples (d_idx, b_idx) rebuilt from the stored raw states (the network's unit of each)."""
    now = raws[d_idx, b_idx]                                              # [M, N, RAW]
    T = V.T
    hist = torch.zeros((T,) + now.shape, device=now.device)
    for k in range(T):
        d = d_idx - 1 - k
        ok = d >= 0
        hist[k] = torch.where(ok[:, None, None], raws[d.clamp(min=0), b_idx], torch.zeros((), device=now.device))
    ar = {k: a[b_idx] for k, a in area.items()}
    return V.build(now, hist, card[b_idx], own[b_idx], area=ar, cell_m=cfg["cell_m"], bounds=cfg["bounds"],
                   everyone=everyone)


def rollout(net, critic, cfg, P, device, duty):
    B = cfg["battles"]
    st, own, area = smoke_battles(B, P, device)
    card = cards(st, device)
    spacing = P.sim["formation"]["spacing_m"]
    D = int(cfg["seconds"])
    steps = round(1.0 / P.dt)
    leash = Dc.Leash(B, st.N, device)
    raws = torch.zeros((D, B, st.N, len(V.RAW)), device=device)
    store = {k: torch.zeros((D, B), dtype=torch.long, device=device) for k in Nt.HEADS}
    lp_old = torch.zeros((D, B), device=device)
    val = torch.zeros((D, B), device=device)
    rew = torch.zeros((D, B), device=device)
    done = torch.zeros((D, B), dtype=torch.bool, device=device)
    valid = torch.zeros((D, B), dtype=torch.bool, device=device)
    stats = {"bans_broken": 0, "leash_s": 0.0, "out_free_s": 0.0, "out_pushed_s": 0.0, "nan": 0}
    order = Nt.order_vec(B, pace=Nt.PACE_ANY, going=False, has_area=True, device=device)
    cm = torch.full((B,), cfg["cell_m"], device=device)
    d_all = torch.arange(D, device=device)
    t_sim = 0.0
    sim_step = battle.stepper(device)          # compiled on CUDA (~10x), plain on the CPU
    for d in range(D):
        duty.start()
        valid[d] = ~st.done
        raws[d] = V.raw(st, spacing, leash=leash.on)
        with torch.no_grad():
            b_idx = torch.arange(B, device=device)
            v = views(raws, d_all[d].expand(B), b_idx, card, own, area, cfg)
            vc = views(raws, d_all[d].expand(B), b_idx, card, own, area, cfg, everyone=True)
            lg = net(v, order, cm)
            act = Nt.sample(lg)
            lp, _ = Nt.log_prob(lg, act)
            val[d] = critic(vc, order, cm)
        stats["nan"] += int(sum(torch.isnan(lg[k]).sum() for k in Nt.HEADS))
        # the bans (the network's masks): never a cell or a target outside them
        stats["bans_broken"] += int((~v["cells_ok"].flatten(1).gather(1, act["where"][:, None])[:, 0]).sum())
        t_ok = torch.cat([v["target_ok"], torch.ones((B, 1), dtype=torch.bool, device=device)], 1)
        stats["bans_broken"] += int((~t_ok.gather(1, act["target"][:, None])[:, 0]).sum())
        for k in Nt.HEADS:
            store[k][d] = act[k]
        lp_old[d] = lp
        intended = Dc.to_orders(act, v, raws[d], cfg["cell_m"], B, st.N)
        r = torch.zeros(B, device=device)
        t0 = time.time()
        for _ in range(steps):
            was_done = st.done.clone()
            o = O.merge(opponents.nearest(st), leash.apply(st, intended, area, own), own)
            before = reward.measure(st)
            sim_step(st, o, P)
            r = r + torch.where(was_done, torch.zeros_like(r), reward.step(before, reward.measure(st), st.done & ~was_done,
                                                                           st.winner)[:, 0])
            u = st.u
            out = Dc.signed_dist(u["x"][:, 0], u["z"][:, 0], {k: a[:, 0] for k, a in area.items()}) < -0.5
            alive = ~was_done & (u["men"][:, 0] > 0)
            stats["leash_s"] += float((leash.on[:, 0] & alive).sum()) * P.dt
            pushed = u["m"][:, 0] | u["r"][:, 0]
            stats["out_free_s"] += float((out & alive & ~pushed).sum()) * P.dt
            stats["out_pushed_s"] += float((out & alive & pushed).sum()) * P.dt
        t_sim += time.time() - t0
        rew[d] = r
        done[d] = st.done & valid[d]
        duty.end()
        if bool(st.done.all()):
            D = d + 1
            break
    with torch.no_grad():
        nxt = torch.zeros((B, st.N, len(V.RAW)), device=device)
        nxt[:] = V.raw(st, spacing, leash=leash.on)
        last = torch.cat([raws[:D], nxt[None]])
        vc = views(last, torch.full((B,), D, device=device), torch.arange(B, device=device), card, own, area, cfg,
                   everyone=True)
        last_value = torch.where(st.done, torch.zeros(B, device=device), critic(vc, order, cm))
    batch = {"raws": raws[:D], "act": {k: v_[:D] for k, v_ in store.items()}, "lp": lp_old[:D], "value": val[:D],
             "reward": rew[:D], "done": done[:D], "valid": valid[:D], "last_value": last_value, "card": card, "own": own,
             "area": area, "order": order}
    wins = float((st.winner == 1).float().mean())
    stats.update({"decisions": D, "battles": B, "win1": wins, "ended": float(st.done.float().mean()),
                  "reward_mean": float(rew[:D][valid[:D]].sum() / B), "sim_s": round(t_sim, 2)})
    return batch, stats


def update(net, critic, opt, batch, cfg, device, duty):
    adv, ret = ppo.gae(batch["reward"], batch["value"], batch["done"], batch["last_value"], cfg["gamma"], cfg["lam"])
    d_idx, b_idx = batch["valid"].nonzero(as_tuple=True)
    M = d_idx.shape[0]
    a_all = adv[d_idx, b_idx]
    a_all = (a_all - a_all.mean()) / (a_all.std() + 1e-8)
    r_all = ret[d_idx, b_idx]
    out = {"pi": [], "v": [], "ent": {k: [] for k in Nt.HEADS}, "clip": [], "value_pred": [], "ret": []}
    for _ in range(cfg["epochs"]):
        perm = torch.randperm(M, device=device)
        for i in range(0, M, cfg["minibatch"]):
            duty.start()
            j = perm[i:i + cfg["minibatch"]]
            di, bi = d_idx[j], b_idx[j]
            v = views(batch["raws"], di, bi, batch["card"], batch["own"], batch["area"], cfg)
            vc = views(batch["raws"], di, bi, batch["card"], batch["own"], batch["area"], cfg, everyone=True)
            order = batch["order"][bi]
            cm = torch.full((len(j),), cfg["cell_m"], device=device)
            lg = net(v, order, cm)
            act = {k: batch["act"][k][di, bi] for k in Nt.HEADS}
            lp, ent = Nt.log_prob(lg, act)
            pl, clip = ppo.policy_loss(lp[:, None], batch["lp"][di, bi][:, None], a_all[j], torch.ones_like(lp[:, None]),
                                       cfg["clip"])
            value = critic(vc, order, cm)
            vl = F.mse_loss(value, r_all[j])
            loss = pl + cfg["vf"] * vl - cfg["entropy"] * ent.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(net.parameters()) + list(critic.parameters()), 1.0)
            opt.step()
            out["pi"].append(float(pl.detach()))
            out["v"].append(float(vl.detach()))
            out["clip"].append(float(clip))
            for k in Nt.HEADS:
                d_ = torch.distributions.Categorical(logits=lg[k])
                p = d_.probs
                out["ent"][k].append(float(-(torch.where(p > 0, p * d_.logits, torch.zeros_like(p))).sum(1).mean()))
            out["value_pred"].append(value.detach())
            out["ret"].append(r_all[j])
            duty.end()
    ev = float(ppo.explained_variance(torch.cat(out["value_pred"]), torch.cat(out["ret"])))
    return {"policy_loss": float(np.mean(out["pi"])), "value_loss": float(np.mean(out["v"])),
            "clip": float(np.mean(out["clip"])), "explained_var": ev, "samples": M,
            "entropy": {k: round(float(np.mean(v)), 4) for k, v in out["ent"].items()}}


def save(path, net, critic, opt, cfg):
    torch.save({"kind": KIND, "net": net.state_dict(), "critic": critic.state_dict(), "opt": opt.state_dict(),
                "cfg": cfg}, path)


def restore(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck.get("kind") != KIND:
        raise ValueError(f"not a {KIND} checkpoint: {path}")
    net, critic = Nt.UnitNet().to(device), Nt.Critic().to(device)
    net.load_state_dict(ck["net"])
    critic.load_state_dict(ck["critic"])
    return net, critic, ck


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--battles", type=int, default=256)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--updates", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--gpu-duty", type=float, default=1.0)
    ap.add_argument("--out", default="build/diag_g49/smoke")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)
    P = load()
    cfg = {"battles": a.battles, "seconds": a.seconds, "epochs": a.epochs, "minibatch": a.minibatch, "cell_m": 8.0,
           "bounds": None, "gamma": 0.9994, "lam": 0.9025, "clip": 0.2, "vf": 0.5, "entropy": 0.01}
    dev = a.device
    net, critic = Nt.UnitNet().to(dev), Nt.Critic().to(dev)
    opt = torch.optim.Adam(list(net.parameters()) + list(critic.parameters()), lr=a.lr)
    duty = Duty(dev, a.gpu_duty)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = []
    for k in range(a.updates):
        t0 = time.time()
        batch, rs = rollout(net, critic, cfg, P, dev, duty)
        t1 = time.time()
        us = update(net, critic, opt, batch, cfg, dev, duty)
        t2 = time.time()
        row = {"update": k, **rs, **us, "rollout_s": round(t1 - t0, 1), "update_s": round(t2 - t1, 1),
               "battle_s_per_s": round(rs["decisions"] * rs["battles"] / max(t1 - t0, 1e-9), 1),
               "slept_s": round(duty.slept, 1)}
        log.append(row)
        print(json.dumps(row), flush=True)
    path = out / "latest.pt"
    save(path, net, critic, opt, cfg)
    net2, critic2, _ = restore(path, dev)
    # the checkpoint reads back: the same output bit for bit
    st, own, area = smoke_battles(4, P, dev)
    raws = V.raw(st, P.sim["formation"]["spacing_m"])[None]
    v = views(raws, torch.zeros(4, dtype=torch.long, device=dev), torch.arange(4, device=dev), cards(st, dev), own, area,
              cfg)
    o, cm = Nt.order_vec(4, has_area=True, device=dev), torch.full((4,), 8.0, device=dev)
    with torch.no_grad():
        same = all(torch.equal(net.eval()(v, o, cm)[k], net2.eval()(v, o, cm)[k]) for k in Nt.HEADS)
    print(json.dumps({"checkpoint_same": same}), flush=True)
    (out / "log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    return log, same


if __name__ == "__main__":
    main()

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
from tools.nn.train import cadence, opponents, ppo, reward
from tools.nn.unit import decode as Dc
from tools.nn.unit import net as Nt
from tools.nn.unit import view as V

ME, FOE = "wh_main_emp_inf_spearmen_0", "wh2_main_skv_inf_clanrat_spearmen_0"
KIND = "unit_v1"
AREA_REWARD = 0.001     # per second the area is ours / the enemy's (U2): 150 s ~ 0.15 of the budget, the trade ~0.4


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


def views(raws, d_idx, b_idx, i_idx, card, area, cfg, everyone=False):
    """The views of the samples (d_idx, b_idx, i_idx: decision, battle, the network's unit's slot) rebuilt from the stored
    raw states - one viewer a sample; v["bi"] is set to the battles (to_orders writes the orders there)."""
    now = raws[d_idx, b_idx]                                              # [M, N, RAW]
    T = V.T
    hist = torch.zeros((T,) + now.shape, device=now.device)
    for k in range(T):
        d = d_idx - 1 - k
        ok = d >= 0
        hist[k] = torch.where(ok[:, None, None], raws[d.clamp(min=0), b_idx], torch.zeros((), device=now.device))
    ar = {k: a[b_idx] for k, a in area.items()}
    who = torch.zeros(now.shape[:2], dtype=torch.bool, device=now.device)
    who[torch.arange(now.shape[0], device=now.device), i_idx] = True
    v = V.build(now, hist, card[b_idx], who, area=ar, cell_m=cfg["cell_m"], bounds=cfg["bounds"], everyone=everyone)
    v["bi"] = b_idx
    return v


def viewer_ok(st, bi, ii):
    """[R] the network's units still deciding: alive, not gone, not routing (a router's orders are ignored)."""
    u = st.u
    return (u["men"][bi, ii] > 0) & ~u["gone"][bi, ii] & ~u["r"][bi, ii]


def smoke_setup(cfg, P, device, gen=None):
    st, own, area = smoke_battles(cfg["battles"], P, device)
    return {"st": st, "own": own, "area": area, "ender": None}


def hold_setup(cfg, P, device, gen=None, sp=None):
    """Drill U1 (drill_hold.py): cfg battles random skirmishes (or the given specs sp)."""
    from tools.nn.unit import drill_hold as DH
    sp = sp if sp is not None else DH.specs(cfg["battles"], gen or torch.Generator())
    st, own, area = DH.build(sp, P, device)
    return {"st": st, "own": own, "area": area, "ender": DH.Ender(st.B, device), "specs": sp}


def go_setup(cfg, P, device, gen=None, sp=None):
    """Drill U2 (drill_go.py): reach the area and hold it; the enemy goes to the area too."""
    from tools.nn.unit import drill_go as DG
    sp = sp if sp is not None else DG.specs(cfg["battles"], gen or torch.Generator())
    st, own, area, pace = DG.build(sp, P, device)
    return {"st": st, "own": own, "area": area, "ender": DG.Ender(sp, device), "specs": sp, "pace": pace,
            "script": lambda s_: DG.to_area(s_, area)}


def orders_now(raw_d, area, pace, arrived, bi, ii):
    """[R] orders (net.order_of) of the network's units (battle bi, slot ii) from the raw state raw_d [B, N, RAW]."""
    return Nt.order_of(raw_d[bi, ii], {k: v[bi, ii] for k, v in area.items() if k != "on"}, pace[bi], arrived[bi])


def multi_setup(cfg, P, device, gen=None, sp=None):
    """Drill U3 (drill_multi.py): hold the area against two enemies; the branches' kit of four variants."""
    from tools.nn.unit import branch as Br
    from tools.nn.unit import drill_multi as DM
    sp = sp if sp is not None else DM.specs(cfg["battles"], gen or torch.Generator())
    st, own, area = DM.build(sp, P, device)
    return {"st": st, "own": own, "area": area, "ender": DM.Ender(st.B, device), "specs": sp, "kit": Br.KIT_MULTI}


def pair_setup(cfg, P, device, gen=None, sp=None):
    """Drill U4 (drill_pair.py): two of our units side by side, each with its own area, both commanded by the network
    (the side's reward shared by both)."""
    from tools.nn.unit import drill_pair as DP
    sp = sp if sp is not None else DP.specs(cfg["battles"], gen or torch.Generator())
    st, own, area = DP.build(sp, P, device)
    return {"st": st, "own": own, "area": area, "ender": DP.Ender(st.B, device), "specs": sp}


SETUPS = {"smoke": smoke_setup, "hold": hold_setup, "go": go_setup, "multi": multi_setup, "hold2": pair_setup}
MIX = ("hold", "go", "multi", "hold2")          # --drill mix: one drill a batch, in turn


def rollout(net, critic, cfg, P, device, duty, setup, watch=None, greedy=False, branch=None):
    """One batch of battles played by the network (its unit) against the script `nearest`; the orders land after the
    game's latency (tools/nn/train/cadence.py: 0.36 s on average - one 0.5 s step late with probability 0.72), the
    leash every step. critic None: no values (the exam). watch(st, before_hp, step_was_done): the exam's metrics.
    branch {share, weight, lead_s, horizon_s} (drill U1, branch.py): the share of the battles copied at the approach and
    played on as 'stand' and 'attack'; the network's behaviour at the contact matching the better one earns weight x
    their trade difference at the contact's decision. Per battle results in stats["_branch"]."""
    st, own, area, ender = setup["st"], setup["own"], setup["area"], setup["ender"]
    B = st.B
    bi, ii = own.nonzero(as_tuple=True)              # the network's units: R viewers (one a battle in U1-U3)
    R = bi.shape[0]
    card = cards(st, device)
    spacing = P.sim["formation"]["spacing_m"]
    D = int(cfg["seconds"])
    steps = round(1.0 / P.dt)
    cad = cadence.Cadence()
    leash = Dc.Leash(B, st.N, device)
    raws = torch.zeros((D, B, st.N, len(V.RAW)), device=device)
    store = {k: torch.zeros((D, R), dtype=torch.long, device=device) for k in Nt.HEADS}
    lp_old = torch.zeros((D, R), device=device)
    val = torch.zeros((D, R), device=device)
    rew = torch.zeros((D, R), device=device)
    done = torch.zeros((D, R), dtype=torch.bool, device=device)
    valid = torch.zeros((D, R), dtype=torch.bool, device=device)
    stats = {"bans_broken": 0, "leash_s": 0.0, "out_free_s": 0.0, "out_pushed_s": 0.0, "nan": 0,
             "entropy": {k: 0.0 for k in Nt.HEADS}, "n_dec": 0}
    pace = setup.get("pace", torch.full((B,), Nt.PACE_ANY, dtype=torch.long, device=device))
    script = setup.get("script", opponents.nearest)
    arrived_d = torch.ones((D, B), dtype=torch.bool, device=device)
    cm = torch.full((R,), cfg["cell_m"], device=device)
    d_all = torch.arange(D, device=device)
    t_sim = 0.0
    sim_step = battle.stepper(device)          # compiled on CUDA (~10x), plain on the CPU
    in_force = O.hold(B, st.N, device)        # the orders in force until the new ones land
    if branch is not None:
        from tools.nn.unit import branch as Br
        assert R == B and bool((ii == 0).all()), "branches: one network unit (slot 0) a battle"
        H = st.N // 2
        will = torch.rand(B, device=device) < branch["share"]
        branched = torch.zeros(B, dtype=torch.bool, device=device)
        snaps = []
        contact_d = torch.full((B,), -1, dtype=torch.long, device=device)
        beh_d = torch.full((B,), -1, dtype=torch.long, device=device)     # the decision that settled the behaviour
        beh = torch.full((B,), Br.OTHER, dtype=torch.long, device=device)
        brace = Br.bracing_deg(P)
        kit = setup.get("kit", Br.KIT1)
    for d in range(D):
        duty.start()
        valid[d] = ~st.done[bi] & viewer_ok(st, bi, ii)
        raws[d] = V.raw(st, spacing, leash=leash.on)
        with torch.no_grad():
            dd = d_all[d].expand(R)
            v = views(raws, dd, bi, ii, card, area, cfg)
            arrived = getattr(ender, "arrived", None)
            arrived = torch.ones(B, dtype=torch.bool, device=device) if arrived is None else arrived.clone()
            arrived_d[d] = arrived
            order = orders_now(raws[d], area, pace, arrived, bi, ii)
            lg = net(v, order, cm)
            act = Nt.sample(lg, greedy=greedy)
            lp, ent = Nt.log_prob(lg, act)
            if critic is not None:
                vc = views(raws, dd, bi, ii, card, area, cfg, everyone=True)
                val[d] = critic(vc, order, cm)
        live = valid[d]
        stats["n_dec"] += int(live.sum())
        for k in Nt.HEADS:
            dk = torch.distributions.Categorical(logits=lg[k])
            p = dk.probs
            e = -(torch.where(p > 0, p * dk.logits, torch.zeros_like(p))).sum(1)
            stats["entropy"][k] += float(e[live].sum())
        stats["nan"] += int(sum(torch.isnan(lg[k]).sum() for k in Nt.HEADS))
        # the bans (the network's masks): never a cell or a target outside them
        stats["bans_broken"] += int((~v["cells_ok"].flatten(1).gather(1, act["where"][:, None])[:, 0] & live).sum())
        t_ok = torch.cat([v["target_ok"], torch.ones((R, 1), dtype=torch.bool, device=device)], 1)
        stats["bans_broken"] += int((~t_ok.gather(1, act["target"][:, None])[:, 0] & live).sum())
        for k in Nt.HEADS:
            store[k][d] = act[k]
        lp_old[d] = lp
        intended = Dc.to_orders(act, v, raws[d], cfg["cell_m"], B, st.N)
        lands = cad.delays(P.dt, B, device)                   # [B] the step this decision's orders land at
        r = torch.zeros(B, device=device)
        t0 = time.time()
        for j in range(steps):
            was_done = st.done.clone()
            now_in = lands <= j
            mine = O.Orders(*(torch.where(now_in[:, None], getattr(intended, f), getattr(in_force, f)) for f in O.FIELDS))
            o = O.merge(script(st), leash.apply(st, mine, area, own), own)
            before = reward.measure(st)
            hp0 = st.u["hp_abs"][:, 0].clone()
            sim_step(st, o, P)
            if ender is not None:
                finished = ender.after_step(st, spacing, area)
                st.winner = torch.where(st.done, st.winner, torch.zeros_like(st.winner))
            else:
                finished = st.done & ~was_done
            r = r + torch.where(was_done, torch.zeros_like(r),
                                reward.step(before, reward.measure(st), finished, st.winner)[:, 0])
            if cfg.get("area_reward"):
                # the area's signal (drill_go.whose_area): + per second ours; - per second the enemy's once we have
                # arrived (before, the march is the order's - its pace, the way - not the unit's choice: 0); 0 empty
                from tools.nn.unit import drill_go as DG
                wa = DG.whose_area(st, area)
                arr = getattr(ender, "arrived", None)
                if arr is not None:
                    wa = torch.where((wa == DG.THEIRS) & ~arr, torch.zeros_like(wa), wa)
                r = r + torch.where(was_done, torch.zeros_like(r), cfg["area_reward"] * P.dt * wa.float())
            if branch is not None:
                m = will & ~branched & ~was_done & Br.moment(st, spacing, branch["lead_s"])
                if bool(m.any()):
                    idx = m.nonzero()[:, 0]
                    snaps.append((idx, Br.take(st, idx)))
                    branched = branched | m
                first = (contact_d < 0) & Br.contact(st) & ~was_done
                if bool(first.any()):
                    contact_d = torch.where(first, d, contact_d)
                    beh_d = torch.where(first, d, beh_d)
                    beh = torch.where(first, kit.behaviour(st, brace), beh)
                if hasattr(kit, "track") and bool(branched.any()):
                    nb = kit.track(st, beh, spacing)
                    beh_d = torch.where((nb != beh) & branched & ~was_done, d, beh_d)
                    beh = torch.where(branched & ~was_done, nb, beh)
            if watch is not None:
                watch(st, hp0, was_done, area, d * steps + j + 1)
            u = st.u
            out = Dc.signed_dist(u["x"], u["z"], {k: a for k, a in area.items() if k != "on"}) < -0.5
            alive = own & ~was_done[:, None] & (u["men"] > 0)
            stats["leash_s"] += float((leash.on & alive).sum()) * P.dt
            pushed = u["m"] | u["r"]
            stats["out_free_s"] += float((out & alive & ~pushed).sum()) * P.dt
            stats["out_pushed_s"] += float((out & alive & pushed).sum()) * P.dt
        in_force = intended
        t_sim += time.time() - t0
        rew[d] = r[bi]                      # the side's reward, shared by its network units
        # the drill's time limit is a cut, not an end: no 'done' there, the critic's value of the last state goes on;
        # a unit that routed or died (the battle going on) ends its own trajectory
        cut = (ender.why == 3) if ender is not None else torch.zeros_like(st.done)
        done[d] = valid[d] & ~(~st.done[bi] & viewer_ok(st, bi, ii)) & ~(st.done[bi] & cut[bi])
        duty.end()
        if bool(st.done.all()):
            D = d + 1
            break
    if branch is not None:
        tb = time.time()
        Vn = len(kit.variants)
        info = {"branched": branched, "contact_d": contact_d, "beh_d": beh_d, "behaviour": beh, "names": kit.names(),
                "trades": torch.full((B, Vn), float("nan"), device=device),
                "best": torch.full((B,), -1, dtype=torch.long, device=device)}
        if snaps:
            idx = torch.cat([i for i, _ in snaps])
            snap = Br.cat([x for _, x in snaps])
            rep = {}
            # (pad the branches to a fixed size - twice the expected share - so the compiled step keeps its shape)
            rows = max(1, int(round(B * min(1.0, 2 * branch["share"]))))
            info["trades"][idx] = kit.play(snap, {k: v[idx] for k, v in area.items()}, P, sim_step,
                                           branch["horizon_s"], rows=rows, report=rep)
            ok = branched & (contact_d >= 0) & (contact_d < D)
            bon, best = Br.bonus_of(beh, torch.nan_to_num(info["trades"]), branch["weight"])
            bon = torch.where(ok, bon, torch.zeros_like(bon))
            rew[beh_d.clamp(0, D - 1), torch.arange(B, device=device)] += bon
            info["best"] = torch.where(branched, best, torch.full_like(best, -1))
            _, margin = Br.best_of(torch.nan_to_num(info["trades"]))
            n = max(int(ok.sum()), 1)
            stand_v = [v for v, (k_, _) in enumerate(kit.variants) if k_ == "stand"]
            braced = rep["braced"][:, stand_v][rep["contact"][:, stand_v]]
            stats["branch"] = {"share": round(float(branched.float().mean()), 3),
                               "best": {nm: round(float((best[branched] == v).float().mean()), 3)
                                        for v, nm in enumerate(kit.names())},
                               "margin": round(float(margin[branched].mean()), 4),
                               "beh": {nm: round(float((beh[ok] == v).float().sum()) / n, 3)
                                       for v, nm in enumerate(kit.names())},
                               "chose_best": round(float((beh[ok] == best[ok]).float().sum()) / n, 3),
                               "bonus_mean": round(float(bon.sum()) / B, 4), "weight": branch["weight"],
                               "stand_branch_braced": round(float(braced.float().mean()), 3) if braced.numel() else None,
                               "seconds": round(time.time() - tb, 1)}
        if Vn == 2:                         # (U1's names in the exam)
            info["t_stand"], info["t_attack"] = info["trades"][:, Br.STAND], info["trades"][:, Br.ATTACK]
        stats["_branch"] = info
    last_value = torch.zeros(R, device=device)
    if critic is not None:
        with torch.no_grad():
            nxt = V.raw(st, spacing, leash=leash.on)
            last = torch.cat([raws[:D], nxt[None]])
            vc = views(last, torch.full((R,), D, device=device), bi, ii, card, area, cfg, everyone=True)
            cut = (ender.why == 3) if ender is not None else torch.zeros_like(st.done)
            arr = getattr(ender, "arrived", None)
            order = orders_now(nxt, area, pace, torch.ones(B, dtype=torch.bool, device=device) if arr is None else arr,
                               bi, ii)
            going = (~st.done[bi] & viewer_ok(st, bi, ii)) | (st.done[bi] & cut[bi])
            last_value = torch.where(going, critic(vc, order, cm), torch.zeros(R, device=device))
    batch = {"raws": raws[:D], "act": {k: v_[:D] for k, v_ in store.items()}, "lp": lp_old[:D], "value": val[:D],
             "reward": rew[:D], "done": done[:D], "valid": valid[:D], "last_value": last_value, "card": card, "own": own,
             "area": area, "pace": pace, "arrived": arrived_d[:D], "bi": bi, "ii": ii}
    n = max(stats.pop("n_dec"), 1)
    stats["entropy"] = {k: round(v_ / n, 4) for k, v_ in stats["entropy"].items()}
    stats.update({"decisions": D, "battles": B, "win1": float((st.winner == 1).float().mean()),
                  "ended": float(st.done.float().mean()), "reward_mean": float(rew[:D][valid[:D]].sum() / R),
                  "sim_s": round(t_sim, 2)})
    if ender is not None:
        stats["end"] = {w: float((ender.why == k).float().mean()) for k, w in ((1, "we_routed"), (2, "enemy_left"),
                                                                              (3, "limit"), (0, "open"))}
    return batch, stats


def update(net, critic, opt, batch, cfg, device, duty):
    adv, ret = ppo.gae(batch["reward"], batch["value"], batch["done"], batch["last_value"], cfg["gamma"], cfg["lam"])
    d_idx, r_idx = batch["valid"].nonzero(as_tuple=True)
    M = d_idx.shape[0]
    a_all = adv[d_idx, r_idx]
    a_all = (a_all - a_all.mean()) / (a_all.std() + 1e-8)
    r_all = ret[d_idx, r_idx]
    out = {"pi": [], "v": [], "ent": {k: [] for k in Nt.HEADS}, "clip": [], "value_pred": [], "ret": []}
    for _ in range(cfg["epochs"]):
        perm = torch.randperm(M, device=device)
        for i in range(0, M, cfg["minibatch"]):
            duty.start()
            j = perm[i:i + cfg["minibatch"]]
            di, ri = d_idx[j], r_idx[j]
            bi, si = batch["bi"][ri], batch["ii"][ri]
            v = views(batch["raws"], di, bi, si, batch["card"], batch["area"], cfg)
            vc = views(batch["raws"], di, bi, si, batch["card"], batch["area"], cfg, everyone=True)
            order = Nt.order_of(batch["raws"][di, bi, si], {k: v[bi, si] for k, v in batch["area"].items() if k != "on"},
                                batch["pace"][bi], batch["arrived"][di, bi])
            cm = torch.full((len(j),), cfg["cell_m"], device=device)
            lg = net(v, order, cm)
            act = {k: batch["act"][k][di, ri] for k in Nt.HEADS}
            lp, ent = Nt.log_prob(lg, act)
            pl, clip = ppo.policy_loss(lp[:, None], batch["lp"][di, ri][:, None], a_all[j], torch.ones_like(lp[:, None]),
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


def save(path, net, critic, opt, cfg, updates=0):
    torch.save({"kind": KIND, "net": net.state_dict(), "critic": critic.state_dict(), "opt": opt.state_dict(),
                "cfg": cfg, "updates": updates}, path)


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
    ap.add_argument("--drill", default="smoke", choices=sorted(SETUPS) + ["mix"],
                    help="mix: U1 / U2 / U3 / U4 (hold / go / multi / hold2) in turn, one drill a batch - equal shares of the "
                         "skirmishes; a batch of mixed drills would wait for its longest skirmish (go's up to 700 s)")
    ap.add_argument("--battles", type=int, default=256)
    ap.add_argument("--seconds", type=float, default=None, help="decisions a batch at most (smoke 120, hold 180)")
    ap.add_argument("--updates", type=int, default=5, help="at most this many updates")
    ap.add_argument("--minutes", type=float, default=0.0, help="> 0: no new update after this many minutes")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--init", default=None, help="go on from this checkpoint (latest.pt of a run): weights, Adam, counter")
    ap.add_argument("--exam", action="store_true", help="the drill's exam at the end (exam_hold.py / exam_go.py) -> "
                    "exam.json")
    ap.add_argument("--branch-share", type=float, default=0.0, help="drill hold: the share of battles branched at the "
                    "approach (stand / attack, branch.py); 0 off")
    ap.add_argument("--branch-bonus", type=float, default=1.0, help="the bonus's weight x the branches' trade difference")
    ap.add_argument("--branch-decay", type=int, default=20, help="the weight falls linearly to 0 over this many updates of this run")
    ap.add_argument("--branch-lead-s", type=float, default=4.0)
    ap.add_argument("--branch-horizon-s", type=float, default=60.0)
    ap.add_argument("--area-reward", type=float, default=None, help="per second the area is ours +W, the enemy's -W (after our arrival) "
                    "(drill_go.whose_area); default 0.001 for --drill go (~0.15 of the budget over 150 s), else 0")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--gpu-duty", type=float, default=1.0)
    ap.add_argument("--out", default="build/diag_g49/smoke")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)
    P = load()
    drills = list(MIX) if a.drill == "mix" else [a.drill]

    def cfg_of(drill):
        """The batch's settings for one drill (mix: the area's signal off unless --area-reward says)."""
        area = a.area_reward if a.area_reward is not None else (AREA_REWARD if a.drill == "go" else 0.0)
        return {"battles": a.battles,
                "seconds": a.seconds or {"hold": 180.0, "go": 700.0, "multi": 180.0, "hold2": 180.0}.get(drill, 120.0),
                "epochs": a.epochs, "minibatch": a.minibatch, "cell_m": 8.0, "bounds": None, "gamma": 0.9994,
                "lam": 0.9025, "clip": 0.2, "vf": 0.5, "entropy": 0.01, "drill": drill, "area_reward": area}
    cfg = cfg_of(drills[0])
    dev = a.device
    net, critic = Nt.UnitNet().to(dev), Nt.Critic().to(dev)
    opt = torch.optim.Adam(list(net.parameters()) + list(critic.parameters()), lr=a.lr)
    done_updates = 0
    if a.init:
        net, critic, ck = restore(a.init, dev)
        opt = torch.optim.Adam(list(net.parameters()) + list(critic.parameters()), lr=a.lr)
        opt.load_state_dict(ck["opt"])
        done_updates = int(ck.get("updates", 0))
    duty = Duty(dev, a.gpu_duty)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(a.seed + 1000 * done_updates)
    log = []
    start = time.time()
    for k in range(a.updates):
        if a.minutes > 0 and time.time() - start > a.minutes * 60:
            break
        t0 = time.time()
        drill = drills[k % len(drills)]
        cfg = cfg_of(drill)
        br = None
        if a.branch_share > 0 and drill in ("hold", "multi"):
            from tools.nn.unit import branch as Br
            # (the decay counts this run's updates: a run going on from a checkpoint starts the bonus afresh)
            br = {"share": a.branch_share, "weight": Br.weight_at(a.branch_bonus, k, a.branch_decay),
                  "lead_s": a.branch_lead_s, "horizon_s": a.branch_horizon_s}
        batch, rs = rollout(net, critic, cfg, P, dev, duty, SETUPS[drill](cfg, P, dev, gen), branch=br)
        rs.pop("_branch", None)
        t1 = time.time()
        us = update(net, critic, opt, batch, cfg, dev, duty)
        t2 = time.time()
        done_updates += 1
        row = {"update": done_updates, "drill": drill, **rs, **us, "rollout_s": round(t1 - t0, 1), "update_s": round(t2 - t1, 1),
               "battle_s_per_s": round(float(batch["valid"].sum()) / max(t1 - t0, 1e-9), 1),
               "slept_s": round(duty.slept, 1)}
        log.append(row)
        print(json.dumps(row), flush=True)
        save(out / "latest.pt", net, critic, opt, cfg, done_updates)
    path = out / "latest.pt"
    save(path, net, critic, opt, cfg, done_updates)
    net2, critic2, _ = restore(path, dev)
    # the checkpoint reads back: the same output bit for bit
    st, own, area = smoke_battles(4, P, dev)
    raws = V.raw(st, P.sim["formation"]["spacing_m"])[None]
    v = views(raws, torch.zeros(4, dtype=torch.long, device=dev), torch.arange(4, device=dev),
              torch.zeros(4, dtype=torch.long, device=dev), cards(st, dev), area, cfg)
    o, cm = Nt.order_vec(4, has_area=True, device=dev), torch.full((4,), 8.0, device=dev)
    with torch.no_grad():
        same = all(torch.equal(net.eval()(v, o, cm)[k], net2.eval()(v, o, cm)[k]) for k in Nt.HEADS)
    net.train()
    print(json.dumps({"checkpoint_same": same, "updates": done_updates,
                      "minutes": round((time.time() - start) / 60, 1)}), flush=True)
    (out / "log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    if a.exam:
        from tools.nn.unit import exam_go, exam_hold, exam_multi, exam_pair
        exams = {"hold": exam_hold, "go": exam_go, "multi": exam_multi, "hold2": exam_pair}
        if a.drill == "mix":                       # all three, one after another
            for d_ in MIX:
                exams[d_].run(net, P, dev, duty, out / f"exam_{d_}.json")
        else:
            exams.get(a.drill, exam_hold).run(net, P, dev, duty, out / "exam.json")
    return log, same


if __name__ == "__main__":
    main()

"""The operative's battle in the simulator: the operative every decide_s (15 s), the tacticians every 5 s (the formula
by default; a TacticNet for the end-to-end run), the frozen unit network every second (lower.py).

    play(setup, unit_net, net=None, critic=None, mode='formula', ...)  one batch to its end. Who decides: an
                      OperativeNet (sampling or its best) or the formula / a plain operator (teacher.MODES). Stored per
                      decision: the raw state, the groups before it, the choices (rebuilt in the update: only raw numbers
                      are kept, as at the tactician). Every parcel sent goes through tactic/parcel.check (the count of
                      errors: not 0 is a bug of the code)

The reward (config reward, plan.md "Награда"): at a clump's contact + if its infantry stands between (the split: half
a clump), - if the enemy went straight into its shooters; gather: + the share gathered at the first contact; at the
end the trade in gold; go 'destroy': - passive without a contact by the end; go 'hold': - advance once, when my centre
went advance_m towards the enemy; - reform for a re-formed group the network chose, - move for a zone moved move_m or
more by its choice.
"""
import time

import torch

from tools.nn.operative import cfg
from tools.nn.operative import cases as CS
from tools.nn.operative import frame as FR
from tools.nn.operative import groups as GR
from tools.nn.operative import inputs as IN
from tools.nn.operative import lower as LW
from tools.nn.operative import net as NN
from tools.nn.operative import teacher as TC
from tools.nn.operative import watch as WT
from tools.nn.sim import battle
from tools.nn.tactic import cfg as tcfg
from tools.nn.tactic import parcel as PC
from tools.nn.tactic import raw as TR
from tools.nn.train import reward


def rows_of(N):
    return min(int(cfg()["rows"]), N)


def field_dict(axes, bi):
    f = axes.field
    return {"cx": f["x"], "cz": f["z"], "b": f["b"], "front": f["front"], "depth": f["depth"]}


@torch.no_grad()
def play(setup, unit_net, net=None, critic=None, mode="formula", greedy=False, keep=True, tactic_net=None,
         duty=None, check=True, unit_greedy=False):
    st, P, dev = setup.st, setup.P, setup.device
    B, N = st.B, st.N
    c = cfg()
    rw = c["reward"]
    G = int(c["groups"])
    K = rows_of(N)
    every = int(c["decide_s"])
    t_every = int(tcfg()["decide_s"])
    limit = int(float(setup.limit.max()))
    Dmax = limit // every + 1
    steps = round(1.0 / P.dt)
    seer = TR.Seer(P)
    groups = GR.Groups.empty(B, N, G, dev)
    tac = LW.Tacticians(B, N, G, dev, net=tactic_net)
    units = LW.Units(setup, unit_net, unit_greedy)
    rd = WT.Readings(setup, P.sim["formation"]["spacing_m"])
    ender = WT.Ender(setup)
    script = CS.Script(setup.stands)
    sim_step = battle.stepper(dev)
    m0 = reward.measure(st)
    gather_case = setup.case == CS.CASES.index("gather")
    go = setup.case == CS.CASES.index("go")
    go_hold, go_destroy = go & ~setup.destroy, go & setup.destroy
    axes = None
    store = {"raw": [], "groups": [], "act": []}
    lp_old = torch.zeros((Dmax, B), device=dev)
    val = torch.zeros((Dmax, B), device=dev)
    rew = torch.zeros((Dmax, B), device=dev)
    done = torch.zeros((Dmax, B), dtype=torch.bool, device=dev)
    valid = torch.zeros((Dmax, B), dtype=torch.bool, device=dev)
    stats = {"decisions": 0, "reforms": 0.0, "moves": 0.0, "keeps": 0.0, "parcels": 0, "parcel_errors": 0,
             "errors": [], "groups_on": 0.0, "ent": {h: 0.0 for h in NN.HEADS}}
    d = -1
    r_acc = torch.zeros(B, device=dev)
    t_op = t_low = t_sim = 0.0
    sec = 0
    for sec in range(limit + 1):
        if bool(st.done.all()):
            break
        if duty is not None:
            duty.start()
        raw_t = seer.raw(st)
        if sec % every == 0:
            t0 = time.time()
            if d >= 0:
                rew[d] = r_acc
                done[d] = valid[d] & st.done
                r_acc = torch.zeros(B, device=dev)
            d += 1
            if axes is None:
                axes = FR.Axes.of(raw_t, setup.side, st.bounds)
            groups.drop_dead(raw_t)
            inp = IN.build(raw_t, setup.card, setup.side, axes, groups, setup.command, bounds=st.bounds, K=K)
            ctx = NN.ctx_of(inp, raw_t, axes, groups)
            if net is not None:
                out, _ = net(inp, ctx, greedy=greedy)
                lp_old[d] = out["lp"]
                if critic is not None:
                    val[d] = critic(inp)
                for h in stats["ent"]:
                    stats["ent"][h] += float(out["ent"][h][~st.done].sum())
            else:
                out = TC.play_formula(inp, ctx, setup.destroy, mode)
            valid[d] = ~st.done & ctx.own.any(1)
            if keep:
                store["raw"].append(raw_t)
                store["groups"].append(groups.clone())
                store["act"].append(out["act"])
            new = out["new"]
            live = valid[d]
            if net is not None:
                paid = out["reformed"] & live[:, None]
                far = (new.cx - groups.cx) ** 2 + (new.cz - groups.cz) ** 2 >= float(c["teacher"]["move_m"]) ** 2
                mv = out["task"] & out["moved"] & far & live[:, None]
                r_acc = r_acc - float(rw["reform"]) * paid.sum(1).float() - float(rw["move"]) * mv.sum(1).float()
            stats["decisions"] += int(live.sum())
            stats["reforms"] += float((out["reformed"] & live[:, None]).sum())
            stats["moves"] += float((out["task"] & out["moved"] & live[:, None]).sum())
            stats["keeps"] += float((groups.on & (out["act"]["keep"] == GR.KEEP) & live[:, None]).sum())
            stats["groups_on"] += float((new.on & live[:, None]).sum())
            new.age = torch.where(new.on, new.age + every, new.age)
            parcels, idx = tac.apply(new, out["task"] & live[:, None], out["moved"], raw_t)
            if check and idx.numel():
                errs = PC.check(raw_t, parcels, field_dict(axes, None))
                stats["parcels"] += len(errs)
                bad = [e for e in errs if e]
                stats["parcel_errors"] += len(bad)
                if bad and len(stats["errors"]) < 5:
                    stats["errors"].append(bad[0])
            groups = new
            t_op += time.time() - t0
        t0 = time.time()
        if sec % t_every == 0:
            tac.tick(raw_t, setup.card, st.bounds)
            units.write(tac)
        lands = units.second(st)
        t_low += time.time() - t0
        t0 = time.time()
        for j in range(steps):
            was = st.done.clone()
            o = units.orders(st, j, lands, script)
            sim_step(st, o, P)
            st.winner = torch.where(st.done, st.winner, torch.zeros_like(st.winner))
            bt, ht, gat, adv = rd.step(st, was, P.dt, units.area)
            finished = ender.after_step(st, rd)
            # (go 'hold': the infantry between at a contact earns nothing - the contact itself is the disobedience;
            # with it an advance under 'hold' paid +0.5 - 0.5 = 0, as much as obeying: build/diag_operative/diag.md)
            r_acc = r_acc + float(rw["between"]) * bt * (~go_hold).float() - float(rw["hit"]) * ht \
                + float(rw["gather"]) * gat * gather_case.float() - float(rw["advance"]) * (adv & go_hold).float()
            if bool(finished.any()):
                now = reward.measure(st)
                trade = (m0[:, 1, 3] - now[:, 1, 3]) - (m0[:, 0, 3] - now[:, 0, 3])
                passive = go_destroy & ~rd.contact
                r_acc = r_acc + (float(rw["trade"]) * trade - float(rw["passive"]) * passive.float()) * finished.float()
        units.end_second()
        t_sim += time.time() - t0
        if duty is not None:
            duty.end()
    if d >= 0:
        rew[d] = r_acc
        done[d] = valid[d] & st.done
    D = d + 1
    m1 = reward.measure(st)
    trade = (m0[:, 1, 3] - m1[:, 1, 3]) - (m0[:, 0, 3] - m1[:, 0, 3])
    n = max(stats["decisions"], 1)
    out_stats = {k: v for k, v in stats.items() if k != "ent"}
    out_stats.update({"battles": B, "seconds": sec, "t_op": round(t_op, 1), "t_low": round(t_low, 1),
                      "t_sim": round(t_sim, 1), "reward_mean": round(float(rew[:D].sum()) / B, 4),
                      "reforms_per_battle": round(stats["reforms"] / B, 2), "moves_per_battle": round(stats["moves"] / B, 2),
                      "groups_mean": round(stats["groups_on"] / n, 2), "tactic_fell": tac.fell,
                      "tactic_fell_share": round(tac.fell / max(tac.ticks, 1), 4), "unit_decisions": units.n,
                      "ent": {k: round(v / n, 3) for k, v in stats["ent"].items()}, "trade": round(float(trade.mean()), 4)})
    readings = {"contact": rd.contact, "between": rd.between_share(), "hit": rd.hit_share(), "gathered": rd.gathered,
                "advanced": rd.advanced, "trade": trade, "why": ender.why, "t_contact": rd.watch[0].t_contact,
                "groups": groups}
    batch = None
    if keep:
        batch = {"raw": torch.stack(store["raw"]), "groups": GR.Groups.stack(store["groups"]),
                 "act": {k: torch.stack([a[k] for a in store["act"]]) for k in store["act"][0]},
                 "lp": lp_old[:D], "value": val[:D], "reward": rew[:D], "done": done[:D], "valid": valid[:D],
                 "last_value": torch.zeros(B, device=dev), "card": setup.card, "side": setup.side,
                 "command": setup.command, "destroy": setup.destroy, "axes": axes, "bounds": st.bounds, "K": K}
    return batch, out_stats, readings


def rebuild(batch, di, bi):
    """(inp, ctx, act) of the samples (decision di, battle bi) from the stored raw numbers."""
    raw = batch["raw"][di, bi]
    g = batch["groups"].take((di, bi))
    axes = batch["axes"].take(bi)
    inp = IN.build(raw, batch["card"][bi], batch["side"][bi], axes, g, batch["command"][bi], bounds=batch["bounds"],
                   K=batch["K"])
    ctx = NN.ctx_of(inp, raw, axes, g)
    act = {k: v[di, bi] for k, v in batch["act"].items()}
    return inp, ctx, act


def measures(setup, rd):
    """{case: [B] float} the main measure of each battle (its own case's; plan.md)."""
    gathered = torch.nan_to_num(rd["gathered"], nan=0.0)
    obey = torch.where(setup.destroy, rd["contact"] & (rd["between"] > 0.5),
                       ~rd["contact"] & ~rd["advanced"]).float()
    return {"gather": gathered, "split": rd["between"], "go": obey}

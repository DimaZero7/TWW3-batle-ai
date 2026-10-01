"""Warm start: the network first copies one or more of our scripted opponents in the simulator
(behaviour cloning), then PPO improves on it (docs/en/training/training.md).

Why: from random weights every unit's order is noise, and PPO needs thousands of battles before
an army moves together (10 minutes of PPO from random weights moved the policy by a KL of ~0.003 per
update and lost every battle to the scripts). Copying `nearest` (every unit attacks the nearest
standing enemy, running) gives an army that already fights as one; PPO then has to find what beats it.
The teachers are our own scripts, not the game's AI: the check against the game's AI stays independent.

    bash tools/nn/dock.sh tools.nn.train.imitate --minutes 3 --out build/nn-train/runs/bc/bc.pt
    bash tools/nn/dock.sh tools.nn.train.imitate --teacher nearest,ai_like --generated 19 --out ...

Battles: every scene, each side played by a random script (the teachers 70 % of the time); the
sides a teacher plays are copied. The student acts nowhere: it only watches (DAgger is left to PPO).
Labels per unit that takes orders: the order kind (hold, move, attack, withdraw), the attack target,
the move / withdraw point (the nearest of the network's point bins), run.

Several teachers (`--teacher nearest,ai_like`): the copy is their mixture. Where they disagree
(ai_like holds, walks or steps back where nearest attacks) the network keeps some probability on
each: a prior PPO can move either way, instead of one that knows only "attack".
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tools.nn.model import observation as ob
from tools.nn.model.decide import frame_to
from tools.nn.sim import orders as O
from tools.nn.train import checkpoint, opponents, randomise, rollout, scenes
from tools.nn.train.run import networks

SCRIPTS = ("nearest", "hold_shoot", "hold", "ai_like")


def point_bin(cfg, x, z, obs_t, frame):
    """World points [B, N] -> the nearest move-point bin of the network (heads.point_offsets) [B, N]."""
    f, l = frame.point(x, z)
    f = f - obs_t["pos"][..., 0] * ob.POS
    l = l - obs_t["pos"][..., 1] * ob.POS
    ang = torch.atan2(l, f) % (2 * math.pi)
    direction = torch.round(ang / (2 * math.pi / cfg.n_dir)).long() % cfg.n_dir
    dist = torch.sqrt(f * f + l * l).clamp(cfg.dist_min, cfg.dist_max)
    step = (math.log(cfg.dist_max) - math.log(cfg.dist_min)) / max(1, cfg.n_dist - 1)
    k = torch.round((torch.log(dist) - math.log(cfg.dist_min)) / step).long().clamp(0, cfg.n_dist - 1)
    return direction * cfg.n_dist + k


def labels(orders, ctrl, point=None):
    """The teacher's orders [B, N] -> (kind, target, run, point) targets and the masks where they
    count: (ctrl, attack, moving (move / withdraw), running (move / attack)). KEEP counts as hold.
    point: the orders' bins [B, N] (None: move and withdraw count as hold)."""
    kind = torch.where(orders.kind == O.KEEP, O.HOLD, orders.kind)
    if point is None:
        kind = torch.where(kind == O.ATTACK, O.ATTACK, O.HOLD)
        point = torch.zeros_like(kind)
    attack = ctrl & (kind == O.ATTACK)
    moving = ctrl & ((kind == O.MOVE) | (kind == O.WITHDRAW))
    running = attack | (ctrl & (kind == O.MOVE))
    return kind, orders.target.clamp(min=0), orders.run.float(), point, ctrl, attack, moving, running


def loss_of(logits, kind, target, run, point, ctrl, attack, moving, running):
    """Cross-entropy of the kind (units that take orders), the target (attacking units), the point
    (moving units) and run (moving or attacking units)."""
    n = ctrl.float().sum().clamp(min=1)
    na = attack.float().sum().clamp(min=1)
    nm = moving.float().sum().clamp(min=1)
    nr = running.float().sum().clamp(min=1)
    lk = F.cross_entropy(logits["kind"].transpose(1, 2), kind, reduction="none")
    lt = F.cross_entropy(logits["target"].transpose(1, 2), target, reduction="none")
    lp = F.cross_entropy(logits["point"].transpose(1, 2), point, reduction="none")
    lr = F.binary_cross_entropy_with_logits(logits["run"], run, reduction="none")
    total = (lk * ctrl).sum() / n + (lt * attack).sum() / na + (lp * moving).sum() / nm + (lr * running).sum() / nr
    with torch.no_grad():
        right = torch.stack([((logits["kind"].argmax(-1) == kind) & ctrl).float().sum(), ctrl.float().sum(),
                             ((logits["target"].argmax(-1) == target) & attack).float().sum(), attack.float().sum()])
    return total, right


def train(actor, minutes=3.0, battles=1024, device="cuda", lr=1e-3, teacher="nearest", seed=0, log=print,
          max_steps=None, scene_list=scenes.SCENES, source=None):
    """teacher: a script name, or several comma-separated (copied alike). source: where battles come
    from (scenes.Generated for random armies); default the fixed scenes."""
    teachers = [k for k in teacher.split(",") if k]
    device = torch.device(device)
    rng = np.random.default_rng(seed)
    params = rollout.params_with_limit(3600)
    source = source or scenes.Fixed(np.arange(battles) % len(scene_list), scene_list, params, device)
    want = torch.zeros(battles, dtype=torch.long, device=device)
    st, setup, _ = rollout.open_rows(source, want)
    B, N = st.B, st.N
    gen = torch.Generator(device=device).manual_seed(seed)
    randomise.apply(st, torch.ones(B, dtype=torch.bool, device=device), randomise.Spread(), gen)
    look = rollout.observer(device)
    advance = rollout.battle.stepper(device)
    mem = {s: rollout.ob.start(st.observation(), setup, s) for s in (1, 2)}
    h = {s: torch.zeros(B, 1 + N, actor.cfg.d, device=device) for s in (1, 2)}

    def draw(n):
        others = len(SCRIPTS) - len(teachers)
        p = np.array([0.7 / len(teachers) if k in teachers else 0.3 / max(1, others) for k in SCRIPTS])
        return rng.choice(len(SCRIPTS), size=n, p=p / p.sum())
    taught_codes = torch.as_tensor([SCRIPTS.index(k) for k in teachers], device=device)
    who = torch.as_tensor(np.stack([draw(B), draw(B)], 1), device=device)            # [B, 2] script per side
    opt = torch.optim.Adam(actor.parameters(), lr=lr, eps=1e-5)
    actor.train()
    t0, step, stats = time.time(), 0, []
    while time.time() - t0 < minutes * 60 and (max_steps is None or step < max_steps):
        state = st.observation()
        side_orders = {k: opponents.SCRIPTS[k](st) for k in SCRIPTS}
        final = O.hold(B, N, device)
        loss_sum, right = 0.0, 0.0
        for s in (1, 2):
            obs, mem[s] = look(state, setup, s, mem[s])
            obs_t = rollout.policy.to_torch(obs, device)
            teach = O.hold(B, N, device)
            for i, k in enumerate(SCRIPTS):
                use = (who[:, s - 1] == i)[:, None].expand(B, N)
                final = O.merge(final, side_orders[k], use & (st.u["side"] == s))
                if k in teachers:
                    teach = O.merge(teach, side_orders[k], use)
            taught = torch.isin(who[:, s - 1], taught_codes)[:, None] & obs_t["ctrl"]
            point = point_bin(actor.cfg, teach.x, teach.z, obs_t, frame_to(obs.frame, device))
            logits, h_new = actor(obs_t, h[s])
            h[s] = h_new.detach()
            loss, r = loss_of(logits, *labels(teach, taught, point))
            loss_sum, right = loss_sum + loss, right + r
        opt.zero_grad(set_to_none=True)
        loss_sum.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        opt.step()
        was = st.done.clone()
        with torch.no_grad():
            advance(st, final, params, params.dt)
        fin = st.done & ~was
        if bool(fin.any()):
            rollout.restart_rows(st, setup, source, fin, want)
            randomise.apply(st, fin, randomise.Spread(), gen)
            fresh = st.observation()
            for s in (1, 2):
                mem[s] = rollout.merge_memory(mem[s], rollout.ob.start(fresh, setup, s), fin)
                h[s] = h[s] * (~fin).float()[:, None, None]
            new = torch.as_tensor(np.stack([draw(B), draw(B)], 1), device=device)
            who = torch.where(fin[:, None], new, who)
        step += 1
        r = right.tolist()
        stats.append((float(loss_sum.detach()), r[0] / max(1, r[1]), r[2] / max(1, r[3])))
        if step % 200 == 0:
            m = np.mean(stats[-200:], 0)
            log(f"bc step {step:5d} {time.time() - t0:5.0f}s loss {m[0]:.3f} kind acc {m[1]:.3f} target acc {m[2]:.3f}")
    actor.eval()
    return {"steps": step, "loss": float(np.mean(stats[-200:], 0)[0]), "kind_acc": float(np.mean(stats[-200:], 0)[1]),
            "target_acc": float(np.mean(stats[-200:], 0)[2]), "teacher": teacher}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=3.0)
    ap.add_argument("--battles", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--preset", default="small")
    ap.add_argument("--generated", type=int, default=0, help="max units a side of random armies (0: the scenes)")
    ap.add_argument("--teacher", default="nearest", help="the script(s) copied, comma-separated")
    ap.add_argument("--bank", type=int, default=2048)
    ap.add_argument("--out", default=str(checkpoint.DIR / "bc.pt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    actor, critic = networks(args.preset, torch.device(args.device))
    source = None
    if args.generated:
        from tools.nn.armies import generate
        seeds = np.random.default_rng(7).integers(generate.TRAIN_SEEDS.start, generate.TRAIN_SEEDS.stop, args.bank)
        source = scenes.Generated(seeds, args.generated, rollout.params_with_limit(3600), args.device)
    res = train(actor, args.minutes, args.battles, args.device, args.lr, teacher=args.teacher,
                log=lambda m: print(m, flush=True), source=source)
    checkpoint.save(Path(args.out), actor, critic, args.preset, dict(res, note=f"behaviour cloning of {args.teacher}"))
    print("saved", args.out, res, flush=True)


if __name__ == "__main__":
    main()

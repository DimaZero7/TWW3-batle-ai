"""Widen a trained network k times in width, keeping exactly what it computes (docs/en/training/model.md "Widening").

    bash tools/nn/dock.sh tools.nn.model.widen --src build/nn-train/test5/r7_lostworst/m15.pt \\
        --dst build/nn-train/wide/w0.pt --critic-src build/nn-train/runs/test5_r7_lostworst/latest.pt \\
        --critic-dst build/nn-train/wide/w0_critic.pt --factor 2

Every width x k: the token (residual) width d, the attention heads (the head width stays: new heads beside
the old ones), the feed-forward, the GRU, the hidden layers of the small networks (unit, context, ability
encoders, the ability key, the critic's value) and the pointers' width. The depth stays. The actor and the
critic alike. The widened network gives the same logits, memory and value (a check on simulator battles
runs after every widening: --check-battles 0 skips it).

How (Net2WiderNet, Chen et al., 2016, with exact copies):

* The token (residual) stream is copied k times: x -> [x, x]. A LayerNorm over [x, x] sees the same mean
  and variance as over x, so it gives [y, y] with its weight and bias repeated. Zero new channels would
  not do: they change the LayerNorm's mean and variance.
* What writes into the stream (the encoders' last layers, attention out, feed-forward out, the GRU) writes
  every copy: its output rows repeated. A GRU unit and its copies have equal states for ever (equal inputs,
  equal weights); a copy's gates are the unit's.
* What reads the stream reads each copy with W / k + E_j, the noises E_j summing to zero: (W / k + E_1) y +
  ... + (W / k + E_k) y = W y exactly while the copies are equal. The noise (--split-noise x the weights'
  std) breaks the symmetry: the copies get different gradients and grow apart in training.
* New hidden units (feed-forward, the encoders' and value's hidden layers, new attention heads' queries,
  keys and values): incoming weights small random (--new-scale x the layer's weights' std), bias 0,
  OUTGOING weights 0 - they add nothing until trained, and their outgoing weights get a gradient at once.
  New heads take the old heads' distance bias.
* Pointers (attack target, ability): q.k / sqrt(width). The new query dims are random, the new key dims
  0 (q.k unchanged) and the old query is scaled by sqrt(k) for the wider sqrt(width).

The optimizer is not part of a checkpoint: a run (tools/nn/train/run.py) starts its own Adam, with the
small network's optimizer steps per update (its minibatch in parts) and lr / k on the stream's readers
(stream_readers; docs/en/training/training.md "Training a widened network").
"""
import argparse
import contextlib
import copy
import dataclasses
import math
import sys
from pathlib import Path

import torch

from tools.nn.model import config as model_config
from tools.nn.model import critic as model_critic
from tools.nn.model import policy as model_policy

NEW_SCALE = 0.1     # std of a new unit's incoming weights, x the std of the layer's trained weights
SPLIT_NOISE = 0.1   # std of the zero-sum noise on the readers of the copied stream, x the weights' std


def wider_config(cfg, k):
    """The ModelConfig k times wider (the head width the same: k times the heads)."""
    return dataclasses.replace(cfg, d=cfg.d * k, heads=cfg.heads * k, pointer=cfg.pointer * k,
                               critic_d=cfg.critic_d * k, critic_heads=cfg.critic_heads * k)


# How each layer's outputs (rows) and inputs (columns) widen:
#   rows:  raw  - the same outputs (logits)          res  - the stream: each row copied k times
#          gru  - the GRU's gates r, z, n: each copied k times
#          hid  - hidden units: the old ones, then new ones (random incoming weights, bias 0)
#          qkv  - the attention's q, k, v blocks: each as hid (new heads after the old ones)
#          query - a pointer's query: hid with the old rows x sqrt(k);  key - the new rows 0
#   cols:  raw  - the same inputs                    res  - the copied stream: W / k + zero-sum noise
#          res3 - three copied streams side by side  hid  - hidden units: the new ones' columns 0
ACTOR = {"encoder.unit.0": ("hid", "raw"), "encoder.unit.2": ("res", "hid"),
         "encoder.ctx.0": ("hid", "raw"), "encoder.ctx.2": ("res", "hid"),
         "abilities.net.0": ("hid", "raw"), "abilities.net.2": ("res", "hid"),
         "memory.cell.weight_ih": ("gru", "res"), "memory.cell.weight_hh": ("gru", "res"),
         "heads.kind": ("raw", "res"), "heads.point": ("raw", "res"), "heads.run": ("raw", "res"),
         "heads.q": ("query", "res"), "heads.k": ("key", "res"), "heads.ability_q": ("query", "res"),
         "heads.ability_k.0": ("hid", "raw"), "heads.ability_k.2": ("key", "hid"),
         "heads.ability_none": ("raw", "res")}
CRITIC = {"encoder.unit.0": ("hid", "raw"), "encoder.unit.2": ("res", "hid"),
          "encoder.ctx.0": ("hid", "raw"), "encoder.ctx.2": ("res", "hid"),
          "value.0": ("hid", "res3"), "value.2": ("raw", "hid")}
BLOCK = {"qkv": ("qkv", "res"), "out": ("res", "hid"), "f1": ("hid", "res"), "f2": ("res", "hid")}


def stream_readers(model, spec):
    """Names of the weights that read the copied stream (columns "res"/"res3" above: attention q/k/v,
    feed-forward in, the GRU, the heads, the critic's value): their fan-in is k times the small one's.
    Each copy gets the gradient the small weight got, so Adam moves their sum k times as far
    (run.py gives them the learning rate / k: the first update's KL 0.0148 -> 0.0103, the small 0.0082)."""
    names = set()
    for name, _ in model.named_parameters():
        module, _, param = name.rpartition(".")
        if param in ("weight_ih", "weight_hh"):
            rule = spec.get(name)
        elif param == "weight":
            rule = spec.get(module) or (BLOCK.get(module.split(".")[-1]) if module.startswith("blocks.") else None)
        else:
            rule = None
        if rule and rule[1] in ("res", "res3"):
            names.add(name)
    return names


class Widener:
    def __init__(self, k=2, seed=0, new_scale=NEW_SCALE, split_noise=SPLIT_NOISE):
        assert k >= 2 and int(k) == k
        self.k, self.new_scale, self.split_noise = int(k), new_scale, split_noise
        self.gen = torch.Generator().manual_seed(seed)

    def _randn(self, *shape):
        return torch.randn(*shape, generator=self.gen, dtype=torch.float64)

    def cols(self, w, kind):
        """Columns of [O, I] weights -> [O, I'] (the old units' rows)."""
        k = self.k
        if kind == "raw":
            return w
        if kind == "hid":
            return torch.cat([w, w.new_zeros(w.shape[0], (k - 1) * w.shape[1])], 1)
        if kind == "res3":
            return torch.cat([self.cols(p, "res") for p in w.chunk(3, 1)], 1)
        assert kind == "res", kind
        e = self._randn(k, *w.shape) * (self.split_noise * float(w.std()))
        e = e - e.mean(0)                                    # the copies' noises sum to zero
        return torch.cat([w / k + e[j] for j in range(k)], 1)

    def new_rows(self, n, w, n_in):
        return self._randn(n, n_in) * (self.new_scale * float(w.std()))

    def linear(self, w, b, rows, cols):
        """Weights [O, I] and bias [O] (or None) -> widened, in float64 (rounded back by the caller)."""
        k = self.k
        if rows == "raw":
            return self.cols(w, cols), b
        if rows == "res":
            return torch.cat([self.cols(w, cols) for _ in range(k)]), None if b is None else b.repeat(k)
        if rows == "gru":
            ws, bs = w.chunk(3), None if b is None else b.chunk(3)
            return (torch.cat([self.cols(g, cols) for g in ws for _ in range(k)]),
                    None if b is None else torch.cat([g.repeat(k) for g in bs]))
        if rows == "qkv":
            parts = [self.linear(g, gb, "hid", cols) for g, gb in zip(w.chunk(3), b.chunk(3))]
            return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])
        old = self.cols(w, cols)
        n_new = (k - 1) * w.shape[0]
        if rows == "key":
            new = old.new_zeros(n_new, old.shape[1])
        else:
            assert rows in ("hid", "query"), rows
            new = self.new_rows(n_new, w, old.shape[1])
        if rows == "query":
            old, b = old * math.sqrt(k), b * math.sqrt(k)
        return torch.cat([old, new]), torch.cat([b, b.new_zeros(n_new)])

    def state(self, state, spec):
        """A network's state_dict (the current format) -> the widened state_dict."""
        src = {n: v.detach().to(torch.float64) for n, v in state.items()}
        out = {}
        for name, v in src.items():
            module, _, param = name.rpartition(".")
            if name in out:
                continue
            if module.endswith("dist"):                                  # [bins + 1, heads]: new heads copy
                out[name] = v.repeat(1, self.k)
            elif param in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                kind = "weight_ih" if param.endswith("ih") else "weight_hh"
                w, b = self.linear(src[f"{module}.{kind}"], src[f"{module}.bias_{kind[-2:]}"],
                                   *spec[f"{module}.{kind}"])
                out[f"{module}.{kind}"], out[f"{module}.bias_{kind[-2:]}"] = w, b
            elif v.dim() == 1 and module.split(".")[-1] in ("norm", "n1", "n2"):     # LayerNorm
                out[name] = v.repeat(self.k)
            else:
                rule = spec.get(module) or (BLOCK.get(module.split(".")[-1]) if module.startswith("blocks.") else None)
                if rule is None:
                    raise KeyError(f"no widening rule for {name}")
                w, b = self.linear(src[f"{module}.weight"], src.get(f"{module}.bias"), *rule)
                out[f"{module}.weight"] = w
                if b is not None:
                    out[f"{module}.bias"] = b
        return {n: out[n].to(state[n].dtype) for n in state}

    def actor(self, actor):
        wide = model_policy.Actor(wider_config(actor.cfg, self.k))
        wide.load_state_dict(self.state(actor.state_dict(), ACTOR))
        return wide.eval()

    def critic(self, critic):
        wide = model_critic.Critic(wider_config(critic.cfg, self.k))
        wide.load(self.state(critic.state_dict(), CRITIC))
        return wide.eval()


# --- the check: the same outputs on simulator battles ---

def battles(actor, critic, n=16, decisions=48, seed=0, device="cpu"):
    """A batch of real observations: `n` random armies (tools/nn/armies, up to 19 units a side) played by
    `actor` against ai_like for `decisions` decisions (rollout.collect: the learner's rows, both roles)."""
    from tools.nn.armies import generate
    from tools.nn.train import league, randomise, rollout, scenes
    params = rollout.params_with_limit(3600.0)
    seeds = range(generate.TRAIN_SEEDS.start + 7919 * seed, generate.TRAIN_SEEDS.start + 7919 * seed + n)
    src = scenes.Generated(seeds, 19, params, device, seed=seed, sequential=True)
    lay = league.layout(n, len(scenes.SCENES), {"ai_like": 1.0}, scene_attacker=scenes.attackers())
    env = rollout.Battles(lay, scenes.SCENES, device, params, randomise.Spread(), seed=seed, source=src)
    torch.manual_seed(seed)
    batch = rollout.collect(env, actor, critic, decisions)
    return rollout.full_obs(batch["obs"], batch["abil_static"]), batch["critic_obs"], batch["reset"]


@torch.no_grad()
def compare(small, wide, small_critic, wide_critic, obs, critic_obs, reset):
    """Max abs differences of the logits (per head), the memory (each copy) and the value; the logits'
    largest magnitude (masked choices left out); the share of equal greedy choices."""
    from tools.nn.model import heads as hd
    lo, h_small = small.sequence(obs, None, reset)
    lw, h_wide = wide.sequence(obs, None, reset)
    out = {f"logits.{n}": float((lo[n] - lw[n]).abs().max()) for n in lo}
    d = h_small.shape[-1]
    out["memory"] = max(float((h_small - c).abs().max()) for c in h_wide.split(d, -1))
    flat = {n: v.reshape(-1, *v.shape[2:]) for n, v in critic_obs.items()}
    out["value"] = float((small_critic(flat) - wide_critic(flat)).abs().max())
    out["magnitude"] = {n: float(v[v > hd.NEG / 2].abs().max()) for n, v in lo.items()}
    a, b = hd.sample(lo, greedy=True, abilities=True), hd.sample(lw, greedy=True, abilities=True)
    ctrl = obs["ctrl"]
    out["greedy_same"] = float(((a.kind == b.kind) & (a.point == b.point) & (a.target == b.target)
                                & (a.ability == b.ability))[ctrl].float().mean())
    out["unit_decisions"] = int(ctrl.sum())
    return out


@contextlib.contextmanager
def float64():
    """Networks and observations in float64 (the empty memory is made in the default dtype)."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


def double(x):
    if isinstance(x, dict):
        return {k: v.double() if v.is_floating_point() else v for k, v in x.items()}
    return copy.deepcopy(x).double().eval()


def worst(res):
    return max(v for k, v in res.items() if k.startswith(("logits.", "memory", "value")))


def check(actor, crit, wide, wide_crit, k, seed, obs, critic_obs, reset):
    """The proof that widening keeps the outputs, on a batch of real observations:

    exact   the small networks in float64, widened in float64 (the same seed): only float64 rounding
            differs (the method is exact);
    stored  the widened checkpoint (float32) against the small one: float32 rounding, no more than the
            small network's own (its float32 against its float64: floor)."""
    with float64(), torch.no_grad():
        a64, c64 = double(actor), double(crit)
        w = Widener(k, seed)
        exact = compare(a64, w.actor(a64), c64, w.critic(c64), double(obs), double(critic_obs), reset)
        lo, _ = a64.sequence(double(obs), None, reset)
    stored = compare(actor, wide, crit, wide_crit, obs, critic_obs, reset)
    with torch.no_grad():
        l32, _ = actor.sequence(obs, None, reset)
    stored["floor"] = {n: float((l32[n].double() - lo[n]).abs().max()) for n in lo}
    return exact, stored


def widen_file(src, dst, widener):
    """Widen a checkpoint: (small actor, small critic, wide actor, wide critic); dst written."""
    from tools.nn.train import checkpoint
    data = checkpoint.read(src)
    cfg = checkpoint.config_of(data)
    actor = model_policy.Actor(cfg)
    actor.load_state_dict(data["actor"])                  # an older format converts on the way in
    actor.eval()
    crit = None
    if data.get("critic") is not None:
        crit = model_critic.Critic(cfg).load(data["critic"]).eval()
    wide, wide_crit = widener.actor(actor), None if crit is None else widener.critic(crit)
    # the record: the preset of these sizes ("wide" for small x 2), else "<preset>x<k>"
    preset = next((n for n, c in model_config.PRESETS.items() if c == wide.cfg), f"{data.get('preset')}x{widener.k}")
    meta = dict(data.get("meta", {}), widened_from=str(src), factor=widener.k, new_scale=widener.new_scale,
                split_noise=widener.split_noise)
    checkpoint.save(dst, wide, wide_crit, preset, meta)
    return actor, crit, wide, wide_crit


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="the checkpoint to widen (actor and its critic)")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--critic-src", help="a second checkpoint to widen (e.g. the run's latest.pt for --critic-init)")
    ap.add_argument("--critic-dst", help="where it goes (default: <dst>_critic.pt)")
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--new-scale", type=float, default=NEW_SCALE)
    ap.add_argument("--split-noise", type=float, default=SPLIT_NOISE)
    ap.add_argument("--check-battles", type=int, default=16, help="simulator battles of the check (0: none)")
    ap.add_argument("--check-decisions", type=int, default=48)
    args = ap.parse_args()
    jobs = [(args.src, args.dst)]
    if args.critic_src:
        dst = Path(args.dst)
        jobs.append((args.critic_src, args.critic_dst or dst.with_name(dst.stem + "_critic" + dst.suffix)))
    for i, (src, dst) in enumerate(jobs):
        widener = Widener(args.factor, args.seed + i, args.new_scale, args.split_noise)
        actor, crit, wide, wide_crit = widen_file(src, dst, widener)
        n = model_policy.parameters
        print(f"{src} -> {dst}: actor {n(actor):,} -> {n(wide):,} parameters"
              + ("" if crit is None else f", critic {n(crit):,} -> {n(wide_crit):,}"), flush=True)
        if args.check_battles and crit is not None:
            obs, cobs, reset = battles(actor, crit, args.check_battles, args.check_decisions)
            exact, stored = check(actor, crit, wide, wide_crit, args.factor, args.seed + i, obs, cobs, reset)
            show = lambda r: ", ".join(f"{k} {v:.1e}" for k, v in r.items() if isinstance(v, float) and k != "greedy_same")
            print(f"   check: {args.check_battles} simulator battles x {args.check_decisions} decisions against "
                  f"ai_like, {stored['unit_decisions']} unit decisions", flush=True)
            print(f"   exact (float64): {show(exact)}; greedy choices the same {exact['greedy_same']:.4f}", flush=True)
            print(f"   stored (float32): {show(stored)}; greedy choices the same {stored['greedy_same']:.4f}", flush=True)
            print("   float32 floor (the small network's own float32 error): "
                  + ", ".join(f"{k} {v:.1e}" for k, v in stored["floor"].items())
                  + "; logits up to " + ", ".join(f"{k} {v:.0f}" for k, v in stored["magnitude"].items()), flush=True)
            if worst(exact) >= 1e-9:
                raise SystemExit(f"the widened network differs: max abs difference {worst(exact):.2e} in float64")


if __name__ == "__main__":
    main()

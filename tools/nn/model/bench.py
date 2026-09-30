"""Sizes and speed of the network: parameters per preset, time of one decision for 20 vs 20 units.

    bash tools/nn/dock.sh tools.nn.model.bench            # in the training container (torch, GPU)

A decision = observation of a side + the actor + sampling + orders (no training, random weights).
"""
import sys
import time

import torch

from tools.nn.model import config, critic, decide, factions, lora, policy, sources
from tools.nn.model import observation as ob


def sizes():
    out = {}
    for name, cfg in config.PRESETS.items():
        actor = policy.Actor(cfg)
        with_lora = policy.Actor(config.preset(name, lora_rank=8))
        per_adapter = sum(p.numel() for p in lora.adapter_parameters(with_lora)) // factions.ADAPTERS
        out[name] = {"actor": policy.parameters(actor), "critic": policy.parameters(critic.Critic(cfg)),
                     "lora_r8_per_pair": per_adapter}
    return out


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def decision_ms(cfg, device, batch=1, units=20, repeat=50, threads=None, actor_only=False):
    """Median milliseconds of one decision step for a batch of `batch` battles of units vs units
    (actor_only: the network's forward pass alone, without the observation and the orders)."""
    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(0)
    actor = policy.Actor(cfg).to(device).eval()
    setup, state = sources.synthetic(batch=batch, own=units, enemy=units)
    tstate = {k: torch.as_tensor(v, device=device) for k, v in state.items()}
    memory, h, times = None, None, []
    for i in range(repeat + 5):
        tstate["t"] = tstate["t"] + 0.5
        _sync(device)
        t0 = time.perf_counter()
        if actor_only and i > 0:
            with torch.no_grad():
                _, h = actor(obs_t, h)
        else:
            obs, memory = ob.observe(tstate, setup, 1, memory)
            orders, h, _, _ = decide.act(actor, obs, setup, h)
            obs_t = policy.to_torch(obs, device)
        _sync(device)
        if i >= 5:
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    for name, s in sizes().items():
        print(f"{name:7} actor {s['actor'] / 1e6:6.2f} M  critic {s['critic'] / 1e6:6.2f} M  "
              f"LoRA r8 per (faction, role) {s['lora_r8_per_pair'] / 1e6:5.2f} M")
    cpu = torch.device("cpu")
    for name, cfg in config.PRESETS.items():
        for threads in (1, 4):
            ms, net = decision_ms(cfg, cpu, threads=threads), decision_ms(cfg, cpu, threads=threads, actor_only=True)
            print(f"cpu  {name:7} threads {threads}: {ms:7.2f} ms / decision (20 v 20; network alone {net:6.2f} ms)")
    if torch.cuda.is_available():
        gpu = torch.device("cuda")
        print("gpu:", torch.cuda.get_device_name(0))
        for name, cfg in config.PRESETS.items():
            one, net = decision_ms(cfg, gpu), decision_ms(cfg, gpu, actor_only=True)
            many = decision_ms(cfg, gpu, batch=256, repeat=20)
            print(f"gpu  {name:7} batch 1: {one:6.2f} ms / decision (network alone {net:5.2f} ms); "
                  f"batch 256: {many:6.2f} ms ({many / 256 * 1000:6.1f} us per battle)")


if __name__ == "__main__":
    main()

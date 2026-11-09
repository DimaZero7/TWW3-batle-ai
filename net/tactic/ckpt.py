"""The tactician's checkpoint without the training (the bridge loads the networks through this file).

    restore(path, device) -> (net, critic, ck)   a tactic_v1 checkpoint (a missing 'keep' head at its start: ck["padded"])
    unit_network(device, path=None)   the frozen unit network (config train.unit_ckpt: weights/unit/...)
    save(path, net, critic, opt, cfg, updates)
The forward pass: decide.TacticNet (its input: raw.Seer -> inputs.build; decide.order_vec).
"""
import torch

from net import ROOT
from net.tactic import cfg
from net.tactic import decide as DC
from net.unit import ckpt as UT

KIND = "tactic_v1"


def unit_network(device, path=None):
    """The frozen unit network (config train.unit_ckpt): evaluation only, no gradients."""
    net, _, _ = UT.restore(str(ROOT / (path or cfg()["train"]["unit_ckpt"])), device)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def save(path, net, critic, opt, cfg_, updates):
    torch.save({"kind": KIND, "net": net.state_dict(), "critic": critic.state_dict(), "opt": opt.state_dict(),
                "cfg": cfg_, "updates": updates}, path)


def restore(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck.get("kind") != KIND:
        raise ValueError(f"not a {KIND} checkpoint: {path}")
    net, critic = DC.TacticNet().to(device), DC.TacticCritic().to(device)
    ck["padded"] = False
    sd, own = dict(ck["net"]), net.state_dict()
    for name, p in own.items():
        if name.startswith("heads.keep") and (name not in sd or sd[name].shape != p.shape):
            # a checkpoint from before the 'keep or change' head (or an older form of it): its start (decide.init_keep)
            sd[name] = p
            ck["padded"] = True
    net.load_state_dict(sd)
    critic.load_state_dict(ck["critic"])
    return net, critic, ck

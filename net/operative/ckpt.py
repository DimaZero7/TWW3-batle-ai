"""The operative's checkpoint without the training (the bridge loads the network through this file).

    restore(path, device) -> (net, critic, ck)   an operative_v1 checkpoint of config groups (another number: ValueError)
    save(path, net, critic, opt, cfg, updates)
The forward pass: net.OperativeNet (its input: inputs.build, frame; groups / step / target decode its choices).
"""
import torch

from net.operative import cfg
from net.operative import net as NN

KIND = "operative_v1"


def save(path, net, critic, opt, cfg_, updates):
    torch.save({"kind": KIND, "net": net.state_dict(), "critic": critic.state_dict(), "opt": opt.state_dict(),
                "cfg": cfg_, "updates": updates}, path)


def restore(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck.get("kind") != KIND:
        raise ValueError(f"not a {KIND} checkpoint: {path}")
    net, critic = NN.OperativeNet().to(device), NN.OperativeCritic().to(device)
    G = int(cfg()["groups"])
    have = tuple(ck["net"]["enc.number"].shape)
    if have[0] != G:
        # (the groups' number is the network's size: run5 - run8 were built for 6; config groups 40 since 07.11)
        raise ValueError(f"{path}: a network of {have[0]} groups, config groups {G} - it does not load; build anew")
    ck["padded"] = False
    for mod, key in ((net, "net"), (critic, "critic")):
        own, sd = mod.state_dict(), dict(ck[key])
        for name, p in own.items():
            if name not in sd or sd[name].shape != p.shape:
                # a checkpoint from before a head's change (net.Heads where_a, join, cmd): this layer starts afresh
                sd[name] = p
                ck["padded"] = True
        mod.load_state_dict(sd)
    return net, critic, ck

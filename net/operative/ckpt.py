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


KIND_O1 = "operative_o1_v1"           # (heads.OpNet1: drill O0-1's network, tmp/engine/design_merged.md 5)


def save_o1(path, net, extra=None):
    """An OpNet1 checkpoint, atomic: a .part file, fsync on the written handle, os.replace."""
    import os
    tmp = str(path) + ".part"
    with open(tmp, "wb") as fh:
        torch.save({"kind": KIND_O1, "net": net.state_dict(), "pick": bool(getattr(net, "use_pick", False)),
                    **(extra or {})}, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def restore_o1(path, device="cpu"):
    """(OpNet1, ck) of a save_o1 checkpoint (another kind: ValueError); the card part stays frozen."""
    from net.operative import heads as HD
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck.get("kind") != KIND_O1:
        raise ValueError(f"not a {KIND_O1} checkpoint: {path}")
    net = HD.OpNet1(card=False, pick=bool(ck.get("pick", False))).to(device)      # (heads.PICK_WHY)
    net.load_state_dict(ck["net"])
    for p in net.analysis.parameters():
        p.requires_grad_(False)
    return net, ck

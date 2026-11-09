"""The strategist's checkpoint without the training (the bridge loads the network through this file).

    restore(path, device) -> (net, critic, ck)   a strategist_v1 checkpoint of config groups (another number: ValueError)
    save(path, net, critic, opt, updates, imitated)
The forward pass: net.StrategistNet (its input: inputs.build; battle.inputs_of, deploy lays the groups out).
"""
import torch

from net.strategist import cfg
from net.strategist import net as NN

KIND = "strategist_v1"


def save(path, net, critic, opt, updates, imitated):
    torch.save({"kind": KIND, "net": net.state_dict(), "critic": critic.state_dict(), "opt": opt.state_dict(),
                "cfg": cfg(), "updates": updates, "imitated": imitated}, path)


def restore(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck.get("kind") != KIND:
        raise ValueError(f"not a {KIND} checkpoint: {path}")
    net, critic = NN.StrategistNet().to(device), NN.StrategistCritic().to(device)
    try:
        net.load_state_dict(ck["net"])
        critic.load_state_dict(ck["critic"])
    except RuntimeError as e:
        # (08.11: the strategist's groups 6 -> 40, net/config/strategist.json groups; the nets run1-run5 had 6)
        raise ValueError(f"{path}: a strategist of another number of groups than net/config/strategist.json groups "
                         f"({cfg()['groups']}): {str(e).splitlines()[0]}") from e
    return net, critic, ck

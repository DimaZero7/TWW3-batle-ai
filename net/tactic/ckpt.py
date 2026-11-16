"""The tactician's checkpoint without the training (the bridge loads the networks through this file).

    restore(path, device) -> (net, critic, ck)   by the checkpoint's kind:
              tactic_v1 - decide.TacticNet + TacticCritic (run4; a missing 'keep' head at its start: ck["padded"]);
              tactic_v2 - heads.TacticNet2 (the pointer; tmp/tactic_slice/design_attack_v23.md section 6 item 7), critic
              None; refused when its row width (rows.ROW) or its extra columns' list (learn.D2) is not this code's
    unit_network(device, path=None)   the frozen unit network (config train.unit_ckpt: weights/unit/...)
    save(path, net, critic, opt, cfg, updates)   tactic_v1
    save_v2(path, net, opt, drills, unit_commit, sim_commit, **more)   tactic_v2: {net, opt, cols (the extra columns'
              names), row_width, drills (the journal's ids it learnt), unit_commit (the unit script's), card8 (sha256 of
              weights/tactic/card8/card8.pt), sim_commit, orders (config tactic.json orders as trained), attack (config
              stage0.json drills.attack as trained)} + more
The forward pass: decide.TacticNet (its input: raw.Seer -> inputs.build; decide.order_vec); heads.TacticNet2 through
pin.NetPlayer in lower.Tacticians(player=) (its input: the same inputs.build + heads.net_inputs).
"""
import copy
import hashlib

import torch

from net import ROOT
from net.tactic import cfg
from net.tactic import decide as DC
from net.unit import ckpt as UT

KIND = "tactic_v1"
KIND2 = "tactic_v2"


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


def card8_sha(path=None):
    return hashlib.sha256((path or DC.CARD_CKPT).read_bytes()).hexdigest()


def _shape():
    from net.tactic import heads as HD
    from net.tactic import rows as RW
    return len(RW.ROW), list(HD.d2_cols()[0])


def save_v2(path, net, opt, drills, unit_commit="", sim_commit="", **more):
    from net.stage0 import cfg as scfg
    width, cols = _shape()
    assert list(net.enc.col_names) == cols, (net.enc.col_names, cols)
    torch.save({"kind": KIND2, "net": net.state_dict(), "opt": None if opt is None else opt.state_dict(),
                "cols": cols, "row_width": width, "drills": list(drills), "unit_commit": unit_commit,
                "card8": card8_sha(), "sim_commit": sim_commit, "orders": copy.deepcopy(cfg()["orders"]),
                "attack": copy.deepcopy(scfg()["drills"]["attack"]), **more}, path)


def _restore_v2(ck, path, device):
    from net.tactic import heads as HD
    width, cols = _shape()
    if ck.get("row_width") != width or list(ck.get("cols") or []) != cols:
        raise ValueError(f"{path}: trained on rows of {ck.get('row_width')} and extra columns {ck.get('cols')}; this code "
                         f"has {width} and {cols}")
    net = HD.TacticNet2(card=False).to(device)
    net.load_state_dict(ck["net"])
    for p in net.frozen():
        p.requires_grad_(False)
    return net, None, ck


def restore(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck.get("kind") == KIND2:
        return _restore_v2(ck, path, device)
    if ck.get("kind") != KIND:
        raise ValueError(f"not a {KIND} or {KIND2} checkpoint: {path}")
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

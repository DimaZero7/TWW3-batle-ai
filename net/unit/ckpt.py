"""The unit network's checkpoint and its input, without the training (the bridge loads the network through this file).

    restore(path, device) -> (net, critic, ck)   a unit_v1 checkpoint (older layouts padded with zero columns: ck["padded"])
    frozen(path, device)  the network for decisions only (eval, no gradients)
    save(path, net, critic, opt, cfg, updates)
    cards(st, device)     [B, N, passport.SIZE] the units' cards (net/cards/passport.py)
    views(raws, d_idx, b_idx, i_idx, card, area, cfg, ...)   the network's input of the samples from the raw states
                          (view.raw per decision); the forward pass: net(v, order, v["cell_m"]) -> net.sample -> decode.to_orders
"""
import numpy as np
import torch

from net.cards import passport
from net.unit import decode as Dc
from net.unit import net as Nt
from net.unit import view as V

KIND = "unit_v1"


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


def prio_now(raw_d, prio, bi, ii):
    """[R] the main target named by each unit's order (drill U8: prio [B, N] the slot, -1 none) while it is still there
    (alive, not gone: view.raw 'alive'), else -1; raw_d [B, N, RAW] the raw state now, prio None: none anywhere."""
    if prio is None:
        return torch.full_like(ii, -1)
    p = prio[bi, ii]
    alive = raw_d[bi, p.clamp(min=0), V.I["alive"]] > 0
    return torch.where((p >= 0) & alive, p, torch.full_like(p, -1))


def views(raws, d_idx, b_idx, i_idx, card, area, cfg, everyone=False, prio=None, wlim=None):
    """The views of the samples (d_idx, b_idx, i_idx: decision, battle, the network's unit's slot) rebuilt from the stored
    raw states - one viewer a sample; v["bi"] is set to the battles (to_orders writes the orders there). prio [B, N]
    (drill U8): the order's main target - its row's 'my_target' flag; v["prio"] [M] its slot now (prio_now).
    wlim: decode.width_limits [B, N] - v["width_ok"] [M, WIDTHS], the width head's ban (a line wider than the area's
    front; decode.width_ok); None: no ban. v["shooter"] [M]: the viewer has a missile weapon (obedience_parts)."""
    now = raws[d_idx, b_idx]                                              # [M, N, RAW]
    T = V.T
    hist = torch.zeros((T,) + now.shape, device=now.device)
    for k in range(T):
        d = d_idx - 1 - k
        ok = d >= 0
        hist[k] = torch.where(ok[:, None, None], raws[d.clamp(min=0), b_idx], torch.zeros((), device=now.device))
    ar = {k: a[b_idx] for k, a in area.items()}
    M = now.shape[0]
    m_idx = torch.arange(M, device=now.device)
    who = torch.zeros(now.shape[:2], dtype=torch.bool, device=now.device)
    who[m_idx, i_idx] = True
    p = prio_now(now, None if prio is None else prio[b_idx], m_idx, i_idx)
    tgt = torch.full(now.shape[:2], -1, dtype=torch.long, device=now.device)
    tgt[m_idx, i_idx] = p
    # the map's cell: infantry cfg cell_m, a shooter its range / view.RANGE_CELLS (its range circle on the map)
    cm = V.cell_size(now[m_idx, i_idx], cfg["cell_m"])
    v = V.build(now, hist, card[b_idx], who, order_target=tgt, area=ar, cell_m=cm, bounds=cfg["bounds"],
                everyone=everyone)
    v["bi"] = b_idx
    v["prio"] = p
    v["cell_m"] = cm
    v["shooter"] = now[m_idx, i_idx, V.I["range"]] > 0
    if wlim is not None:
        v["width_ok"] = Dc.width_ok(wlim[0][b_idx, i_idx], wlim[1][b_idx, i_idx],
                                    None if area is None else area["front"][b_idx, i_idx])
    return v


def save(path, net, critic, opt, cfg, updates=0):
    torch.save({"kind": KIND, "net": net.state_dict(), "critic": critic.state_dict(), "opt": opt.state_dict(),
                "cfg": cfg, "updates": updates}, path)


def restore(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck.get("kind") != KIND:
        raise ValueError(f"not a {KIND} checkpoint: {path}")
    net, critic = Nt.UnitNet().to(device), Nt.Critic().to(device)
    ck["padded"] = False
    for mod, key in ((net, "net"), (critic, "critic")):
        sd = dict(ck[key])
        own = mod.state_dict()
        w = sd["trunk.core.net.0.weight"]
        want = mod.trunk.core.net[0].weight.shape[1]
        if w.shape[1] < want:
            # a checkpoint from before the order's later fields (the fire sector, U5): their columns zero - the old
            # network's outputs unchanged for any order - inserted before the last column (the cell size)
            new = torch.zeros((w.shape[0], want), dtype=w.dtype, device=w.device)
            new[:, :w.shape[1] - 1] = w[:, :-1]
            new[:, -1] = w[:, -1]
            sd["trunk.core.net.0.weight"] = new
            ck["padded"] = True
        w = sd["trunk.map.c1.weight"]
        want = own["trunk.map.c1.weight"].shape[1]
        if w.shape[1] < want:
            # a checkpoint from before the shooter's map layers (zmap.LAYERS: reach, fire_arc - the last ones): their
            # input channels zero, the old map features unchanged
            new = torch.zeros((w.shape[0], want) + tuple(w.shape[2:]), dtype=w.dtype, device=w.device)
            new[:, :w.shape[1]] = w
            sd["trunk.map.c1.weight"] = new
            ck["padded"] = True
        for name, col in (("trunk.rows.state.weight", None), ("trunk.memory.gru.weight_ih_l0", -1)):
            # a checkpoint from before the row's later columns (view.ROW: dh, the ground's height - the last): their
            # input columns zero; the memory's GRU takes the row and the snapshot's mask last, so its new columns go
            # before that one
            if name in sd and sd[name].shape[1] < own[name].shape[1]:
                w = sd[name]
                new = torch.zeros(own[name].shape, dtype=w.dtype, device=w.device)
                if col is None:
                    new[:, :w.shape[1]] = w
                else:
                    new[:, :w.shape[1] - 1] = w[:, :-1]
                    new[:, -1] = w[:, -1]
                sd[name] = new
                ck["padded"] = True
        for name, p in own.items():
            if name not in sd and name.startswith("width."):
                # a checkpoint from before the width head: its start (net.init_width: zero weights, 'keep' first)
                sd[name] = p
                ck["padded"] = True
        mod.load_state_dict(sd)
    return net, critic, ck


def frozen(path, device="cpu"):
    """The network of a checkpoint for decisions only: eval, no gradients."""
    net, _, _ = restore(str(path), device)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net

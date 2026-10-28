"""A new width of the line (Orders.width): the limits, the re-form's time and the change of the box over it
(config/nn/sim.json formation.reform with its sources; build/diag_g49/formation_width/mechanic.md).

In the game a width order is the move order of a facing (goto_location_angle_width on the unit's own place): the men
walk to their new places. Here:

* limits(u, rf): the narrowest files max(min_files, round(men x wmin_k)), the widest one rank of all the men (not wider
  than max_m if set); the width is snapped to whole files of the template's h (snap);
* ready_s: the time the men need - widening (front gain / 2) / speed (+ walk_widen_extra_s at a walk, standing, out of
  melee), narrowing the new depth / speed; the speed is the unit's walk or run by the order, in melee the walk;
* order(): a width order given now (HOLD / MOVE / WITHDRAW, not ATTACK; a formation whose width is not fixed) starts a
  re-form from the width it has now; on a move the narrowing is ready no sooner than the arrival;
* advance(): each step the width goes evenly from the old to the new over the re-form's time; a held unit keeps its
  front rank where it is (its centre moves by half the depth's change); a rout ends the re-form.
The melee exit's window of such an order given in melee is battle.py's (contact.face_breakoff).
"""
import torch

from tools.nn.sim import geometry


def on(params):
    rf = params.sim["formation"].get("reform") or {}
    return bool(rf.get("on"))


def _hv(u, spacing):
    h = torch.where(u["sp_h"] > 0, u["sp_h"], torch.full_like(u["sp_h"], float(spacing)))
    v = torch.where(u["sp_v"] > 0, u["sp_v"], torch.full_like(u["sp_v"], float(spacing)))
    return h, v


def file_limits(u, rf, men=None):
    """(least, most) files [B, N] float: least max(min_files, round(men x wmin_k)), most all the men in one rank (and
    within max_m); never fewer than 1 or more than the men. men: [B, N] (default the men left)."""
    men = (u["men"] if men is None else men).clamp(min=1)
    least = torch.maximum(torch.round(men * u["wmin_k"]), torch.full_like(men, float(rf.get("min_files", 2))))
    most = men.clone()
    if rf.get("max_m"):
        h, _ = _hv(u, 1.5)
        most = torch.minimum(most, torch.floor(float(rf["max_m"]) / h + 1e-4).clamp(min=1))
    least = torch.minimum(least, men)
    return least, torch.maximum(most, least)


def limits(u, rf, spacing, men=None):
    """(narrowest, widest) width [B, N], m (whole files x h); men as file_limits."""
    h, _ = _hv(u, spacing)
    least, most = file_limits(u, rf, men)
    return least * h, most * h


def snap(u, width, rf, spacing):
    """[B, N] the width the unit re-forms to for an ordered one (m): floor(width / h) files (as geometry.dims reads a
    width) within file_limits, x h."""
    h, _ = _hv(u, spacing)
    least, most = file_limits(u, rf)
    files = torch.maximum(torch.minimum(torch.floor(width / h + 1e-4), most), least)
    return files * h


def ready_s(u, w_from, w_to, speed, extra, spacing):
    """[B, N] the re-form's time, s: widening (front gain / 2) / speed + extra, narrowing (ranks - 1) x v / speed of the
    new box (the game's measured depth: the men's centres, data/roster). Also returns widen [B, N] bool."""
    h, v = _hv(u, spacing)
    men = u["men"].clamp(min=1)
    f0 = torch.minimum(torch.floor(w_from / h + 1e-4).clamp(min=1), men)
    f1 = torch.minimum(torch.floor(w_to / h + 1e-4).clamp(min=1), men)
    widen = f1 > f0
    gain = (f1 - f0) * h
    depth = (torch.ceil(men / f1) - 1).clamp(min=0) * v
    sp = speed.clamp(min=0.1)
    return torch.where(widen, gain / 2 / sp + extra, depth / sp), widen


def order(u, width, take, kind, run, point, rf, spacing):
    """Start the re-forms the orders give now (in place on u: wnew, w0, wT, ws). width [B, N] the orders' (m, < 0 none),
    take [B, N] the units that took an order (not KEEP), kind / run / point (MOVE or WITHDRAW) the orders in force.
    Returns new [B, N] bool: a width order given now - another number than the width order in force (ow; the same one
    re-given each step is no new order, as a facing) whose snapped width differs from the one in force or being formed -
    in melee it opens the melee exit's window (battle.py)."""
    from tools.nn.sim import orders as O
    able = take & (width >= 0) & (kind != O.ATTACK) & (u["men0"] > 1) & (u["wmin_k"] > 0)
    given = able & ((width - u["ow"]).abs() > 1e-3)
    u["ow"] = torch.where(able, width, u["ow"])
    goal = snap(u, width.clamp(min=0), rf, spacing)
    now_goal = torch.where(u["wnew"] >= 0, u["wnew"], u["width"])
    new = given & ((goal - now_goal).abs() > 1e-3)
    melee = u["m"]
    pd = torch.sqrt((u["ox"] - u["x"]) ** 2 + (u["oz"] - u["z"]) ** 2)
    moving = point & (pd > 1.0) & ~melee
    speed = torch.where(melee | ~run, u["walk"], u["run"])
    extra = torch.where(~melee & ~moving & ~run, torch.full_like(speed, float(rf.get("walk_widen_extra_s", 0.0))),
                        torch.zeros_like(speed))
    T, widen = ready_s(u, u["width"], goal, speed, extra, spacing)
    T = torch.where(moving & ~widen, torch.maximum(T, pd / speed.clamp(min=0.1)), T)
    u["w0"] = torch.where(new, u["width"], u["w0"])
    u["wnew"] = torch.where(new, goal, u["wnew"])
    u["wT"] = torch.where(new, T, u["wT"])
    u["ws"] = torch.where(new, torch.zeros_like(u["ws"]), u["ws"])
    return new


def advance(u, dt, spacing, held, melee_depth=0.0):
    """One step of the re-forms (in place): the width from w0 to wnew evenly over wT; held [B, N] bool (standing under a
    hold): the front rank stays, the centre moves back (forward) by half the depth's gain (loss)."""
    alive = (u["men"] > 0) & ~u["gone"]
    active = (u["wnew"] >= 0) & alive & ~u["r"]
    u["wnew"] = torch.where((u["wnew"] >= 0) & ~active, torch.full_like(u["wnew"], -1.0), u["wnew"])
    _, d0 = geometry.dims(u, spacing, melee_depth)
    u["ws"] = torch.where(active, u["ws"] + dt, u["ws"])
    frac = (u["ws"] / u["wT"].clamp(min=1e-6)).clamp(max=1.0)
    ready = active & (u["ws"] >= u["wT"] - 1e-6)
    w = torch.where(ready, u["wnew"], u["w0"] + (u["wnew"] - u["w0"]) * frac)
    u["width"] = torch.where(active, w, u["width"])
    u["wnew"] = torch.where(ready, torch.full_like(u["wnew"], -1.0), u["wnew"])
    _, d1 = geometry.dims(u, spacing, melee_depth)
    back = torch.where(active & held, (d1 - d0) / 2, torch.zeros_like(d0))
    b = u["b"] * geometry.DEG
    u["x"] = u["x"] - torch.sin(b) * back
    u["z"] = u["z"] - torch.cos(b) * back

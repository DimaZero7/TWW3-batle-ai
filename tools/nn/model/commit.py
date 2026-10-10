"""The commitment (v2, ModelConfig.sectors > 0): with every new order a unit also chooses how long it keeps it.

The chained heads (chain.py) choose, with a new order (hold, move, attack, withdraw), one of DURATIONS (2, 4, 8,
16 s). Until it runs out the unit's decisions are forced to keep (the order in force goes on): the order kind's
choice is only keep (masked: its log-probability is 0, no gradient, no entropy), except when an INTERRUPT happens -
then the unit decides again at once:
* it came into melee (a fight began: it attacked or was attacked; the token's `melee` was off at the last decision);
* an enemy has threatened its flank or rear for THREAT_HOLD_S (the token's threat_left / right / rear on at every
  decision since: a charge coming in; once per such spell - in the game the flags flicker, 2.4-3.4 rising edges a
  unit-minute against the simulator's 0.9, and every flicker freed the unit: build/v2/ana_game2.py, 08.10.2026);
* its attack's target died, routs or shatters, or is no longer seen (no longer in Obs.target_ok, or routing);
* it routed or rallied (its routing state changed);
* its own lord died (the context's own-lord-slain came on);
* an enemy comes at it (approach): a seen enemy, not routing, moving towards the unit at APPROACH_V m/s or faster
  (its own velocity projected on the line to the unit: our unit's own walk into a standing enemy is not it) and near
  - within NEAR_M, or within LEAD_S s at that speed (a cavalry charge at 8 m/s: 64 m), never beyond WARN_M. Once per
  enemy and unit: the pair stays warned while the enemy is within WARN_M, seen and not routing (the token's velocity
  comes from two positions a second apart: a speed near the threshold must not free the unit every second);
* it bleeds: it lost more than LOSS of its health (the token's hp) over the last LOSS_S s (the hp of the last HIST
  decisions); once per such spell (as the threat), again only after a decision without it.
A unit that chooses keep itself starts no commitment. Routing units take no orders at all (Obs.ctrl).

All from the side's own observation (the unit tokens, Obs.target_ok, the context) and the battle time: the simulator
(tools/nn/train/rollout.py) and the companion (tools/nn/companion, the game's state each second) keep it the same way.
The state is a dict of tensors [B, N] (until: battle time it ends, -1 none; target: the committed attack's target,
-1 none; melee, rout: the flags at the last decision; threat_t: the time the threat came on, -1 off; threat: its
spell has counted as an event; bleed: the unit was bleeding at the last decision), warned [B, N, N] (own unit i,
enemy j: j's approach has counted), hp_h [B, N, HIST] (the unit's hp at the last HIST decisions, -1 none) with hp_t
[B, HIST] (their times, -1 none) and lord [B] (own lord dead at the last decision).

Why these numbers (the network audit build/net_audit, 10.10.2026: the commitment had come down to 2 s (47 %) and
16 s (41 %), and nothing ended a 16-s order - a shooter walked 16 s into a charge):
* NEAR_M 40 m: the game's charge sprint starts at its charge distance, 30 m edge to edge for infantry, 35 m for lords
  (battle_entities charge_distance_commence_run; the morale probe saw the charge start at 36-40 m centre to centre,
  config/nn/sim.json morale.charge): the last moment to answer a charge before it lands;
* LEAD_S 8 s: half the longest commitment - a faster enemy is announced earlier (infantry at 4 m/s: 40 m, cavalry at
  8 m/s: 64 m), with the time to turn and walk away or brace;
* APPROACH_V 1 m/s: below every unit's walk (1.2-1.6 m/s), above the jitter of a standing unit's centre;
* WARN_M 100 m: the most the warning reaches (12.5 m/s x LEAD_S), and how far a warned enemy must go before it can
  warn again;
* LOSS 5 % over LOSS_S 4 s (HIST 4 decisions at the network's 1 s cadence): ~1.25 %/s, 75 % a minute - heavy missile
  fire or a fight going badly (the melee itself is its own event).

The network sees it (Obs "commit" [B, N, 2]): the seconds left / LEFT_MAX and whether the unit is held now; and
Obs "free" [B, N]: the units that decide now (take orders and are not held).
"""
import torch

from tools.nn.model import factions
from tools.nn.model import observation as ob
from tools.nn.sim.orders import ATTACK, KEEP

DURATIONS = (2.0, 4.0, 8.0, 16.0)
LEFT_MAX = 16.0
THREAT_HOLD_S = 2.0                   # s: a flank / rear threat ends a commitment once it has lasted this long
OWN_LORD = factions.SIZE + 8          # the context's own lord slain (observation._context + lords)
THREAT = tuple(ob.INDEX[n] for n in ("threat_left", "threat_right", "threat_rear"))
NEAR_M = 40.0                         # m: an approaching enemy this near ends a commitment (module doc)
LEAD_S = 8.0                          # s: ... or one that would be here within this at its speed
APPROACH_V = 1.0                      # m/s: ... approaching at least this fast
WARN_M = 100.0                        # m: ... never farther; a warned enemy warns again only after going past it
LOSS = 0.05                           # share of the unit's health lost ...
LOSS_S = 4.0                          # ... within this many seconds ends a commitment (bleeding)
HIST = 4                              # decisions of hp kept (LOSS_S at the 1 s cadence)
KEYS = ("until", "target", "melee", "threat", "threat_t", "rout", "lord", "warned", "hp_h", "hp_t", "bleed")


def start(B, N, device=None):
    """No commitment, nothing seen yet."""
    z = torch.zeros((B, N), device=device)
    f = torch.zeros((B, N), dtype=torch.bool, device=device)
    return {"until": z - 1, "target": torch.full((B, N), -1, dtype=torch.long, device=device), "melee": f,
            "threat": f.clone(), "threat_t": z - 1, "rout": f.clone(),
            "lord": torch.zeros(B, dtype=torch.bool, device=device),
            "warned": torch.zeros((B, N, N), dtype=torch.bool, device=device),
            "hp_h": torch.full((B, N, HIST), -1.0, device=device), "hp_t": torch.full((B, HIST), -1.0, device=device),
            "bleed": f.clone()}


def _flags(obs_t):
    tok = obs_t["tokens"]
    melee = tok[..., ob.INDEX["melee"]] > 0.5
    threat = (tok[..., list(THREAT)] > 0.5).any(-1)
    rout = (tok[..., ob.INDEX["state_routing"]] + tok[..., ob.INDEX["state_shattered"]]) > 0.5
    lord = obs_t["ctx"][:, OWN_LORD] > 0.5
    return melee, threat, rout, lord


def _threat(state, threat, t):
    """(since [B, N]: the time the threat came on, -1 off; steady [B, N]: on for THREAT_HOLD_S or longer)."""
    since = torch.where(threat, torch.where(state["threat_t"] >= 0, state["threat_t"], t.expand_as(state["threat_t"])),
                        torch.full_like(state["threat_t"], -1.0))
    return since, threat & (t - since >= THREAT_HOLD_S - 1e-6)


def _approach(obs_t):
    """(coming [B, N, N]: enemy j comes at own unit i now, near [B, N, N]: j is within WARN_M of i, seen, not routing)
    (module doc)."""
    tok = obs_t["tokens"]
    pos = obs_t["pos"].float() * ob.POS                                            # [B, N, 2] m
    vel = torch.stack([tok[..., ob.INDEX["vel_fwd"]], tok[..., ob.INDEX["vel_lat"]]], -1).float() * ob.VEL
    rel = pos[:, :, None, :] - pos[:, None, :, :]                                  # [B, i, j, 2]: from j to i
    d = rel.square().sum(-1).clamp(min=1e-6).sqrt()
    speed = (vel[:, None, :, :] * rel).sum(-1) / d                                # j's speed towards i, m/s
    rout = (tok[..., ob.INDEX["state_routing"]] + tok[..., ob.INDEX["state_shattered"]]) > 0.5
    enemy = obs_t["target_ok"] & ~rout                                             # seen now, alive, not routing
    pair = (obs_t["own"] & obs_t["attend"])[:, :, None] & enemy[:, None, :]
    near = pair & (d <= WARN_M)
    reach = torch.clamp(speed * LEAD_S, min=NEAR_M)
    return near & (speed >= APPROACH_V) & (d <= reach), near


def _bleeding(state, obs_t, t):
    """(bleeding [B, N]: lost more than LOSS of its health over the last LOSS_S s (the state's hp_h, hp_t), hp now
    [B, N] (0: not known, a NaN read)). t [B, 1]."""
    hp = obs_t["tokens"][..., ob.INDEX["hp"]].float()
    recent = (state["hp_t"] >= 0) & (t - state["hp_t"] <= LOSS_S + 1e-3)          # [B, HIST]
    past = torch.where(recent[:, None, :] & (state["hp_h"] > 0), state["hp_h"], torch.zeros_like(state["hp_h"]))
    top = past.max(-1).values
    return (hp > 0) & (top > 0) & (top - hp > LOSS), hp


def interrupts(state, obs_t, t):
    """[B, N] the units whose commitment an event ends now (module doc); t [B, 1] the battle time."""
    melee, threat, rout, lord = _flags(obs_t)
    _, steady = _threat(state, threat, t)
    coming, _ = _approach(obs_t)
    bleed, _ = _bleeding(state, obs_t, t)
    tgt = state["target"]
    has = tgt >= 0
    t = tgt.clamp(min=0)
    gone = has & (~obs_t["target_ok"].gather(1, t) | rout.gather(1, t))
    return ((melee & ~state["melee"]) | (steady & ~state["threat"]) | (rout != state["rout"]) | gone
            | (lord & ~state["lord"])[:, None] | (coming & ~state["warned"]).any(-1) | (bleed & ~state["bleed"]))


def inputs(state, obs_t, t):
    """obs_t with the commitment's inputs: "free" [B, N] (decides now), "commit" [B, N, 2] (seconds left / LEFT_MAX,
    held now). t: the battle time [B] (s)."""
    t = torch.as_tensor(t, device=obs_t["own"].device).float().reshape(-1, 1)
    held = obs_t["ctrl"] & (state["until"] > t) & ~interrupts(state, obs_t, t)
    left = torch.where(held, state["until"] - t, torch.zeros_like(state["until"]))
    out = dict(obs_t)
    out["free"] = obs_t["ctrl"] & ~held
    out["commit"] = torch.stack([left / LEFT_MAX, held.float()], -1)
    return out


def apply(state, obs_t, t, action):
    """The state after the decision (obs_t from inputs(), action with .commit): a free unit's new order starts its
    commitment; a held unit keeps it; an interrupted one, a keep chosen or a unit out of orders: none."""
    t = torch.as_tensor(t, device=obs_t["own"].device).float().reshape(-1, 1)
    held = obs_t["commit"][..., 1] > 0.5
    new = obs_t["free"] & (action.kind != KEEP)
    dur = torch.as_tensor(DURATIONS, device=t.device)[action.commit.clamp(0, len(DURATIONS) - 1)]
    until = torch.where(new, t + dur, torch.where(held, state["until"], torch.full_like(state["until"], -1.0)))
    target = torch.where(new, torch.where(action.kind == ATTACK, action.target, torch.full_like(action.target, -1)),
                         torch.where(held, state["target"], torch.full_like(state["target"], -1)))
    melee, threat, rout, lord = _flags(obs_t)
    since, steady = _threat(state, threat, t)
    coming, near = _approach(obs_t)
    bleed, hp = _bleeding(state, obs_t, t)
    return {"until": until, "target": target, "melee": melee, "threat": steady, "threat_t": since, "rout": rout,
            "lord": lord, "warned": (state["warned"] | coming) & near,
            "hp_h": torch.cat([state["hp_h"][..., 1:], hp[..., None]], -1),
            "hp_t": torch.cat([state["hp_t"][:, 1:], t.reshape(-1, 1)], -1), "bleed": bleed}


def reset(state, rows):
    """The rows [B] (bool) start afresh (a new battle), in place."""
    fresh = start(*state["until"].shape, device=state["until"].device)
    for k in KEYS:
        r = rows.reshape(-1, *([1] * (state[k].dim() - 1)))
        state[k].copy_(torch.where(r, fresh[k], state[k]))

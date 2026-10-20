"""One step of every battle in the batch, and running battles to the end
(docs/en/training/simulator.md).

A step (0.5 s): take the orders -> contacts -> melee blows and shots -> damage, men, kills ->
morale (wavering, rout, rally) -> fatigue -> movement -> the observed flags -> is the battle over.
A side loses when none of its units stands (all dead, routing, shattered or gone); at the time
limit the defender wins (the side that is not the attacker).
"""
import math

import torch

from tools.nn.sim import abilities, effects, fatigue, geometry, knobs, magic, melee, missile, morale, movement, skirmish
from tools.nn.sim import orders as O
from tools.nn.sim.params import load


def threat_flags(u, pw, params, *, alive=None, standing=None):
    """Native lf/rf/bf proxies, shared by observation and exposed-flank morale.

    The optional geometry trial requires the threatening enemy to face the unit.
    It does not redefine the companion's native flags or the direction of a hit.
    """
    cal = params.sim["threat"]
    trial = cal.get("calibration", {})
    enabled = trial.get("on", False)
    radius = trial["range_m"] if enabled else cal["range_m"]
    front = trial["front_deg"] if enabled else 60
    rear = trial["rear_deg"] if enabled else 120
    if alive is None:
        alive = (u["side"] > 0) & (u["men"] > 0) & ~u["gone"]
    if standing is None:
        standing = alive & ~u["r"]
    threat = pw["enemy"] & standing[:, None, :] & (pw["dist"] <= radius)
    if enabled:
        threat = threat & (pw["rel_j"].abs() <= trial["facing_deg"] * geometry.DEG)
    rel = pw["rel_i"] / geometry.DEG
    return {"lf": alive & (threat & (rel < -front) & (rel > -rear)).any(2),
            "rf": alive & (threat & (rel > front) & (rel < rear)).any(2),
            "bf": alive & (threat & (rel.abs() >= rear)).any(2)}


def step(st, orders, params=None, dt=None):
    """Advance every unfinished battle of st by one step (in place). orders: tools/nn/sim/orders.Orders."""
    params = params or load()
    dt = dt or params.dt
    u = st.u
    old = dict(u)
    live = ~st.done
    cal = params.sim
    R = params.battle
    spacing = cal["formation"]["spacing_m"]
    rect = bool(cal["contact"].get("rect_gap"))     # the formations' true rectangle gap (geometry.rect_gap)
    mdepth = geometry.melee_depth(cal)              # a formation in melee keeps room for its dead (geometry.dims)

    present = u["side"] > 0
    alive = present & (u["men"] > 0) & ~u["gone"]
    standing = alive & ~u["r"]
    same_side = u["side"][:, :, None] == u["side"][:, None, :]
    N = st.N
    eye = torch.eye(N, dtype=torch.bool, device=st.device)[None]

    # (the skirmish run's far point shown in ox, oz last step - skirmish.show_far_point, below: the order's own point back)
    if "sk_shown" in u:
        u["ox"] = torch.where(u["sk_shown"], u["sk_ox"], u["ox"])
        u["oz"] = torch.where(u["sk_shown"], u["sk_oz"], u["oz"])
    # --- orders --- (KEEP: the order in force goes on)
    take = standing & (orders.kind != O.KEEP)
    kind = torch.where(take, orders.kind, u["order_kind"])
    tgt = torch.where(take, orders.target, u["order_target"])
    # A withdraw always runs, whatever the order's run flag: in the game the companion and the bridge give it at a run
    # (exchange.orders_list, goto_location(point, true); the gate's withdrawing units ran 89 % of their seconds at 3.0
    # m/s median, build/audit_orders), as ai_like gives it. Before, the network's withdraw walked here (decide.to_orders
    # keeps the run flag of MOVE and ATTACK only) and ran in the game.
    run = torch.where(take, orders.run | (orders.kind == O.WITHDRAW), u["order_run"])
    t_ok = (tgt >= 0) & alive.gather(1, tgt.clamp(min=0)) & ~same_side.gather(2, tgt.clamp(min=0)[:, :, None]).squeeze(2)
    kind = torch.where((kind == O.ATTACK) & ~t_ok, torch.full_like(kind, O.HOLD), kind)
    tgt = torch.where(kind == O.ATTACK, tgt, torch.full_like(tgt, -1))
    # --- spells (magic.py; config/nn/sim.json magic): the winds flow, casts asked (magic.request) start, a caster out of
    # range runs in, one winding up stands; their orders here replace the given ones ---
    cast_move = None
    if magic.on(params) and "cast_k" in u:
        # a cast ordered (Orders.ability SPELL_AT + k at Orders.cast_target: the network's) is asked of the caster now
        magic.request(u, orders.spell(), orders.cast_target)
        kind, tgt, run, cast_move, cast_x, cast_z = magic.steer(u, params, dt, kind, tgt, run, standing, alive, same_side)
    point = (kind == O.MOVE) | (kind == O.WITHDRAW)
    # another order than the one in force (kind, target or a point moved over 1 m): ends a latched leave (below)
    new_order = (kind != u["order_kind"]) | (tgt != u["order_target"]) | (
        take & point & (((orders.x - u["ox"]) ** 2 + (orders.z - u["oz"]) ** 2) > 1.0))
    # (the skirmish run's cut, skirmish.py: another kind, another target or a point moved new_point_m or more)
    sk_cal = cal.get("skirmish") or {}
    sk_new = (kind != u["order_kind"]) | (tgt != u["order_target"]) | (take & point & (
        ((orders.x - u["ox"]) ** 2 + (orders.z - u["oz"]) ** 2)
        >= knobs.get(u, "sk_new_point_m", float(sk_cal.get("new_point_m", 5.0))) ** 2))
    u["ox"] = torch.where(take & point, orders.x, u["ox"])
    u["oz"] = torch.where(take & point, orders.z, u["oz"])
    if cast_move is not None:
        u["ox"] = torch.where(cast_move, cast_x, u["ox"])
        u["oz"] = torch.where(cast_move, cast_z, u["oz"])
    # A new order (another kind, or another attack target) makes a shooter aim again (missile.aim_reset_on_order;
    # measured: a firing unit given one in the game shoots its next 10 s at 0.6-0.75 of the rate of one given none).
    # By category, measured (missile.order_quiet_why): direct fire and throwing weapons lose nothing (rt_s 0: loaded men
    # fire at the new target at once); arcing fire already aiming keeps quiet_s (8.5 s) without a shot from each new
    # order whatever its reload (the window does not hide in it), then fires by its reload.
    changed = (kind != u["order_kind"]) | (tgt != u["order_target"])
    if cal["missile"].get("aim_reset_on_order"):
        own_rt = u["rt_s"] >= 0
        u["aim"] = torch.where(changed, torch.where(own_rt, torch.minimum(u["aim"], u["aim_s"] - u["rt_s"]),
                                                    torch.zeros_like(u["aim"])), u["aim"])
        if cal["missile"].get("burst_cut_reload"):
            # a burst weapon given a new order mid-burst ends the burst: the men who had begun it reload from the cut
            # (the cycle less the burst), the others stay loaded; a new full burst once loaded and spun up
            # (missile.burst_cut_reload; the ratlingrt / ratlingrt2 probes)
            missile.burst_cut(u, changed, float(cal["missile"].get("burst_cut_spread_s") or 0.0))
    u["order_kind"], u["order_target"], u["order_run"] = kind, tgt, run
    if "order_s" in u:
        # the order's clock (contact.fresh_incidental, melee.py): a new order given in melee starts it at 0, out of
        # melee at the window (an order given before the fight is not a fresh order in melee)
        win = float(R.get("melee_breakoff_secs", 0.0))
        u["order_s"] = torch.where(new_order, torch.where(u["m"], torch.zeros_like(u["order_s"]),
                                                          torch.full_like(u["order_s"], win)), u["order_s"] + dt)

    # --- contacts ---
    pw = geometry.pairwise(u, spacing, rect=rect, melee_depth=mdepth)
    both = alive[:, :, None] & alive[:, None, :]
    # Formations touch when their edges overlap by -reach_m (the game's formations step into each other at the
    # contact); a lone man (a lord) stops at a formation's edge (contact.lord_reach_m). Units already fighting stay in
    # contact a little longer (contact.hold_m; with contact.engage two formations stay in contact while their outer men
    # are within the database's formed combat distance: geometry.contact_limit).
    lim = geometry.contact_limit(u, pw, cal["contact"], R, spacing)
    # The rout's exit (contact.rout_pin_s; config/nn/sim.json contact.rout_pin_why): a formation that began to rout in
    # melee keeps its men in contact for rout_pin_s seconds - the contact test does not count the way it has run since
    # (rpin_d), so the enemies it fought go on striking it (at the pursuit rate, melee.py) while it pulls away.
    gap_c = pw["gap"]
    if cal["contact"].get("rout_pin_s") and "rpin_s" in u:
        back = torch.where(alive & u["r"] & (u["rpin_s"] > 0), u["rpin_d"], torch.zeros_like(u["rpin_d"]))
        gap_c = gap_c - back[:, :, None] - back[:, None, :]
    touch = pw["enemy"] & both & (gap_c <= lim)
    # The pursuit's reach (contact.pursue_gap_m; config/nn/sim.json contact.pursue_gap_why; absent / 0: off): a router
    # stays in contact with a formation chasing it - a standing enemy, not a missile unit, with an attack order on it -
    # while their formations' edges are within pursue_gap_m (the pursuers' men pick charge targets among the routing
    # mob's men within the database's pursue_charge_max_unit_separation 8 m, and the mob strings out ~8 m beyond its box);
    # the chaser strikes it (melee.py: the pursuit's rate), the router counts as in melee (u["m"] below), the chaser's own
    # melee flag is not lit by it (the game's: off 5 s after the rout). touch_geo: the contacts without these pairs (the
    # missiles' contact test, the morale's fought enemies, the chaser's melee flag).
    touch_geo = touch
    chase_touch = torch.zeros_like(touch)
    pursue_gap = float(cal["contact"].get("pursue_gap_m") or 0.0)
    if pursue_gap > 0:
        chaser = alive & ~u["r"] & (kind == O.ATTACK) & (tgt >= 0) & (u["range"] <= 0)
        onto = tgt[:, :, None] == torch.arange(N, device=tgt.device)[None, None, :]
        chase_touch = (chaser[:, :, None] & onto & (alive & u["r"])[:, None, :] & pw["enemy"]
                       & (pw["gap"] <= pursue_gap))
        chase_touch = chase_touch | chase_touch.transpose(1, 2)
        touch = touch | chase_touch
    # Stuck men (contact.rout_stuck; config/nn/sim.json contact.rout_stuck_why; absent / off: none): a router whose men ran
    # into a standing enemy formation leaves stk_n of them among its men (below, after the melee) - they stay in contact
    # with it wherever the rest of the router is: it strikes them (melee.py), the router counts as in melee, the striker's
    # own flag is not lit by them (the game's bystanders: off).
    stuck_cal = cal["contact"].get("rout_stuck") or {}
    stuck_touch = torch.zeros_like(touch)
    if stuck_cal.get("on") and "stk_n" in u:
        k_of = u["stk_k"].clamp(min=0)
        k_ok = alive.gather(1, k_of) & ~u["r"].gather(1, k_of)
        has = alive & u["r"] & (u["stk_n"] > 0) & (u["stk_k"] >= 0) & k_ok
        u["stk_n"] = torch.where(has, u["stk_n"], torch.zeros_like(u["stk_n"]))
        among = has[:, None, :] & (u["stk_k"][:, None, :] == torch.arange(N, device=st.device)[None, :, None])
        stuck_touch = (among | among.transpose(1, 2)) & pw["enemy"]
        touch = touch | stuck_touch
    # Leaving melee: a withdraw order, or any unit told to move contact.leave_m or more away (0: off).
    # It walks out of the fight: it strikes nobody and is not held in place, the enemies in contact
    # still strike it (measured in the game, config/nn/sim.json contact.why).
    leave_m = float(cal["contact"].get("leave_m", 0.0))
    point_d = torch.sqrt((u["ox"] - u["x"]) ** 2 + (u["oz"] - u["z"]) ** 2)
    far_point = point_d >= leave_m
    if cal["contact"].get("leave_away_only"):
        # A move order through the enemy it touches is not a leave (contact.leave_away_only; the probe build/probes7
        # P3: spearmen ordered to walk to a point 60 m beyond the clanrat spearmen attacking them, as CA's planner
        # does, fought on - 213 / 317 / 521 HP dealt at 0-5 / 5-15 / 15-30 s, the simulator's leaver none): it leaves
        # only when its point lies away from every enemy it touches (the way to the point at more than 90 deg from
        # the way to that enemy's centre).
        mdx, mdz = u["ox"] - u["x"], u["oz"] - u["z"]
        dx, dz = u["x"][:, None, :] - u["x"][:, :, None], u["z"][:, None, :] - u["z"][:, :, None]
        toward = (mdx[:, :, None] * dx + mdz[:, :, None] * dz) > 0
        # A point away from every enemy it touches is a leave from contact.leave_away_m on (the fresh-order probe,
        # build/charge-probe/runs/20261008-145559 / -145641: swordsmen in melee told to walk 5 m back turned about
        # and struck nothing for 21-25 s, 2 lanes of 2; config/nn/sim.json contact.leave_away_why); absent: leave_m.
        away_m = float(cal["contact"].get("leave_away_m", leave_m))
        far_point = ~(touch & toward).any(2) & (point_d >= away_m)
        if cal["contact"].get("missile_leave_any"):
            # A formation with a missile weapon leaves on a move leave_m or more away whatever its direction, through
            # the enemy too, given before the contact or in it (contact.missile_leave_any; config/nn/sim.json
            # contact.missile_leave_why: the 72 gate battles, the game AI's shooters in melee under a run move
            # 10-1000 m away moved 0.92 of the seconds at 2.3 m/s with the point through the nearest enemy, 0.96 at
            # 3.0 m/s with it away; P3 above - spearmen that fought on - is a melee unit's)
            far_point = far_point | ((u["range"] > 0) & (u["men0"] > 1) & (point_d >= leave_m))
        if cal["contact"].get("move_melee_leave") and "order_s" in u:
            # A move order given in melee is a leave in any direction, through the enemy too (contact.move_melee_leave;
            # the meleeorders probe, build/charge-probe/runs/20261008-162358 / -162441: swordsmen in melee told to run
            # 5 / 15 / 40 m ahead through the clanrats or 5 / 15 m aside struck 0-17 HP/s in 0-5 s and none at 5-20 s,
            # moving all the time, 7 lanes of 7, then fought again at 20-30 s - the 24 s window). One given before the
            # contact keeps pushing and fighting (P3 above; R3: walking into the target 0.70 of attacking): order_s
            # starts at the window for an order given out of melee.
            in_melee = u["order_s"] < float(R.get("melee_breakoff_secs", 0.0))
            far_point = far_point | (in_melee & (point_d >= away_m))
    leaving = (kind == O.WITHDRAW) | ((kind == O.MOVE) & far_point & (leave_m > 0))
    # A move given before the contact (contact.pre_order_free; config/nn/sim.json contact.shootcontact_why): a
    # formation with a missile weapon that an enemy catches while it already carries out a move or withdraw - one given
    # out of melee (its order clock order_s started at the window, melee_breakoff_secs) or over the window before its
    # exit began - is not held (pin_s) and the 24 s window (breakoff) does not drop its order: it goes on to its point
    # (the shootcontact probe: archers and crossbowmen walking / running 300 m away from clanrats that caught them kept
    # moving all 75 s, 4 lanes of 4); at its point the move is over (no leave_latch) and it fights. A move given in
    # melee keeps both (the wavemiss2 probe: such a shooter strikes back 25-35 s after the order).
    # pre_move: such a move of such a formation (contact.leave_run_missile below uses it too).
    pre_move = torch.zeros_like(leaving)
    if "order_s" in u and "exit_s" in u:
        win = float(R.get("melee_breakoff_secs", 0.0))
        pre_move = ((u["range"] > 0) & (u["men0"] > 1) & ((kind == O.MOVE) | (kind == O.WITHDRAW))
                    & ((u["order_s"] - u["exit_s"]) >= win - 0.5 * dt))
    pre_order = pre_move if cal["contact"].get("pre_order_free") else torch.zeros_like(leaving)
    if cal["contact"].get("leave_latch") and "exit_s" in u:
        # A move that began as a leave in contact stays one while its order stands (contact.leave_latch): arrived at
        # a near point, the unit still strikes nobody until the 24 s window drops the order (the same probe: the
        # 5 m leavers stood chased at their point, moving, 0 HP dealt until 21-25 s, then fought on).
        leaving = leaving | ((kind == O.MOVE) & (u["exit_s"] > 0) & ~new_order & ~pre_order)
    if cal["contact"].get("attack_leave"):
        # An attack order on an enemy it does not touch, its edge contact.leave_m or more away, given to a formation
        # without a missile weapon that touches a standing enemy, is a leave too (contact.attack_leave;
        # config/nn/sim.json contact.attack_leave_why): it walks off towards its target, strikes nobody, is held
        # pin_melee_s and chased like a withdraw; the 24 s window (breakoff) drops the order if it is still in contact
        # then. Any direction: through the enemy or away. A lone man (a lord) too (contact.attack_leave_lord; config/nn/
        # sim.json contact.attack_leave_lord_why: the network's lord in melee told to attack an enemy 15 m or more away
        # walks off and barely strikes in the game) - not held (pin: formations only), leave_taken the formations'.
        ti_a = tgt.clamp(min=0)[:, :, None]
        tgt_away = ~touch.gather(2, ti_a).squeeze(2) & (pw["gap"].gather(2, ti_a).squeeze(2) >= leave_m)
        fighting = (touch & standing[:, None, :]).any(2)
        who = (u["men0"] > 1) | bool(cal["contact"].get("attack_leave_lord"))
        leaving = leaving | ((kind == O.ATTACK) & (tgt >= 0) & tgt_away & fighting & (u["range"] <= 0)
                             & who & (leave_m > 0))
    # --- the skirmish mode (skirmish.py; config/nn/sim.json skirmish): a shooter with the mode on runs, over its order
    # and without shooting, while an enemy moving at it is within its gap; caught, it goes on running - a leave ---
    sk_run = torch.zeros_like(standing)
    if sk_cal.get("on") and "skr" in u:
        in_touch = standing & (touch & standing[:, None, :]).any(2)
        sk_run, sk_x, sk_z = skirmish.update(u, pw, standing, in_touch, st.bounds, sk_cal, spacing, st.t, dt,
                                             new_order=sk_new if sk_cal.get("order_cuts", 1) else None)
        if sk_cal.get("melee_run"):
            leaving = leaving | (sk_run & in_touch)
    if cal["contact"].get("engage") and cal["contact"].get("engage_release"):
        # The engage hold is for units that stay fighting (contact.engage_release; config/nn/sim.json
        # contact.engage_release_why): a pair where either unit leaves (a withdraw, a far move, an attack_leave, a
        # skirmish run) or has an attack order on another unit touches only by the old hold (reach_m + hold_m). The
        # chase and the rout's stuck men keep their own contacts.
        ar = torch.arange(N, device=tgt.device)
        other = (kind == O.ATTACK)[:, :, None] & (tgt >= 0)[:, :, None] & (tgt[:, :, None] != ar[None, None, :])
        rel = leaving[:, :, None] | other
        rel = rel | rel.transpose(1, 2)
        lim_old = geometry.contact_limit(u, pw, cal["contact"], R, spacing, engage=False)
        drop = rel & (gap_c > lim_old) & ~chase_touch & ~stuck_touch
        touch = touch & ~drop
        touch_geo = touch_geo & ~drop
    striker = standing & ~leaving
    if cal["contact"].get("hold_breakoff") and "brk_s" in u:
        # A hold does not close the melee exit's window (contact.hold_breakoff; config/nn/sim.json
        # contact.hold_breakoff_why): a formation without a missile weapon given in melee a leave (a withdraw, a far
        # move, an attack_leave) or an attack on an enemy it does not touch, then a hold while still in melee, strikes
        # nobody until the database's melee_breakoff_secs (24 s) from that first order are over, then fights as any
        # held unit (the holdmelee probe, build/charge-probe/runs/20261011-090746 and -090858: swordsmen halted 3 / 10 s
        # after a run 15 m back or an attack on clanrats 4 m aside dealt 0 HP for 10-25 s, 6 lanes of 6, back from
        # 25-26 s after the first order; a plain halt fought on). brk_s: seconds left of that window. Another order
        # (not a hold) ends it; the other rules (fresh_incidental, leave_latch, breakoff) go on as they were.
        # A lone man (a lord) too (contact.hold_breakoff_lord; config/nn/sim.json contact.hold_breakoff_lord_why: the
        # holdlord probe - the General told in melee to attack clanrats 25 m aside, halted 3 s later, struck the
        # clanrats he touched nothing for 15 s after the halt and fought again 22.5 s / over 40 s after the attack order).
        win = float(R.get("melee_breakoff_secs", 0.0))
        form = ((u["men0"] > 1) | bool(cal["contact"].get("hold_breakoff_lord"))) & (u["range"] <= 0)
        ti_b = tgt.clamp(min=0)[:, :, None]
        att_other = (kind == O.ATTACK) & (tgt >= 0) & ~touch.gather(2, ti_b).squeeze(2)
        begin = new_order & u["m"] & standing & form & (leaving | att_other)
        hold_on = new_order & take & (orders.kind == O.HOLD) & u["m"]      # (an order, not a dead target's hold)
        u["brk_s"] = torch.where(begin, torch.full_like(u["brk_s"], win),
                                 torch.where(new_order & ~hold_on, torch.zeros_like(u["brk_s"]),
                                             (u["brk_s"] - dt).clamp(min=0)))
        striker = striker & ~((kind == O.HOLD) & (u["brk_s"] > 0) & form)
    strike = touch & striker[:, :, None]
    engaged = standing & (touch & standing[:, None, :]).any(2)
    # Pinned: a missile unit leaving melee stays held where it is (not striking, struck) until it has been leaving in
    # contact for contact.pin_s seconds; a unit without a missile weapon for contact.pin_melee_s (absent: 0, not held:
    # it walks out unless it is chased, below); a lone man (a lord) is not held (config/nn/sim.json contact.pin_why;
    # pin_s is 0 there: the game's shooters move from the order on).
    pin_s = knobs.get(u, "pin_s", float(cal["contact"].get("pin_s", 0.0)))
    pin_melee_s = knobs.get(u, "pin_melee_s", float(cal["contact"].get("pin_melee_s", 0.0)))
    in_exit = leaving & engaged & (u["men0"] > 1)
    hold_for = torch.where(u["range"] > 0, knobs.full_like(u["leave_s"], pin_s),
                           knobs.full_like(u["leave_s"], pin_melee_s))
    stuck = in_exit & (hold_for > 0) & ~pre_order
    pinned = stuck & (u["leave_s"] < hold_for)
    u["leave_s"] = torch.where(stuck, u["leave_s"] + dt, torch.zeros_like(u["leave_s"]))
    # The melee exit's window (the database's melee_breakoff_secs, 24 s; contact.breakoff on): a unit still touching
    # an enemy that long after it began to leave melee drops its order and fights again (the melee probe,
    # build/movelords: chased swordsmen and spearmen struck nothing for 24 s after a withdraw order, then fought on).
    # exit_s: seconds since it began to leave in contact, out of contact too, while it keeps leaving.
    # A lone man (a lord) too (contact.breakoff_lord; the leave probe: the Empire General leaving clanrats that chase
    # him stood and fought 24-25 s after the order; config/nn/sim.json contact.breakoff_lord_why) - he is still never
    # held (pin) and leave_taken stays the formations'.
    exit_now = in_exit
    if cal["contact"].get("breakoff_lord"):
        exit_now = leaving & engaged
    if "exit_s" in u:
        u["exit_s"] = torch.where(leaving & (exit_now | (u["exit_s"] > 0)), u["exit_s"] + dt, torch.zeros_like(u["exit_s"]))
    # (the order is dropped at the end of this step: it fights from the next one)
    brk = float(R.get("melee_breakoff_secs", 0.0)) if cal["contact"].get("breakoff") and "exit_s" in u else 0.0
    broke_off = (exit_now & (u["exit_s"] >= brk) & ~pre_order) if brk > 0 else None
    # The chase (contact.chase; build/open_battle/spec.md 2, build/open_melee/spec.md R4): a unit with an attack order
    # on an enemy that is leaving melee in contact with it is not held in the fight: it follows the leaver at its own
    # speed (closing up whenever the formations overlap by less than contact.reach_m) and strikes it while they touch -
    # so a chased unit stays in contact and the 24 s window drops its order, an unchased one walks out (the game's
    # probe: out of contact ~4 s after a withdraw order without a chaser; with one the chasers ran 10-15 m behind,
    # centre to centre, and kept striking until the window).
    chasing = torch.zeros_like(leaving)
    if cal["contact"].get("chase"):
        ti0 = tgt.clamp(min=0)
        chasing = (engaged & (kind == O.ATTACK) & (tgt >= 0) & leaving.gather(1, ti0)
                   & touch.gather(2, ti0[:, :, None]).squeeze(2))
    speed = torch.sqrt(u["vx"] ** 2 + u["vz"] ** 2)
    # --- innate effects (attributes, passives, game-fired timed passives: config/nn/effects.json): their
    # stats and rule flags hold for this step ---
    innate = effects.apply(u, params, dt, standing, engaged, pw["dist"], same_side)
    # The melee clock: seconds since the contact began; it goes on through gaps out of contact shorter than
    # contact.reset_s (the game: pull-outs of 4-7 s do not restart a fight, build/cyclecharge).
    reset_s = float(cal["contact"].get("reset_s", 0.0))
    fresh = engaged & (u["contact_s"] <= 0)              # its fight starts this step
    # The opening once a battle (contact.opening_once; config/nn/sim.json contact.opening_once_why): a formation's
    # opening - the wave (melee.py, its clock melee_s: seconds in melee over the whole battle, never reset) and the
    # first strike below - comes at its first contact of the battle only; back in melee after any time out it fights at
    # the steady rate from the first second (the recontact probe: swordsmen charging clanrats again after 2-80 s out
    # struck 0.24-1.35 of their 10-20 s rate in the first 3 s, their first contact 2.87).
    opened = torch.zeros_like(fresh)
    if "melee_s" in u:
        if cal["contact"].get("opening_once"):
            opened = (u["melee_s"] > 0) & (u["men0"] > 1)
        u["melee_s"] = torch.where(engaged, u["melee_s"] + dt, u["melee_s"])
    u["out_s"] = torch.where(engaged, torch.zeros_like(u["out_s"]), u["out_s"] + dt)
    u["contact_s"] = torch.where(engaged, u["contact_s"] + dt,
                                 torch.where(u["out_s"] < reset_s, u["contact_s"], torch.zeros_like(u["contact_s"])))
    # --- the charge (CA Feature Focus #2, build/charge/spec.md): an attack order and a run-up - charge.min_runup_m
    # run at charge.min_speed_share of the run speed or faster - give the full charge bonus at the contact (an
    # attack order given within charge.order_grace_s after the contact still counts); it fades on its own clock,
    # linear to 0 over charge_decay_duration from the first blow, in contact or out of it; a new charge restarts it.
    ccal = cal["charge"]
    decay = R["charge_decay_duration"]
    u["charge"] = (u["charge"] - dt / decay).clamp(min=0)
    # The attack target is already fighting another of our units (charge.free_target_only; build/open_battle/spec.md
    # 5): the way in is blocked by our own men - no charge sprint and no charge (CA: no charge when the path is blocked
    # by one's own unit; the recordings: 2 s before contact 1.09-1.12 of the run speed into a free target, 0.79-0.86
    # into one already in melee).
    busy_t = torch.zeros_like(standing)
    if ccal.get("free_target_only"):
        rows = touch.gather(1, tgt.clamp(min=0)[:, :, None].expand(-1, -1, N))        # [B, i, k]: i's target touches k
        block = rows & same_side & ~eye & standing[:, None, :]
        if ccal.get("blocked_side"):
            # (charge.blocked_side, the busycharge probe) only our unit fighting the target on the side the charger
            # comes from blocks the way (front, right, rear, left of the target: contact.front_deg / rear_deg); into
            # its flank or back the charge is whole - the game's swordsmen into clanrats busy with our spearmen: rear
            # 402 / 594 HP in 0-2 / 0-5 s (free rear 418 / 597), sprint 3.70 m/s in the last 10 m (free 3.71)
            side4 = geometry.side4(pw["rel_j"], cal["contact"]["front_deg"], cal["contact"]["rear_deg"])   # [B, x, j]
            ti = tgt.clamp(min=0)
            s_i = side4.gather(2, ti[:, :, None]).squeeze(2)                               # i's side of its target
            s_k = side4.transpose(1, 2).gather(1, ti[:, :, None].expand(-1, -1, N))       # k's side of i's target
            block = block & (s_k == s_i[:, :, None])
        busy_t = (tgt >= 0) & block.any(2)
    min_runup = knobs.get(u, "min_runup_m", float(ccal["min_runup_m"]))
    fire = engaged & (kind == O.ATTACK) & (u["runup"] >= min_runup) & ~busy_t
    u["charge"] = torch.where(fire, torch.ones_like(u["charge"]), u["charge"])
    charge_now = u["charge"]
    fast = (speed >= float(ccal["min_speed_share"]) * u["run"]) & (speed > 0.1)
    speed_to = speed
    if ccal.get("runup_towards"):
        # (build/open_melee/spec.md R6) the run-up counts only the speed towards the attack target, else towards the
        # nearest enemy - not running away from it (the game: a Warlord that ran 10 m off came back at 1.5 m/s and
        # struck one ordinary blow)
        dd = torch.where(pw["enemy"] & both, pw["dist"], torch.full_like(pw["dist"], 1e9))
        toward = torch.where(tgt >= 0, tgt, dd.argmin(2))
        dxt = u["x"].gather(1, toward) - u["x"]
        dzt = u["z"].gather(1, toward) - u["z"]
        v_to = (u["vx"] * dxt + u["vz"] * dzt) / torch.sqrt(dxt ** 2 + dzt ** 2).clamp(min=1e-6)
        v_to = torch.where((tgt >= 0) | (dd.min(2).values < 1e9), v_to, torch.zeros_like(v_to))
        fast = (v_to >= float(ccal["min_speed_share"]) * u["run"]) & (v_to > 0.1)
        speed_to = v_to.clamp(min=0)
    grace = engaged & (u["contact_s"] < float(ccal["order_grace_s"])) & ~fire
    u["runup"] = torch.where(engaged, torch.where(grace, u["runup"], torch.zeros_like(u["runup"])),
                             torch.where(fast, u["runup"] + speed_to * dt, torch.zeros_like(u["runup"])))
    # The first strike (the game: a man's attack interval starts only after his blow, CA Feature Focus #2): a unit
    # that comes into a fight moving - or whose charge lands - strikes once at once with every man in contact (the
    # charge's full bonus with it), then at the steady rate; a standing unit under HOLD that it reaches strikes once
    # at once too with its men facing it (contact.stand_first_strike, below). ran_in keeps that it came in moving for
    # the rest of the fight (melee.py: men gather round a lord only when he ran in).
    # (an attack order given within order_grace_s of a contact it ran into charges then: no second first strike)
    arrive = fresh & (speed > 0.3)
    first = (arrive | (fire & ~(u["ran_in"] & (u["contact_s"] <= float(ccal["order_grace_s"]) + dt)))) & ~opened
    u["ran_in"] = (u["ran_in"] | arrive | fire) & (u["contact_s"] > 0)

    # --- lord abilities cast by the game's AI by its rule or by the network's order: their effects hold
    # for this step ---
    base = abilities.apply(u, params, dt, standing, engaged, pw["dist"], same_side, orders.ability)
    vigour = abilities.vigour(u, params, standing)
    # --- fatigue's stat multipliers (the database's unit_fatigue_effects_tables), for this step ---
    tired = fatigue.effects(u, params)

    # --- melee ---
    rate, mhit, sector, F_men, swing = melee.strikes(u, pw, strike, params, charge_now, first=True, ran=first & fire)
    if stuck_cal.get("on") and "stk_n" in u:
        # A router running into a standing enemy formation (its boxes touching by their own edges - not the rout's exit
        # contact - with the router moving towards it, not a formation chasing it by order) leaves among its men as many
        # of its own as that formation's men strike it now (F_men); the most of them, one formation per router.
        raw = pw["enemy"] & both & (pw["gap"] <= lim)
        dxk, dzk = u["x"][:, :, None] - u["x"][:, None, :], u["z"][:, :, None] - u["z"][:, None, :]   # [k, j]: j to k
        into = (u["vx"][:, None, :] * dxk + u["vz"][:, None, :] * dzk) > 0
        by_order = (kind == O.ATTACK)[:, :, None] & (tgt[:, :, None] == torch.arange(N, device=st.device)[None, None, :])
        new = (raw & into & ~by_order & standing[:, :, None] & (alive & u["r"])[:, None, :]
               & (u["men0"] > 1)[:, :, None] & (u["men0"] > 1)[:, None, :])
        cand = torch.where(new, F_men, torch.zeros_like(F_men)).clamp(max=u["men"][:, None, :])
        best, k_best = cand.max(1)
        take_k = best > u["stk_n"]
        u["stk_n"] = torch.where(take_k, best, u["stk_n"])
        u["stk_k"] = torch.where(take_k, k_best, u["stk_k"])
    first_pair = first[:, :, None]
    if cal["contact"].get("stand_first_strike"):
        # (build/open_melee/spec.md R1) a unit standing under HOLD that an enemy reaches strikes once at once too, with
        # every man in contact facing it (within contact.front_deg of its front): its men's intervals have run out (CA
        # Feature Focus #2: the interval runs after a blow); the probe: a charger lost 106-194 HP in the first second
        # to braced spearmen, 120 to swordsmen, 0 to a target facing away
        stand = fresh & (speed <= 0.3) & (kind == O.HOLD) & ~opened
        deg = float(cal["contact"]["front_deg"])
        if cal["contact"].get("stand_first_angle") == "bracing":
            # (contact.stand_first_angle "bracing") within the database's bracing_attack_angle (80 deg) of its front, not
            # front_deg: the brace probe - clanrats charging held spearmen at 60 deg lost 83 / 127 HP in their first
            # second (in front 166, at 100 deg 0 / 0); contact.stand_first_why
            deg = float(params.battle["bracing_attack_angle"])
        facing = pw["rel_i"].abs() <= deg * geometry.DEG
        first_pair = first_pair | (stand[:, :, None] & facing)
    hp_melee = rate * dt + torch.where(first_pair, swing, torch.zeros_like(swing))
    # A unit leaving melee (held or walking out) still in contact takes contact.leave_taken of the blows: melee
    # units more (they turn their backs, config/nn/sim.json contact.pin_why), missile units in full (the shootcontact
    # probe, contact.shootcontact_why).
    taken_cal = cal["contact"].get("leave_taken")
    if taken_cal:
        out = leaving & engaged & (u["men0"] > 1)
        taken_melee = knobs.get(u, "leave_taken_melee", float(taken_cal["melee"]))
        mult = torch.where(u["range"] > 0, float(taken_cal["missile"]), taken_melee)
        pursuit = cal["contact"].get("pursuit_rate")
        if cal["contact"].get("leave_run_pursuit") and pursuit is not None:
            # A leaver without a missile weapon (a lord too) on the move, not held, is struck as a router is: at
            # contact.pursuit_rate of the rule (contact.leave_run_pursuit; the leave probe: chased swordsmen and shield
            # spearmen running out lost 102 HP at 5-10 s against the twin's 213 with x1.25, 0.25-0.40 of it; standing
            # at their point, their backs to the chasers, 43-93 HP/s - that keeps leave_taken). On the move: its
            # speed last step over 0.3 m/s (the observed moving flag's threshold). Off for now (contact.leave_run_why:
            # the game makes it up at the stopping point, which the chase here lacks).
            run_out = leaving & engaged & (u["range"] <= 0) & ~pinned & (speed > 0.3)
            out = out | run_out
            mult = torch.where(run_out, knobs.full_like(mult, knobs.get(u, "pursuit_rate", float(pursuit))), mult)
        if cal["contact"].get("leave_run_missile") and pursuit is not None:
            # A formation with a missile weapon running on a move given before the contact (pre_move above; not held,
            # its speed last step over its walk + 0.3 m/s: the simulator's running flag) is struck at
            # contact.pursuit_rate of the rule (contact.leave_run_missile; the shootcontact probe: such a shooter loses
            # 0.41 [0.16-0.63] of a standing one's HP a second in contact, 4 lanes; config/nn/sim.json
            # contact.shootcontact_why); at a walk it takes leave_taken.missile. A run given in melee keeps
            # leave_taken.missile (the wavemiss2 probe: such shooters lost ~0.8 %/s running, as standing ones).
            run_sh = out & pre_move & ~pinned & (speed > u["walk"] + 0.3)
            mult = torch.where(run_sh, knobs.full_like(mult, knobs.get(u, "pursuit_rate", float(pursuit))), mult)
        hp_melee = torch.where(out[:, None, :], hp_melee * mult[:, None, :], hp_melee)

    # --- shooting ---
    # (fire whilst moving, attribute mounted_fire_move: shoots and aims on the move too)
    still = (speed < 0.2) | u["fire_move"]
    # (a unit that shoots in melee too - missile.fire_in_melee, the Ratling Gun: the suppress probe's gun fired on at the
    # swordsmen in contact with it)
    free = ~engaged | u["melee_fire"] if "melee_fire" in u else ~engaged
    ready = standing & free & (u["a"] > 0) & (u["range"] > 0) & still & (~sk_run | u["fire_move"])
    # On the move (fire whilst moving) a unit shoots only at targets within missile.move_fire_arc_deg of its
    # facing (the way it walks): walking away it holds fire (measured, config/nn/sim.json missile.move_fire_why) -
    # unless its own fire arc (the database's, passport missile.fire_arc_deg) is wider: the Night Runners' throwing
    # stars (360) shoot behind on the move (missile.move_fire_own_arc; the probe starsmove: volleys at 178 deg off
    # the facing while walking away, at 82-151 deg walking and running across).
    arc = float(cal["missile"].get("move_fire_arc_deg", 180.0))
    on_move = u["fire_move"] & (speed >= 0.2)
    behind = None
    if arc < 180:
        lim = torch.full_like(u["arc"], arc)
        if cal["missile"].get("move_fire_own_arc"):
            lim = torch.maximum(lim, u["arc"])
        behind = on_move[:, :, None] & (pw["rel_i"].abs() > (lim * geometry.DEG)[:, :, None])
    # Standing, a unit keeps its facing while its target is within missile.stand_fire_arc_deg of it (measured,
    # missile.stand_fire_why) and turns to a target beyond it first (the facing below, at turn.*_deg_s) until the
    # target is within missile.turn_done_deg (then it keeps that facing again); it aims only once the turn is done.
    # Without an ordered target it keeps its target while that stays in range and stands (missile.sticky_target),
    # else takes the nearest within stand_fire_arc_deg, else turns to the nearest beyond it. A target that routs is
    # kept by the measured share of the game's free shooters (missile.router_keep_s / _share, missile.router_keep).
    ms_cal = cal["missile"]
    prev = u["aim_tgt"] if ms_cal.get("sticky_target") else None
    per_man = missile.per_man_mask(u, params)
    reach = missile.reach_of(params)
    keep_r = missile.router_keep(u, prev, params, st.t, dt)
    aim_at = missile.choose_target(u, pw, ready, tgt, kind == O.ATTACK, exclude=behind, prev=prev, per_man=per_man,
                                   reach=reach, keep_router=keep_r)
    s_arc = float(ms_cal.get("stand_fire_arc_deg", 180.0))
    turning = torch.zeros_like(ready)
    done_deg = ms_cal.get("turn_done_deg")
    if s_arc < 180:
        on_stand = ready & (speed < 0.2)
        out = on_stand[:, :, None] & (pw["rel_i"].abs() > s_arc * geometry.DEG)
        ahead = missile.choose_target(u, pw, ready, tgt, kind == O.ATTACK,
                                      exclude=out if behind is None else (out | behind), prev=prev, per_man=per_man,
                                      reach=reach, keep_router=keep_r)
        ordered = (kind == O.ATTACK) & (aim_at == tgt) & (aim_at >= 0)
        aim_at = torch.where(ordered | (ahead < 0), aim_at, ahead)
        beyond = (aim_at >= 0) & out.gather(2, aim_at.clamp(min=0)[:, :, None]).squeeze(2)
        if ms_cal.get("turn_ordered_only"):
            # firing at will a unit never turns (the probe: a target 50 deg off, no turn, 6 % of the men fired);
            # it turns only to the target of an attack order
            beyond = beyond & (kind == O.ATTACK) & (aim_at == tgt)
        if done_deg is not None:
            off = pw["rel_i"].gather(2, aim_at.clamp(min=0)[:, :, None]).squeeze(2).abs()
            # (a skirmish run's end, skirmish.py `skt`: the unit turns back to its target once it stands, whatever its order)
            back_on = u["turn_on"] | u["skt"] if "skt" in u else u["turn_on"]
            go_on = back_on & on_stand & (aim_at >= 0) & (off > float(done_deg) * geometry.DEG)
            turning = beyond | go_on
            u["turn_on"] = turning
            if "skt" in u:
                u["skt"] = u["skt"] & ~on_stand
        else:
            turning = beyond
    # A new target (another enemy than last step's, an order's or its own) costs missile.retarget_s without fire
    # (measured, missile.retarget_why): the aim clock goes back to retarget_s before aim_s.
    # Taking a target from none is a new target too (missile.retarget_from_none; the probe, build/open_missile S4: a
    # standing unit's first projectile came 1.5-2.5 s after a running target entered its range, not at once).
    retarget_s = ms_cal.get("retarget_s")
    new_tgt = (aim_at >= 0) & (aim_at != u["aim_tgt"])
    if retarget_s:
        from_none = bool(ms_cal.get("retarget_from_none"))
        switched = ready & new_tgt & ((u["aim_tgt"] >= 0) | from_none)
        # (a unit's own cost, missile.retarget_cat_s, for a switch from a target it had; taking one from none keeps
        # retarget_s: the handgunners' first shot from the go 2.5-4 s, as before)
        own_rt = (u["rt_s"] >= 0) & (u["aim_tgt"] >= 0)
        cost = torch.where(own_rt, u["rt_s"], torch.full_like(u["rt_s"], float(retarget_s)))
        u["aim"] = torch.where(switched, torch.minimum(u["aim"], u["aim_s"] - cost), u["aim"])
    # the quiet window (missile.order_quiet_cat_s): a new order or a new target to a unit already aiming starts it again
    hush = (u["quiet_s"] > 0) & (u["aim_tgt"] >= 0) & (changed | (ready & new_tgt))
    u["quiet"] = torch.where(hush, u["quiet_s"], u["quiet"])
    reaim = bool(ms_cal.get("reaim"))
    if reaim:
        # a new target: every man has a living target man again
        u["late"] = torch.where(new_tgt | (aim_at < 0), torch.zeros_like(u["late"]), u["late"])
    u["aim_tgt"] = torch.where(ready, aim_at, torch.full_like(aim_at, -1))
    u["aim"] = torch.where(ready & ~turning, u["aim"] + dt, torch.zeros_like(u["aim"]))
    quiet = u["quiet"] > 1e-6
    u["quiet"] = (u["quiet"] - dt).clamp(min=0)
    can = ready & ~turning & (u["aim"] >= u["aim_s"]) & ~quiet
    m_target = torch.where(can, aim_at, torch.full_like(aim_at, -1))
    # direct fire needs a clear line past friends (missile.py clear_shot); arcing fire: unchanged
    m_target, clear, catch = missile.clear_shot(u, pw, m_target, can, tgt, kind == O.ATTACK, params, with_catch=True)
    # a standing unit's men fire only at a target centre within their fire arc (missile.arc_share; the database's
    # battle_entities fire arc): out-of-arc men lose the volley like blocked ones (they reload too)
    if ms_cal.get("per_man_arc"):
        clear = torch.where(speed < 0.2, clear * missile.arc_share(u, pw, m_target), clear)
    if behind is not None:
        back = behind.gather(2, m_target.clamp(min=0)[:, :, None]).squeeze(2) & (m_target >= 0)
        m_target = torch.where(back, torch.full_like(m_target, -1), m_target)
    # the men reload all the time (moving too); the unit shoots when it can and enough of its men are loaded
    # (missile.py: whole-unit volleys one reload apart). When the unit fires, every loaded man
    # starts reloading, those whose line is blocked too (clear_shot): else the blocked men stayed
    # loaded and fired on the next steps, and a screen that blocks half the men blocked nothing.
    unready = (u["unready"] - dt / u["reload"].clamp(min=1e-6)).clamp(min=0)
    loaded = 1 - unready
    # missile.volley_load: the unit shoots only once this share of its men is loaded (1: whole-unit volleys one
    # reload apart, as the game; 0: every loaded man at once, the volley then a steady trickle).
    load_min = float(cal["missile"].get("volley_load", 0.0))
    full = loaded >= load_min - 1e-3
    if reaim:
        # Re-aim (missile.reaim; build/open_missile spec S3): a man keeps his target man; when that man has died since
        # his last shot he aims again (aim_s) before the next one. The unit's volley waits the mean of it, aim_s x
        # the share of its men whose target died (late): its men's interval is reload + aim_s x that share (a lone
        # lord, who does not die man by man: the reload alone).
        # (the wait rounded to the step: the volley comes in the step nearest to aim_s x late after the men loaded)
        # A broken target (missile.reaim_broken_cat_share; arrows and bolts): once the unit's target has lost
        # broken_at of its starting men, every man aims again (aim_s) after each reload - the volley splits and the
        # men's cycle is reload + aim_s (a lone man: never).
        late = u["late"]
        if "broken_at" in u:
            ti = aim_at.clamp(min=0)
            men0_t = u["men0"].gather(1, ti)
            lost_t = 1 - u["men"].gather(1, ti) / men0_t.clamp(min=1e-6)
            broken = (u["broken_at"] > 0) & (aim_at >= 0) & (men0_t > 1) & (lost_t >= u["broken_at"] - 1e-6)
            late = torch.where(broken, torch.ones_like(late), late)
        waited = u["loaded_s"] >= u["aim_s"] * late - dt / 2 - 1e-6
        u["loaded_s"] = torch.where(full & (m_target >= 0), u["loaded_s"] + dt, torch.zeros_like(u["loaded_s"]))
        full = full & waited
    # (between volleys the unit is still shooting at its target: the game's IsFiringMissiles, the `fire` flag)
    volley_target = torch.where(full, m_target, torch.full_like(m_target, -1))
    share = loaded.clamp(max=1.0)
    burst = None
    if "bst_s" in u:
        # A burst (missile.py burst_share): a volley of spv shots a man (the Ratling Gun's 18) leaves at the measured
        # rate over spv / bst_rate s from its start; the reload runs from the start (the cycle start to start)
        burst, start, share, volley_target = missile.burst_share(u, volley_target, m_target, share, dt)
    shots, hp_missile, shit = missile.volley(u, pw, volley_target, dt, params, contact=touch_geo, clear=clear,
                                             loaded=share, catch=catch)
    hp_bomb = kills_bomb = None
    if magic.on(params) and "bm0_s" in u:
        # the bombardments landing this step (magic.py): the HP and the men each caster's strike takes from each unit,
        # their own kill rule; the HP counts with the missiles' below (casualties, the fight's balance, under fire: the
        # spells' projectiles are projectiles of the database)
        bomb_extra = {}
        hp_bomb, kills_bomb, hits_bomb = magic.bombard(u, params, dt, pw["front"], pw["depth"],
                                                       noise=bool((params.sim.get("noise") or {}).get("blows")),
                                                       extra=bomb_extra)
    fired = shots > 0 if burst is None else torch.where(burst, start, shots > 0)
    u["unready"] = torch.where(fired, torch.ones_like(unready), unready)
    if reaim:
        u["late"] = torch.where(shots > 0, torch.zeros_like(u["late"]), u["late"])
        u["loaded_s"] = torch.where(shots > 0, torch.zeros_like(u["loaded_s"]), u["loaded_s"])
    u["a"] = (u["a"] - shots).clamp(min=0)
    firing = (m_target >= 0) & ((shots > 0) | (load_min > 0))
    # The observed flag (u["fire"], the network's "firing" input; the game's CCO IsFiringMissiles) and the target shown
    # with it: with missile.fire_flag_aiming the flag is on from the moment a standing shooter has taken a target and
    # aims at it (not turning, its aim clock still running), as the game's - its flag came on a median 1.5 s after a stop
    # with a target in range and arc, 4.5 s before the first shot (ammo drop), and a new order hardly put it out
    # (missile.fire_flag_why). The shots, the aim clock and everything inside the step keep `firing`.
    if ms_cal.get("fire_flag_aiming"):
        aiming = ready & ~turning & (aim_at >= 0) & ((u["aim"] < u["aim_s"]) | quiet) & ~firing
        shown, shown_tgt = firing | aiming, torch.where(firing, m_target, aim_at)
    else:
        shown, shown_tgt = firing, m_target

    # --- damage, men, kills ---
    if (params.sim.get("noise") or {}).get("blows"):
        # noise.blows: a Poisson number of blows / hits of the pair's mean HP a blow (the game rolls each; the probes'
        # identical lanes differ as independent blows do, config/nn/sim.json noise.blows_why); off: the mean
        def rolled(hp, per):
            ok = (hp > 0) & (per > 0) & torch.isfinite(hp) & torch.isfinite(per)
            lam = torch.where(ok, hp / torch.where(ok, per, torch.ones_like(per)), torch.zeros_like(hp))
            return torch.where(ok, torch.poisson(lam) * per, hp)
        hp_melee = rolled(hp_melee, mhit)
        hp_missile = rolled(hp_missile, shit)
    dmg = hp_melee + hp_missile
    if hp_bomb is not None:
        dmg = dmg + hp_bomb
    hit = torch.where(hp_missile > 0, shit, mhit)
    taken = dmg.sum(1)
    scale = torch.where(taken > u["hp_abs"], u["hp_abs"] / taken.clamp(min=1e-9), torch.ones_like(taken))
    dmg = dmg * scale[:, None, :]
    taken = dmg.sum(1)
    # bookkeeping (no effect on the battle): the gold of enemy health each unit destroyed (dmg [B, striker, struck])
    u["gold_out"] = u["gold_out"] + (dmg * pw["enemy"] * (u["cost"] / u["hp0"].clamp(min=1e-6))[:, None, :]).sum(2)
    share = melee.kill_share(u["hp_man"][:, None, :], hit, cal["kills"]["exponent"])
    db = hp_bomb * scale[:, None, :] if hp_bomb is not None else torch.zeros_like(dmg)    # (the bombardments' own kills)
    if hp_bomb is not None and "bm_hit" in u:
        # bookkeeping (no effect on the battle): who spell projectiles struck (the network's spell_strike) and each
        # caster's gold of enemy and own health and the enemy men knocked down so far (the spell price's bricks)
        per_hp = (u["cost"] / u["hp0"].clamp(min=1e-6))[:, None, :]
        friend = same_side & ~eye & present[:, None, :]
        u["bm_hit"] = hits_bomb.sum(1) > 0
        u["bm_gold_e"] = u["bm_gold_e"] + (db * pw["enemy"] * per_hp).sum(2)
        u["bm_gold_f"] = u["bm_gold_f"] + (db * friend * per_hp).sum(2)
        u["bm_knock"] = u["bm_knock"] + (bomb_extra["knock"] * pw["enemy"]).sum(2)
    kills = (dmg - db) / u["hp_man"][:, None, :].clamp(min=1e-6) * share
    if cal["kills"].get("wound_pool"):
        # The wounded pool (kills.wound_pool; build/open_melee/spec.md R7): the game keeps every man's health and the
        # blows beyond a man's health are lost (CA), so the wounded among the living - W = men x HP a man - the unit's
        # HP - stay at most W* = the unit's own men in contact x (HP a man - the HP-weighted hit of its strikers) (each
        # man struck is cut down in E[N] = HP a man / hit blows); the melee HP beyond the pool is whole men. Missiles
        # keep the exponent rule; a lone man (a lord) is one pool of health.
        hm = u["hp_man"].clamp(min=1e-6)
        dm = hp_melee * scale[:, None, :]
        D = dm.sum(1)
        hit_m = (dm * mhit).sum(1) / D.clamp(min=1e-9)
        own = F_men.sum(2)                                   # its own men striking in contact
        own = torch.where(own > 0, own, F_men.sum(1)).clamp(max=u["men"].clamp(min=0))   # (not striking back: the
        # men striking it)
        W = (u["men"] * u["hp_man"] - u["hp_abs"]).clamp(min=0)
        W_cap = own * (u["hp_man"] - hit_m).clamp(min=0)
        k_m = torch.minimum((W + D - W_cap).clamp(min=0), D) / hm
        pool_k = dm / D.clamp(min=1e-9)[:, None, :] * k_m[:, None, :]
        shot_k = (dmg - dm - db) / hm[:, None, :] * melee.kill_share(u["hp_man"][:, None, :], shit,
                                                                     cal["kills"]["exponent"])
        if cal["kills"].get("missile_uniform"):
            # kills.missile_uniform (build/open2): projectile hits spread evenly over the living men, each with his
            # own health (melee.shot_kills: Poisson hits a man, tau from the unit's wounded); per pair by its hits
            h_pair = (dmg - dm - db).clamp(min=0) / shit.clamp(min=1e-6)
            H = h_pair.sum(1)
            # the hits' mean HP (HP a man / their hits-weighted E[N] to kill)
            K_t = (h_pair * hm[:, None, :] / shit.clamp(min=1e-6)).sum(1) / H.clamp(min=1e-9)
            eff_t = torch.where(H > 0, hm / K_t.clamp(min=1e-6), hm)
            if cal["kills"].get("missile_front") and "fw" in u:
                # kills.missile_front: a flat shot strikes the front man of its file first (missile.front_share), so
                # the hits fall on the front rank (its files) and the rest evenly; a man takes at most his health -
                # the hits beyond the dead men's are lost (melee.front_kills)
                fs = missile.front_share(u, pw, params)
                h_f = (h_pair * fs).sum(1)
                files = (pw["front"] / u["sp_h"].clamp(min=0.1)).clamp(min=1)
                k_t, lost, fw_new = melee.front_kills(h_f, (H - h_f).clamp(min=0), u["men"], files, u["hp_man"],
                                                      u["hp_abs"], eff_t, u["fw"])
                many = u["men0"] > 1
                u["fw"] = torch.where(many & (H > 0), fw_new, u["fw"])
                # the lost hits' health back out of the taken (each pair by its hits)
                lost_hp = torch.where(many, lost * eff_t, torch.zeros_like(lost))
                cut = h_pair / H.clamp(min=1e-9)[:, None, :] * lost_hp[:, None, :]
                cut = torch.minimum(cut, (dmg - dm - db).clamp(min=0))
                dmg = dmg - cut
                taken = dmg.sum(1)
            else:
                k_t = melee.shot_kills(H, u["men"], u["hp_man"], u["hp_abs"], eff_t)
            shot_k = h_pair / H.clamp(min=1e-9)[:, None, :] * k_t[:, None, :]
        kills = torch.where((u["men0"] > 1)[:, None, :], pool_k + shot_k, kills)
    if kills_bomb is not None:
        kills = kills + torch.where((u["men0"] > 1)[:, None, :], kills_bomb * scale[:, None, :], torch.zeros_like(kills))
    hp_new = (u["hp_abs"] - taken).clamp(min=0)
    floor = torch.ceil(hp_new / u["hp_man"].clamp(min=1e-6) - 1e-6)
    men_new = torch.maximum(u["men"] - kills.sum(1), floor)
    men_new = torch.minimum(men_new, u["men"])
    men_new = torch.where(u["men0"] <= 1, (hp_new > 0).float(), men_new)
    men_new = torch.where(hp_new <= 0, torch.zeros_like(men_new), men_new)
    drop = u["men"] - men_new
    if reaim:
        # the shooter's men whose target man died this step: the share of its target's men lost (a lone man: none)
        lost_share = torch.where(u["men0"] > 1, drop / u["men"].clamp(min=1e-6), torch.zeros_like(drop)).clamp(0, 1)
        at = u["aim_tgt"]
        died = lost_share.gather(1, at.clamp(min=0))
        u["late"] = torch.where(at >= 0, 1 - (1 - u["late"]) * (1 - died), torch.zeros_like(u["late"]))
    if burst is not None and "bst_p" in u:
        # A bursting man whose target man died aims again (missile.burst_reaim): its flat fire aims at the target's
        # front rank, so a man lost there is the target of the target's files' share of the burst (files: its front /
        # its spacing across); those men lose aim_s of their burst (burst_share). The probe: the gun's bursts on
        # swordsmen fell from 576 to 364-482 bullets as their men died, the rate halving within 0.5 s of 3-4 deaths;
        # on the lone General (no man dies) every burst was 576
        at = u["aim_tgt"]
        ti = at.clamp(min=0)
        files = (pw["front"] / u["sp_h"].clamp(min=0.1)).clamp(min=1).gather(1, ti)
        lost = torch.where(u["men0"] > 1, drop, torch.zeros_like(drop)).gather(1, ti)
        front = (lost / files).clamp(0, 1)
        on = burst & (at >= 0) & (u["bst_s"] > 0) & bool((cal["missile"].get("burst_reaim") or 0))
        u["bst_p"] = torch.where(on, 1 - (1 - u["bst_p"]) * (1 - front), u["bst_p"])
    want = kills.sum(1)
    credit = kills * (drop / want.clamp(min=1e-9)).clamp(max=10)[:, None, :] * pw["enemy"]
    u["k"] = u["k"] + credit.sum(2)
    u["hp_abs"], u["men"] = hp_new, men_new
    if stuck_cal.get("on") and "stk_n" in u:
        # the stuck men are the ones dying (nobody else reaches them) and the rest work free over escape_s (measured)
        left = (u["stk_n"] - drop).clamp(min=0) * math.exp(-dt / float(stuck_cal["escape_s"]))
        u["stk_n"] = torch.where(left >= float(stuck_cal.get("min_men", 0.5)), torch.minimum(left, men_new),
                                 torch.zeros_like(left))
    u["hp"] = torch.where(present, hp_new / u["hp0"].clamp(min=1e-6), torch.zeros_like(hp_new))
    # seconds since its health first fell below a quarter (never back: the Wounds, effects.py)
    if "low_s" in u:
        low = present & ((u["hp"] < 0.25) | (u["low_s"] > 0))
        u["low_s"] = torch.where(low, u["low_s"] + dt, u["low_s"])
    # Casualty windows (recent casualties: the last 4 s, extended: the last 60 s; morale.casualty_windows);
    # recent_s: the melee balance (dealt / taken; reward.py reads the same).
    morale.casualty_windows(u, taken, params, dt)
    fade = math.exp(-dt / cal["morale"]["recent_s"])
    melee_scaled = hp_melee * scale[:, None, :]
    u["dealt"] = u["dealt"] * fade + melee_scaled.sum(2)
    u["taken"] = u["taken"] * fade + melee_scaled.sum(1)
    # the same balance of missile HP between enemies (the morale's winning / losing the fight counts both)
    shot_scaled = (hp_missile * scale[:, None, :] + db) * pw["enemy"]
    u["shot_dealt"] = u["shot_dealt"] * fade + shot_scaled.sum(2)
    u["shot_taken"] = u["shot_taken"] * fade + shot_scaled.sum(1)
    # contact effects (poison, the censer's contagion, suppressing fire): a hit of a weapon that lays one starts it on
    # the enemy hit (effects.lay; friendly fire lays none); it acts from the next step
    effects.lay(u, params, dt, melee_scaled, mhit, hp_missile * scale[:, None, :], shit, pw["enemy"].float(), st.t)
    # the slowing from hits of gunpowder (missile.hit_slow): a hit starts its timer on the enemy hit; it acts in this
    # step's movement (below)
    missile.hit_slow_lay(u, params, dt, hp_missile * scale[:, None, :], shit, pw["enemy"].float(), st.t)
    shot_at = ((hp_missile + db) * pw["enemy"]).sum(1) > 0     # friendly fire does not count (ume_concerned_under_friendly_fire 0)
    u["under_fire_s"] = torch.where(shot_at, torch.zeros_like(u["under_fire_s"]), u["under_fire_s"] + dt)
    attackers = strike & standing[:, :, None]
    # the worst side it is struck from this step (morale: attacked in the flank / rear while it lasts)
    u["flank_hit"] = torch.where(attackers, sector, torch.zeros_like(sector)).amax(1).float()
    if (cal["morale"].get("attacked_first") or {}).get("on") and "hit_sec" in u:
        # morale.attacked_first: the side is the one each attacker's contact BEGAN on (the database's was_attacked_in_flank
        # / _rear: 'first contact from the flank / rear'), held while that pair stays in contact (the recordings: the
        # first contact's side keeps the game's label as the struck unit turns to face its attacker; the morale probe
        # T-A: from the first blow to the attacker's leaving); u hit_sec [B, N * N] (state.PAIRS), side + 1, 0 none
        Bn, Nn = touch.shape[0], touch.shape[1]
        prev = u["hit_sec"].reshape(Bn, Nn, Nn)
        held = torch.where(prev > 0, prev, torch.where(attackers, sector.float() + 1, torch.zeros_like(prev)))
        held = torch.where(touch & standing[:, :, None] & (held > 0), held, torch.zeros_like(held))
        u["hit_sec"] = held.reshape(Bn, Nn * Nn)
        u["flank_hit"] = (held.amax(1) - 1).clamp(min=0)

    alive = present & (u["men"] > 0) & ~u["gone"]
    standing = alive & ~u["r"]

    # --- the lords ---
    # A lord who falls (dead or shattered) takes his aura with him (the aura below counts standing
    # lords only); st.lord_dead_s times the fall (the metrics' "lord lost").
    has_lord = torch.stack([(u["lord"] & (u["side"] == s)).any(1) for s in (1, 2)], 1)
    lord_alive = torch.stack([(u["lord"] & (u["side"] == s) & alive & ~u["s"]).any(1) for s in (1, 2)], 1)
    dead = has_lord & ~lord_alive
    st.lord_dead_s = torch.where(live[:, None] & dead, torch.where(st.lord_dead_s < 0, torch.zeros_like(st.lord_dead_s),
                                                                    st.lord_dead_s + dt), st.lord_dead_s)
    # sim.json morale.lord_fall: a KILLED lord (health 0) costs the rest of his army the database's
    # general_died_recently (-16) for recent_s, then general_dead (-10) to the end; one who routed OFF
    # THE MAP (gone, men left) general_fled_recently (-16) for fled_s, then nothing; a routed or
    # shattered one still on the field only his aura (+ `routed`, 0) - unless his faction's routed
    # lord crumbles to death (rout_death_s: Vampire Counts, counted as killed that long into his rout).
    fall = cal["morale"]["lord_fall"]
    killed = present & (u["men"] <= 0) & ~u["gone"]
    crumbled = alive & u["r"] & (u["rout_death_s"] > 0) & (u["rout_s"] >= u["rout_death_s"])
    u["dead_s"] = torch.where(live[:, None] & (killed | crumbled | (u["dead_s"] > 0)), u["dead_s"] + dt, u["dead_s"])
    left = present & u["gone"] & (u["men"] > 0) & (u["dead_s"] <= 0)
    u["gone_s"] = torch.where(live[:, None] & (left | (u["gone_s"] > 0)), u["gone_s"] + dt, u["gone_s"])
    side_idx = (u["side"] - 1).clamp(min=0)

    def lords_max(x):           # [B, N]: the side's lords' largest x, for each unit of the side
        return torch.stack([torch.where(u["lord"] & (u["side"] == s), x, torch.zeros_like(x)).amax(1)
                            for s in (1, 2)], 1).gather(1, side_idx)
    killed_s, fled_s = lords_max(u["dead_s"]), lords_max(u["gone_s"])
    fallen = st.lord_dead_s.gather(1, side_idx) >= 0
    M = params.morale
    lord_pts = torch.where(fallen, float(fall.get("routed", 0.0)), 0.0)
    if fall.get("fled"):
        lord_pts = torch.where((fled_s > 0) & (fled_s <= float(fall["fled_s"])),
                               M["ume_concerned_general_fled_recently"], lord_pts)
    if fall.get("killed"):
        lord_pts = torch.where(killed_s > 0, torch.where(killed_s <= float(fall["recent_s"]),
                                                          M["ume_concerned_general_died_recently"],
                                                          M["ume_concerned_general_dead"]), lord_pts)

    # --- morale ---
    d = pw["dist"]
    friends = same_side & ~eye & alive[:, None, :]
    foes = pw["enemy"] & alive[:, None, :]
    # The lord's aura: full within general_aura_radius, fading linearly to 0 at
    # x inspiration_radius_max_effect_range_modifier (database: 70 m, 105 m).
    r0 = M["general_aura_radius"]
    r1 = r0 * (M["inspiration_radius_max_effect_range_modifier"] if cal["morale"].get("aura_fade") else 1.0)
    reach_share = ((r1 - d) / max(r1 - r0, 1e-6)).clamp(0, 1) if r1 > r0 else (d <= r0).float()
    # morale.lord_own_aura false: the aura reaches the lord's units, not the lord himself (measured: a lord at full
    # health with a friend near stands at leadership + flanks secure, no +4; config/nn/sim.json morale.lord_why).
    giver = same_side & standing[:, None, :] & u["encourages"][:, None, :]
    if not cal["morale"].get("lord_own_aura", True):
        giver = giver & ~eye
    aura = torch.where(giver, reach_share, torch.zeros_like(d)).amax(2)
    worth = u["cost"] * u["hp"]
    collapse = morale.army_collapse(u, params)
    mcal = cal["morale"]
    # Flanks secure (morale.secure, the morale probe): no standing enemy within enemy_m (centre to centre), or friends
    # (not lords) within side_m at both sides (side_deg off the unit's facing, left and right); never while a flank is
    # threatened (lf / rf / bf).
    sc = mcal["secure"]
    enemy_d = torch.where(foes & standing[:, None, :], d, torch.full_like(d, 1e9)).amin(2)
    rel = pw["rel_i"] / geometry.DEG
    side_lo, side_hi = float(sc["side_deg"][0]), float(sc["side_deg"][1])
    beside = friends & standing[:, None, :] & (u["men0"][:, None, :] > 1) & (d <= float(sc["side_m"]))
    covered = ((beside & (rel <= -side_lo) & (rel >= -side_hi)).any(2)
               & (beside & (rel >= side_lo) & (rel <= side_hi)).any(2))
    secure = ~(u["lf"] | u["rf"] | u["bf"]) & ((enemy_d >= float(sc["enemy_m"])) | covered)
    # In a fight (the morale's winning / losing): in melee, shooting, or shot at by an enemy shooting at it now.
    slot = torch.arange(N, device=st.device)
    aimed = (firing & (m_target >= 0))[:, :, None] & (m_target.clamp(min=0)[:, :, None] == slot[None, None, :])
    shot_now = (aimed & pw["enemy"]).any(1)
    # the share of health lost by the enemies it fights (in contact, or the target it shoots): winning shows only
    # once one of them has lost morale.combat_lost
    fought = (touch_geo | aimed | aimed.transpose(1, 2)) & pw["enemy"] & alive[:, None, :]
    lost_j = (1 - u["hp_abs"] / u["hp0"].clamp(min=1e-6))[:, None, :]
    foe_lost = torch.where(fought, lost_j, torch.zeros_like(lost_j)).amax(2)
    # The charge's morale (+15, morale.charge): a block of block_s from the charge pose (an attack order, its target's
    # edge within the unit's charge_pose distance - the database's charge_distance_adopt_charge_pose, inside the
    # sprint's charge_dist - not yet in melee), a next one when it ends while the unit still charges or its charge
    # has landed, at most `blocks` a charge; none while routing.
    ti = tgt.clamp(min=0)
    shooter = (u["range"] > 0) & (u["a"] > 0)
    t_reach = pw["gap"].gather(2, ti[:, :, None]).squeeze(2)
    # a shooter under an attack order walks until its target's centre is within range and stops there (the scripted
    # range: archers stopped ~131 m centre to centre, range 130; missile.range_why); range_centre false: the old
    # edge-to-edge 0.95 x range
    if cal["missile"].get("range_centre"):
        in_range = pw["dist"].gather(2, ti[:, :, None]).squeeze(2) <= u["range"]
    else:
        in_range = t_reach <= 0.95 * u["range"]
    close_in = (kind == O.ATTACK) & ~(shooter & in_range) & ~sk_run
    mc = mcal["charge"]
    charging = standing & close_in & ~engaged & (t_reach <= u["charge_pose"])
    landed = standing & engaged & (kind == O.ATTACK) & (charge_now > 0)
    left_s = (u["chm_s"] - dt).clamp(min=0)
    first_block = charging & (u["chm_n"] <= 0)
    next_block = ((u["chm_s"] > 0) & (left_s <= 0) & (charging | landed) & (u["chm_n"] >= 1)
                  & (u["chm_n"] < float(mc["blocks"])))
    if (mc.get("moving") or {}).get("on"):
        # morale.charge.moving: the charge's morale goes with the unit charging = moving (u mv: the last step's own move)
        # under an attack order. A block ends; another follows if the unit still moves, with no cap; in melee a unit that
        # moves under an attack order with no block running starts one (the recordings, 92 battles: at a block's end
        # another block followed in 54-64 % of the cases when the game's moving flag was on, 0-2 % when off, for the
        # 1st, 2nd and 3rd block alike; 93 % of the label's starts deep in melee came with the flag on, against 12 % of
        # deep-melee seconds; the block is the database's charge_timeout 60 in tenths of a second, as waver_base_timeout)
        moving_att = standing & (kind == O.ATTACK) & u["mv"] & ~sk_run
        next_block = (u["chm_s"] > 0) & (left_s <= 0) & moving_att
        if mc["moving"].get("melee_start"):
            first_block = first_block | (moving_att & engaged & (u["chm_s"] <= 0))
    new_block = first_block | next_block
    u["chm_s"] = torch.where(new_block, torch.full_like(left_s, float(mc["block_s"])), left_s)
    u["chm_n"] = torch.where(new_block, u["chm_n"] + 1, u["chm_n"])
    u["chm_n"] = torch.where(~charging & ~engaged & (u["chm_s"] <= 0), torch.zeros_like(u["chm_n"]), u["chm_n"])
    u["chm_s"] = torch.where(standing, u["chm_s"], torch.zeros_like(u["chm_s"]))
    rr = mcal.get("routing_enemies_reach") or {}
    routing_enemies_m = float(rr["m"]) if rr.get("on") else M["routing_unit_effect_distance_front"]
    fight_dealt = fight_taken = None
    if (mcal.get("combat_shooter") or {}).get("on"):
        # morale.combat_shooter: a unit out of melee that shoots weighs only the HP it exchanges with the enemy it shoots
        # at (its missile HP dealt against what that enemy shoots back at it), not the fire of others on it (the
        # recordings: shooters the game shows winning need +8 at every balance of all HP from 0.67 up, +5 below)
        at = u["aim_tgt"]
        tj = at.clamp(min=0)
        back = (at >= 0) & (at.gather(1, tj) == slot[None, :]) & pw["enemy"].gather(2, tj[:, :, None]).squeeze(2)
        own_shot = (at >= 0) & ~engaged
        zero = torch.zeros_like(u["shot_dealt"])
        fight_dealt = torch.where(own_shot, u["shot_dealt"], u["dealt"] + u["shot_dealt"])
        fight_taken = torch.where(own_shot, torch.where(back, u["shot_dealt"].gather(1, tj), zero),
                                  u["taken"] + u["shot_taken"])
    blocks = torch.ones_like(foes)
    rw = mcal.get("rally_weak") or {}
    if rw.get("on"):
        # morale.rally_weak: an enemy weaker than rally_weak.ratio of the router (strategic value: combat potential x
        # health) neither blocks its rally nor counts as the nearest enemy (the probe hyp: night runners at 3 % kept
        # 25-30 m from a router and it rallied at the clock's 18.6 s; the recordings: the strongest enemy within 95 m
        # at 0.17-0.33 of the router - rallies 0.087 a second, none within 95 m 0.136, 0.33-0.5 - 0.006)
        sv = morale.strategic_value(u, params)
        blocks = sv[:, None, :] >= float(rw["ratio"]) * sv[:, :, None]
    ctx = {
        "aura": aura * standing.float(),
        "collapse": collapse,
        "lord_dead_points": lord_pts,
        "secure": secure,
        "in_melee": engaged,
        "in_combat": engaged | firing | shot_now,
        "foe_lost": foe_lost,
        "fight_dealt": fight_dealt,          # (morale.combat_shooter; None: the unit's HP dealt / taken, all of it)
        "fight_taken": fight_taken,
        # A routing expendable unit scares nobody (morale.expendable_scares_expendable: only other expendable units;
        # the database's attribute text, measured: config/nn/sim.json morale.expendable_why).
        "routing_friends": (friends & u["r"][:, None, :] & ~(u["expendable"][:, None, :] & ~(
            u["expendable"][:, :, None] & bool(cal["morale"].get("expendable_scares_expendable"))))
                            & (d <= M["routing_unit_effect_distance_front"])).float().sum(2),
        # routing enemies within the database's routing_unit_effect_distance_front (100 m, centre to centre), or within
        # morale.routing_enemies_reach m when it is on (measured: the recordings' labels count routers to ~200 m)
        "routing_enemies": (foes & u["r"][:, None, :] & (d <= routing_enemies_m)).float().sum(2),
        # The game's "under missile attack" holds morale.under_fire_s after the last projectile hit (measured).
        "under_fire": u["under_fire_s"] < float(cal["morale"].get("under_fire_s", 2.0)),
        # A standing enemy within enemy_effect_range (70 m) worth morale.strong_ratio times the unit or more (cost x
        # health): the game's 'enemies superior in strength and speed' (the morale probe; speed is not needed).
        "strong_enemy": (foes & standing[:, None, :] & (d <= M["enemy_effect_range"])
                         & (worth[:, None, :] >= knobs.get(u, "strong_ratio", float(mcal["strong_ratio"]), 3)
                            * worth[:, :, None])).any(2),
        # morale.strong_enemy rule "balance": the enemies' strength around the unit against its own and its
        # friends' (morale.strong_power); the term is then the database's -3 ... -24 scale (morale.strong_points).
        "strong_power": (morale.strong_power(u, d, foes & standing[:, None, :], friends & standing[:, None, :], params)
                         if (mcal.get("strong_enemy") or {}).get("rule") == "balance" else None),
        # (morale.rally_any_enemy: any enemy blocks the rally, routing ones too - the recordings: within 95 m of a
        # routing enemy 0.9 % rallied a second, 12.6 % with none; build/open_battle/spec.md 4)
        "enemy_near": (foes & (standing[:, None, :] | bool(mcal.get("rally_any_enemy"))) & blocks
                       & (d <= mcal["rally_free_m"])).any(2),
        # the nearest standing enemy's centre (m; morale.rally_hazard: the rally's chance a second by it) and the time
        "foe_d": torch.where(foes & standing[:, None, :] & blocks, d, torch.full_like(d, 1e9)).amin(2),
        "t": st.t,
    }
    began = morale.step(u, ctx, params, dt)
    standing = alive & ~u["r"]
    pin_s = float(cal["contact"].get("rout_pin_s") or 0.0)
    r_pin = torch.zeros_like(alive)
    if pin_s > 0 and "rpin_s" in u:
        # the rout's exit starts for a formation that routs from melee (in melee the step before); it ends after
        # rout_pin_s, at a rally, or when the unit is gone
        start_pin = began & u["m"] & (u["men0"] > 1)
        keep = alive & u["r"] & ((u["rpin_s"] > 0) | start_pin)
        u["rpin_s"] = torch.where(start_pin, torch.full_like(u["rpin_s"], 1e-6),
                                  torch.where(keep, u["rpin_s"], torch.zeros_like(u["rpin_s"])))
        u["rpin_d"] = torch.where(keep & ~start_pin, u["rpin_d"], torch.zeros_like(u["rpin_d"]))
        r_pin = keep

    # --- fatigue ---
    # An ended move is rest too: KEEP preserves the order, not an eternal ready stance.
    # Explicit attacks and aiming/turning shooters remain ready between active bouts.
    # The recordings do not expose a combat-stance flag; pending work is our proxy.
    # movement.velocity brakes to rest within 0.5 m of the destination.
    pending_move = point & (((u["ox"] - u["x"]) ** 2 + (u["oz"] - u["z"]) ** 2) > 0.5 ** 2)
    idle = standing & ~pending_move & (kind != O.ATTACK) & (aim_at < 0) & ~sk_run
    # The charge sprint (the database's battle_entities charge distance and charge speed): under an attack order, at
    # a run or a walk, a unit closes the last charge_dist metres (30 infantry, 35 lords) to its target at its charge
    # speed - a charge then (the run-up above); a move order gives none (the melee probe: the last 30 m at 3.65-3.88
    # m/s at a run of 3.0, the last 10 m 3.9-4.7, at a walk the same). Movement below runs it; fatigue pays for it.
    sprint = close_in & (t_reach <= u["charge_dist"]) & ~busy_t
    if not cal["contact"].get("pursuit_sprint", True):
        # (contact.pursuit_sprint false: no sprint after a routing target - the pursuit has no charge, pursuit_why; a
        # pursuer runs at its run speed and falls behind a faster router: the routmob probe, contact.pursuit_sprint_why)
        sprint = sprint & ~u["r"].gather(1, ti)
    fcal = cal["fatigue"].get("calibration", {})
    if fcal.get("charging") == "sprint":
        # The charge's fatigue (the database's charging +34 a tick) while the unit sprints at its target under the
        # attack order, up to the contact; nothing after the blow (fatigue.calibration.charging_why).
        fat_charge = standing & sprint & ~engaged
    else:
        # (the old rule: +34 only in the first fatigue.calibration.charge_s seconds after a charge's first blow)
        charge_s = float(fcal.get("charge_s", decay))
        fat_charge = engaged & (charge_now > 1 - charge_s / decay)
    activity = {"melee": engaged, "charging": fat_charge, "shooting": firing,
                "running": speed > u["walk"] + 0.3, "walking": speed > 0.3,
                "idle": idle, "active": alive, "attack": kind == O.ATTACK, "single": u["men0"] <= 1,
                "run_order": u["order_run"].bool(), "routing": alive & u["r"], "vigour": vigour,
                "move_order": (kind == O.MOVE) | (kind == O.WITHDRAW), "shooter": u["range"] > 0,
                # it strikes the enemies it touches this step (not leaving, not in a hold's exit window): a lone man
                # in melee pays by it (fatigue.calibration.single_melee)
                "strikes": striker & engaged}
    near_m = float(cal["fatigue"].get("calibration", {}).get("ready_enemy_m", 0.0))
    if near_m > 0:
        activity["enemy_near"] = (foes & standing[:, None, :] & (d <= near_m)).any(2)
    fatigue.step(u, activity, params, dt)

    # --- movement ---
    gx, gz = u["x"].clone(), u["z"].clone()
    gx = torch.where(point, u["ox"], gx)
    gz = torch.where(point, u["oz"], gz)
    tx, tz = u["x"].gather(1, ti), u["z"].gather(1, ti)
    attack = kind == O.ATTACK
    # (t_reach, in_range, close_in: the target's reach above, at the charge's morale)
    gx = torch.where(close_in, tx, gx)
    gz = torch.where(close_in, tz, gz)
    want = torch.where(run, u["run"], u["walk"])
    # (the charge sprint: `sprint` above, at the fatigue)
    want = torch.where(sprint, u["charge_speed"], want)
    moving = standing & (point | close_in)
    if sk_cal.get("on") and "skr" in u:
        # the skirmish run: away from the nearest enemy at the run speed, whatever the order
        gx = torch.where(sk_run, sk_x, gx)
        gz = torch.where(sk_run, sk_z, gz)
        want = torch.where(sk_run, u["run"], want)
        moving = moving | sk_run
    # (morale.flee_near_m: 0 - away from every standing enemy weighted 1/d, the own edge only with none standing;
    # absent / 150 - the old rule: only enemies within 150 m, the own edge beyond; config/nn/sim.json flee_near_why)
    fx, fz = movement.flee_goal(u, pw, alive, st.bounds, float(cal["morale"].get("flee_near_m", 150.0)))
    routing = alive & u["r"]
    dodge = cal["morale"].get("rout_dodge")
    if dodge and dodge.get("on", True):
        # A chased router does not run straight from its chaser (morale.rout_dodge; config/nn/sim.json rout_dodge_why):
        # with a standing enemy formation within dodge["m"] it heads dodge["deg"] off the line from that enemy, to a
        # side fixed per (battle row, unit), so its speed away from the chaser is cos(deg) of its speed (the game's
        # routers: 0.48-0.52 at 3-20 s of the rout; the simulator's straight flight gave 0.85-0.90).
        fx, fz = movement.dodge_goal(u, pw, alive, fx, fz, float(dodge["m"]), float(dodge["deg"]), st.bounds)
    gx = torch.where(routing, fx, gx)
    gz = torch.where(routing, fz, gz)
    want = torch.where(routing, u["run"] * cal["morale"]["rout_speed"], want)
    if pin_s > 0 and cal["contact"].get("rout_pin_speed"):
        # in its rout's exit it gathers speed: the recordings' share of its own free rout speed by second (0.41 in the
        # first second to 1.0 by the seventh; contact.rout_pin_speed)
        want = torch.where(r_pin, want * movement.ramp(u["rpin_s"], cal["contact"]["rout_pin_speed"]), want)
    crowd = cal["morale"].get("rout_crowd")
    if crowd:
        # A router among other formations is slowed by their men (morale.rout_crowd; soft collision of the models): a
        # share of its rout speed by the standing formations (any, lords too) whose centres are within crowd["m"] of
        # its own - own and enemy counted apart, the lower share taken; not in its rout's exit (that has its ramp).
        near = (pw["dist"] < float(crowd["m"])) & standing[:, None, :] & ~eye
        n_own = (near & same_side).sum(2).clamp(max=len(crowd["own"]) - 1)
        n_en = (near & ~same_side & (u["side"][:, None, :] > 0)).sum(2).clamp(max=len(crowd["enemy"]) - 1)
        f_own = torch.tensor(crowd["own"], dtype=want.dtype, device=want.device)[n_own]
        f_en = torch.tensor(crowd["enemy"], dtype=want.dtype, device=want.device)[n_en]
        slow = routing & (u["men0"] > 1) & ~r_pin
        want = torch.where(slow, want * torch.minimum(f_own, f_en), want)
    moving = moving | routing
    # A unit in melee closes up to its nearest opponent (walking) until the formations touch.
    foe_gap = torch.where(touch & standing[:, None, :], pw["gap"], torch.full_like(pw["gap"], 1e9))
    close_to = foe_gap.argmin(2)
    close_gap = foe_gap.min(2).values
    closing = engaged & ~leaving & (close_gap > 0) & (close_gap < 1e9)
    retarget = torch.zeros_like(closing)
    if cal["contact"].get("retarget_walk"):
        # A formation in melee told to attack an enemy it does not touch (and not leaving: the target nearer than
        # attack_leave's leave_m) is not held where it fights nor closes on its nearest foe: it walks to its target
        # (contact.retarget_walk; the fresh-order probe: swordsmen told to attack clanrats 4 m beside the clanrats
        # they fought turned 75-80 deg and walked 8 m to them in 6-8 s, moving 0.85-0.90 of the first 10 s).
        ti_r = tgt.clamp(min=0)[:, :, None]
        retarget = (engaged & ~leaving & (kind == O.ATTACK) & (tgt >= 0) & ~touch.gather(2, ti_r).squeeze(2)
                    & (u["men0"] > 1) & (u["range"] <= 0))
        closing = closing & ~retarget
        want = torch.where(retarget, u["walk"], want)
    push = torch.zeros_like(closing)
    push_cal = cal["contact"].get("move_push") or {}
    if push_cal.get("on") and leave_m > 0:
        # A formation under a move contact.leave_m or more away that is not a leave (given before the contact, its point
        # through the enemy: leave_away_only) is not held where it touches the enemy: it pushes on to its point, fighting,
        # at a share of its order's speed by the share of its front the enemy formations it touches cover across its way
        # (geometry.blocked_share; contact.move_push walk / run tables; config/nn/sim.json contact.move_push_why: the
        # walkpass2 probe - swordsmen whose straight way clipped held clanrats by 2-6 m went on at 0.65-0.89 of their
        # speed with the melee flag lit, by 11 m at 0.26-0.49, by 14-15 m and straight through at 0.08-0.21).
        push = engaged & (kind == O.MOVE) & ~leaving & (point_d >= leave_m) & (u["men0"] > 1)
        cover = geometry.blocked_share(u, pw, touch & standing[:, None, :], u["ox"], u["oz"])
        share = torch.where(run, movement.ramp(cover, push_cal["run"]), movement.ramp(cover, push_cal["walk"]))
        want = torch.where(push, want * share, want)
        closing = closing & ~push
    cx, cz = u["x"].gather(1, close_to), u["z"].gather(1, close_to)
    gx = torch.where(closing, cx, gx)
    gz = torch.where(closing, cz, gz)
    want = torch.where(closing, u["walk"], want)
    moving = moving | closing
    locked = engaged & (~leaving | pinned) & ~closing & ~retarget & ~push
    if cal["contact"].get("chase"):
        # a chaser in contact with its leaving target (above) is not held in place: it moves with it (below)
        locked = locked & ~chasing
    # Hits of gunpowder slow the small formations they hit, routers too (missile.hit_slow; config/nn/sim.json hit_slow.why:
    # the hitslow probe - clanrats running at handgunners x0.68, walking x0.94; bolts nothing; the hit reactions)
    if "hs_s" in u:
        want = want * missile.hit_slow_factor(u, params, want > u["walk"] + 0.3)
    # Turning on the move (turn.move_turn; the database's battle_entities turn_rate, the turning probe
    # build/movelords): a unit heading for a point off its facing turns towards it at its men's turn rate (a formation's
    # men about-face where they stand: its front becomes the old back rank) and runs only the way it faces - its speed
    # towards the point is its speed x cos(the angle between its facing and the way), none beyond 90 deg. A formation
    # stepping a few metres back or aside (contact.step_m) and a unit held in melee do not turn so.
    t_cal = cal.get("turn") or {}
    steer = torch.zeros_like(moving)
    if t_cal.get("move_turn"):
        dxg, dzg = gx - u["x"], gz - u["z"]
        d2 = dxg * dxg + dzg * dzg
        step_m = float(cal["contact"]["step_m"])
        sidestep = standing & point & ~routing & (d2 < step_m ** 2) & (u["men0"] > 1)
        steer = moving & alive & ~locked & ~sidestep & (d2 > 0.25)
        before_b = u["b"].clone()
        movement.face(u, dxg, dzg, steer)
        movement.limit_turn(u, before_b, steer, u["turn"] * dt)
        off = torch.remainder(torch.rad2deg(torch.atan2(dxg, dzg)) - u["b"] + 180, 360) - 180
        want = torch.where(steer, want * torch.cos(torch.deg2rad(off)).clamp(min=0), want)
    vx, vz = movement.velocity(u, gx, gz, want, moving, dt)
    stop = locked | ~alive
    vx = torch.where(stop, torch.zeros_like(vx), vx)
    vz = torch.where(stop, torch.zeros_like(vz), vz)
    if cal["contact"].get("chase"):
        # a chaser in contact keeps pace with its leaving target, at most at its own run speed (it falls behind a faster
        # leaver; out of contact it runs at the target like any attacker)
        tvx, tvz = vx.gather(1, ti), vz.gather(1, ti)
        cap = (u["run"] / torch.sqrt(tvx * tvx + tvz * tvz).clamp(min=1e-6)).clamp(max=1.0)
        keep_pace = chasing & standing
        vx = torch.where(keep_pace, tvx * cap, vx)
        vz = torch.where(keep_pace, tvz * cap, vz)
    u["vx"], u["vz"] = vx, vz
    u["x"] = u["x"] + vx * dt
    u["z"] = u["z"] + vz * dt
    if pin_s > 0 and "rpin_s" in u:
        # the rout's exit: the way run since it began (the contact test above leaves it out) and its clock
        u["rpin_d"] = torch.where(r_pin, u["rpin_d"] + torch.sqrt(vx * vx + vz * vz) * dt, u["rpin_d"])
        u["rpin_s"] = torch.where(r_pin, u["rpin_s"] + dt, u["rpin_s"])
        over = r_pin & (u["rpin_s"] >= pin_s)
        u["rpin_s"] = torch.where(over, torch.zeros_like(u["rpin_s"]), u["rpin_s"])
        u["rpin_d"] = torch.where(over, torch.zeros_like(u["rpin_d"]), u["rpin_d"])
    if cal["contact"].get("friend_push", True):
        # (contact.friend_push false: own formations stand in each other as in the game - the recordings show own
        # centres within 2 m in 0.2-0.3 % of pair-seconds and no drift apart; config/nn/sim.json contact.friend_push_why)
        pw2 = geometry.pairwise(u, spacing)
        px, pz = movement.separate(pw2, standing[:, :, None] & standing[:, None, :] & same_side & ~eye)
        u["x"] = u["x"] + px
        u["z"] = u["z"] + pz
    movement.clamp_to_map(u, st.bounds)

    # --- facing and the observed flags ---
    spd = torch.sqrt(vx * vx + vz * vz)
    mv = alive & (spd > 0.3)
    # A formation steps a few metres back or aside to its point without turning round.
    goal_d = torch.sqrt((gx - u["x"]) ** 2 + (gz - u["z"]) ** 2)
    shuffle = standing & point & ~routing & (goal_d < cal["contact"]["step_m"]) & (u["men0"] > 1)
    # (a unit turning on the move keeps the facing it turned to above)
    movement.face(u, vx, vz, mv & ~shuffle & ~steer)
    near_foe = torch.where(touch & standing[:, None, :], d, torch.full_like(d, 1e9))
    opp = near_foe.argmin(2)
    has_opp = near_foe.min(2).values < 1e9
    ox_, oz_ = u["x"].gather(1, opp), u["z"].gather(1, opp)
    # A formation under a move to a point farther than contact.step_m keeps the facing of its way: it does not turn to the
    # enemy it clips (contact.push_keep_facing; the walkpass2 probe: the game's walker stays within 15 deg of its way while
    # it pushes past held clanrats, melee flag lit; turned to them, the twin's walker ran only cos(53 deg) of its speed
    # towards its point - the stall of rect_gap)
    face_opp = has_opp & ~mv & standing & ~steer
    if cal["contact"].get("push_keep_facing"):
        face_opp = face_opp & ~((kind == O.MOVE) & (point_d >= float(cal["contact"]["step_m"])) & (u["men0"] > 1))
    fdx, fdz = ox_ - u["x"], oz_ - u["z"]
    if cal["contact"].get("square_front"):
        # Formation against formation (contact.square_front; config/nn/sim.json contact.square_front_why): the fight's
        # anchor is the unit's attack target where it touches it, else its nearest opponent. A formation attacking its
        # anchor faces square to the anchor's side it touches (its front if it meets the anchor's front: the fronts
        # parallel), not the anchor's centre; one that does not attack it keeps its facing, and turns (square to that
        # side) only to an anchor attacking it from outside its front (contact.front_deg) - not to a passer or a unit
        # fighting another (the sidecontact probe: swordsmen attacking clanrats 6-18 m off their line turned 6-16 deg on
        # the way in and back to 180-186 in melee, the held clanrats stayed at 3-5; the twin turned both to the other's
        # centre, 206 / 17, the box's corner off the bystander; the walkpass2 probe: held clanrats the walker clipped
        # never targeted it). The anchor's side: where this unit is seen from it, as geometry.pairwise.
        on_t = (kind == O.ATTACK) & (tgt >= 0) & touch.gather(2, tgt.clamp(min=0)[:, :, None]).squeeze(2)
        anc = torch.where(on_t, tgt.clamp(min=0), opp)
        ax_, az_ = u["x"].gather(1, anc), u["z"].gather(1, anc)
        ob = u["b"].gather(1, anc) * geometry.DEG
        rel = geometry.wrap(torch.atan2(u["x"] - ax_, u["z"] - az_) - ob)
        of, od = pw["front"].gather(1, anc), pw["depth"].gather(1, anc)
        s, c = torch.sin(rel), torch.cos(rel)
        thru = s.abs() * od <= c.abs() * of
        side = torch.where(thru, torch.where(c >= 0, 0.0, 180.0), torch.where(s >= 0, 90.0, -90.0)) * geometry.DEG
        into = ob + side + math.pi                                     # against the outward normal of that side
        sq = (u["men0"] > 1) & (u["men0"].gather(1, anc) > 1)
        n_slot = torch.arange(u["men"].shape[1], device=anc.device)[None, :]
        attacked = (kind.gather(1, anc) == O.ATTACK) & (tgt.gather(1, anc) == n_slot)
        seen = geometry.wrap(torch.atan2(ax_ - u["x"], az_ - u["z"]) - u["b"] * geometry.DEG).abs()
        outside = seen > float(cal["contact"]["front_deg"]) * geometry.DEG
        face_opp = face_opp & (~sq | on_t | (attacked & outside))
        fdx = torch.where(sq, torch.sin(into), fdx)
        fdz = torch.where(sq, torch.cos(into), fdz)
    movement.face(u, fdx, fdz, face_opp)
    wt = aim_at.clamp(min=0)
    if done_deg is not None:
        # a standing shooter keeps its facing; it turns only to a target beyond stand_fire_arc_deg (turning above)
        movement.face(u, u["x"].gather(1, wt) - u["x"], u["z"].gather(1, wt) - u["z"],
                      turning & ~mv & ~has_opp & standing)
    else:
        mt = m_target.clamp(min=0)
        movement.face(u, u["x"].gather(1, mt) - u["x"], u["z"].gather(1, mt) - u["z"], firing & ~mv & ~has_opp)
        # A shooter that aims or turns to its target (not yet shooting) faces it too.
        movement.face(u, u["x"].gather(1, wt) - u["x"], u["z"].gather(1, wt) - u["z"],
                      (aim_at >= 0) & ~firing & ~mv & ~has_opp & standing)
    # A formation in melee turns slowly (measured): an enemy on its flank or rear stays there.
    movement.limit_turn(u, old["b"], engaged & ~leaving & (u["men0"] > 1), cal["contact"]["melee_turn_deg_s"] * dt)
    # Out of melee a standing unit turns in place at turn.formation_deg_s (measured, config/nn/sim.json turn.why); a
    # lord or another single entity at his turn rate (the database's battle_entities turn_rate; turn.single_deg_s, if
    # set, overrides it). A unit on the move turns as above (turn.move_turn), else it faces the way it walks.
    if t_cal.get("formation_deg_s"):
        single = (torch.full_like(u["b"], float(t_cal["single_deg_s"])) if t_cal.get("single_deg_s") is not None
                  else u["turn"])
        rate = torch.where(u["men0"] > 1, torch.full_like(u["b"], float(t_cal["formation_deg_s"])), single)
        movement.limit_turn(u, old["b"], standing & ~mv & ~engaged & ~steer, rate * dt)
    router_hit = torch.where(strike & touch_geo & u["r"][:, None, :], d, torch.full_like(d, 1e9))
    router = router_hit.argmin(2)
    has_router = router_hit.min(2).values < 1e9
    target = torch.where(has_opp & standing, opp, torch.where(shown, shown_tgt,
                                                               torch.where(has_router, router, torch.full_like(opp, -1))))
    u["target"] = torch.where(alive, target, torch.full_like(target, -1))
    # a router in its rout's exit touched by a standing enemy is in melee too (the game's flag: median 7 s after a rout
    # from melee, contact.rout_pin_why)
    held_router = r_pin & (u["rpin_s"] > 0) & (touch & standing[:, None, :]).any(2) if pin_s > 0 else torch.zeros_like(alive)
    # a router within the pursuit's reach of a chaser (contact.pursue_gap_m above) is in melee too (the pursuit probe: the
    # game's flag on 0.96-1.00 with the formations' edges within 16 m, 0.26 beyond)
    held_router = held_router | (alive & u["r"] & chase_touch.any(2))
    # a router with men stuck among a standing enemy's (contact.rout_stuck above) is in melee too (the routmob2 probe's
    # 'neighbours': the slaves' flag on to 45 s after the rout while a third to a tenth of their men stayed there)
    held_router = held_router | (alive & u["r"] & stuck_touch.any(1))
    u["m"] = alive & (engaged | has_router | held_router)
    u["mv"] = mv
    u["f"] = mv & (spd > u["walk"] + 0.3)
    u["fire"] = shown
    hold = standing & (kind == O.HOLD)
    u["ox"] = torch.where(hold, u["x"], torch.where(attack & standing, tx, u["ox"]))
    u["oz"] = torch.where(hold, u["z"], torch.where(attack & standing, tz, u["oz"]))
    if broke_off is not None:
        # the melee exit's window is over in contact: HOLD from the next step (contact.breakoff above)
        u["order_kind"] = torch.where(broke_off, torch.full_like(u["order_kind"], O.HOLD), u["order_kind"])
        u["order_target"] = torch.where(broke_off, torch.full_like(u["order_target"], -1), u["order_target"])
        u["ox"] = torch.where(broke_off, u["x"], u["ox"])
        u["oz"] = torch.where(broke_off, u["z"], u["oz"])
        u["leave_s"] = torch.where(broke_off, torch.zeros_like(u["leave_s"]), u["leave_s"])
        u["exit_s"] = torch.where(broke_off, torch.zeros_like(u["exit_s"]), u["exit_s"])
    if sk_cal.get("on") and sk_cal.get("show_far_point") and "sk_shown" in u:
        # the observed order point of a unit on a skirmish run is the run's far point, as the game shows it (its
        # ordered position >= 1000 m away: build/skirm/f_farpoint.py); the order's own point is kept for the next step
        shown = sk_run & standing
        u["sk_ox"] = torch.where(shown, u["ox"], u["sk_ox"])
        u["sk_oz"] = torch.where(shown, u["oz"], u["sk_oz"])
        u["ox"] = torch.where(shown, sk_x, u["ox"])
        u["oz"] = torch.where(shown, sk_z, u["oz"])
        u["sk_shown"] = shown
    threat_cal = cal["threat"].get("calibration", {})
    # A calibrated flag uses the position and bearing visible in this observation.
    # OFF preserves the legacy pre-move geometry for exact baseline comparison.
    threat_pw = geometry.pairwise(u, spacing, rect=rect, melee_depth=mdepth) if threat_cal.get("on") else pw
    masks = {} if threat_cal.get("on") else {"alive": alive, "standing": standing}
    u.update(threat_flags(u, threat_pw, params, **masks))
    dead = present & ((u["men"] <= 0) | u["gone"])
    for k in ("m", "mv", "f", "fire", "w", "lf", "rf", "bf"):
        u[k] = u[k] & ~dead

    u.update(tired)
    abilities.restore(u, base)
    effects.restore(u, innate)
    effects.refresh_on(u, params)

    # --- is the battle over ---
    standing = present & (u["men"] > 0) & ~u["gone"] & ~u["r"]
    stand1 = (standing & (u["side"] == 1)).any(1)
    stand2 = (standing & (u["side"] == 2)).any(1)
    t_new = st.t + dt
    defender = 3 - st.attacker
    over = ~stand1 | ~stand2 | (t_new >= params.limit_s)
    winner = torch.where(stand1 & ~stand2, 1, torch.where(stand2 & ~stand1, 2, defender))
    finish = live & over
    # Frozen battles keep their last state.
    keep = live[:, None]
    for k in list(u):
        if u[k] is not old[k]:
            u[k] = torch.where(keep, u[k], old[k])
    st.t = torch.where(live, t_new, st.t)
    st.winner = torch.where(finish, winner, st.winner)
    st.done = st.done | finish
    return st


_COMPILED = {}


def stepper(device, compile=None):
    """The step function to use: compiled by torch.compile on CUDA (about 10x faster on the GPU;
    the container has no C++ compiler for the CPU), plain elsewhere or when compile is False."""
    use = compile if compile is not None else torch.device(device).type == "cuda"
    if not use:
        return step
    if "step" not in _COMPILED:
        _COMPILED["step"] = torch.compile(step, dynamic=False)
    return _COMPILED["step"]


def run(st, policy, params=None, until_s=None, every_s=None, record=None, compile=None):
    """Run the batch until every battle is over (or until_s): policy(st) -> Orders, asked every
    every_s seconds (default: every step); record(st), if given, after every step."""
    params = params or load()
    dt = params.dt
    limit = until_s or params.limit_s
    every = max(1, int(round((every_s or dt) / dt)))
    advance = stepper(st.device, compile)
    n = 0
    orders = None
    while not bool((st.done | (st.t >= limit)).all()):
        if orders is None or n % every == 0:
            orders = policy(st)
        advance(st, orders, params, dt)
        if record is not None:
            record(st)
        n += 1
    return st

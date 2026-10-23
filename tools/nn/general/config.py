"""The general's settings (S28 skeleton, docs/en/training/general.md; design/general/index.html).

Saved with a checkpoint (checkpoint.py "general": the dataclass's fields), so a network with a general rebuilds with
the same sizes and rules. The skeleton's simplifications are fields too (docs "Simplifications of the skeleton"): each
is a value of its field, never code that cannot be switched; a value whose full version is not built yet is refused
when the config is made (NotBuilt), so a run never thinks it has what it has not.
"""
from dataclasses import asdict, dataclass, fields


TARGET_TYPES = ("own", "enemy", "tile")     # a mission target's type (net.target_kind's 0, 1, 2)


@dataclass(frozen=True)
class MissionKind:
    """One mission kind ("Виды заданий"): its number (the row of the general's learned kind table, the logs' kind), its
    name, the target types it takes (TARGET_TYPES: the general may not pair it with another - its logits there are
    -inf, net.General.decide) and what carrying it out is (mission.py measures it every decision, 0..1 a unit; the
    executor's bonus is that against its running mean, docs "Missions")."""
    index: int
    name: str
    targets: tuple
    meaning: str


# Every mission kind, in one place: the network has GeneralConfig.kinds of them (the first kinds of this list; its
# kind table's rows are these numbers). A new kind goes at the end (the trained rows keep their numbers).
# the task masks' parts on by default (taskmask.PARTS; docs/en/training/general.md "Task masks": the ablation at minute 0
# of g13 - withdraw_attack, attack_target and shoot_* cost the rating 0.2-1.2 and are off) and the volley lock shoot_volley
MASK_PARTS = "withdraw_move,attack_move,hold_target,hold_move,screen_move,shoot_group,shoot_volley"

MISSION_KINDS = (
    MissionKind(0, "attack", ("enemy",), "close on the target group and fight it in melee"),
    MissionKind(1, "hold", ("own", "tile"), "stand within reach of the target point (a tile's centre, a group's)"),
    MissionKind(2, "flank", ("enemy",), "fight the target group from its side or rear"),
    MissionKind(3, "screen", ("own",), "stand between the target group and the enemy nearest to it"),
    MissionKind(4, "shoot", ("enemy",), "keep the target group in missile range, out of melee (no missiles: at a "
                                        "distance)"),
    MissionKind(5, "withdraw", ("own", "tile"), "go to the target point away from the enemy, out of melee"),
)


def allowed(kinds):
    """[[bool] * len(TARGET_TYPES)] * len(kinds): which target types each kind takes."""
    return [[t in k.targets for t in TARGET_TYPES] for k in kinds]


def mission_kinds(cfg):
    """The cfg's mission kinds (MISSION_KINDS[:cfg.kinds])."""
    return MISSION_KINDS[:cfg.kinds]


class NotBuilt(ValueError):
    """A setting whose full version is not built yet (the skeleton's simplification is the only one there is)."""


@dataclass(frozen=True)
class GeneralConfig:
    card: int = 64             # D: the width of every card (units, tiles, groups, mission kinds)
    reader_hidden: int = 128   # the army reader's first layer (army_reader.ArmyReader hidden)
    council_layers: int = 2    # "Совещание": units with tiles, each asks each (simplification 3: small)
    council_heads: int = 4
    group_layers: int = 1      # "Совещание групп": groups listen to the tiles and all groups
    ff: int = 2                # feed-forward width = ff x card in both councils
    places: int = 10           # G: group places a side ("Сбор групп": 10 ours + 10 the enemy's)
    rounds: int = 3            # slot attention's rounds inside one decision
    kinds: int = len(MISSION_KINDS)  # K: mission kinds ("Виды заданий": a learned table; MISSION_KINDS names them)
    target_grid: int = 4       # a mission's target is a group (ours or theirs) or a tile of this coarse grid (4 x 4 of
    #                            the side's frame: blocks of (sectors / 4)^2 map sectors) - <= 2G + 16 candidates
    center: float = 0.7        # the group collection's keys less this x their side's mean key (a learned share, starts
    #                            here): the units compete for the places by what tells them apart - without it a fresh
    #                            general put nearly every unit into one group (1.0-1.3 groups a side; with the layer-
    #                            normed keys below ~2.3 groups of ~5 units); 1: ~6 groups of ~2 (46 % single units);
    #                            0.7: ~3.4 groups of ~3.6 (build/gfix/slotproto.py, real armies); 0: plain slot attention
    sharp: float = 15.0        # the group collection's keys and queries layer-normed, their dot product / sqrt(card) x
    #                            this learned sharpness (its start): a fresh general's units have one clear main place
    #                            (share 0.87-0.92 at 15, 0.79-0.82 at 10) and the groups' cards differ (cos 0.76-0.88) -
    #                            with the plain dot product (0) every unit split ~0.100 over the 10 places, its group a
    #                            draw of float noise, the group cards of a plan alike (g8: cos 0.995)
    stick: float = 1.0         # a re-plan: the bonus on a unit's logit of its last group in the group collection (a
    #                            learned value, starts here): the groups keep their units unless the attention prefers
    #                            another place clearly; 0: none
    kind_stick: float = 1.0    # a re-plan: the bonus on a group's logit of its last plan's mission kind (a fixed
    #                            rule, not a weight): the group that kept its number and some of its units (net.decide:
    #                            prev_kind) keeps its kind unless the general prefers another clearly; a kind with no
    #                            target of its types now stays masked; 0: none (g10: the kind changed at 57 % of the
    #                            re-plans of a group kept, the choice drifting)
    task_mask: bool = True     # the mission limits the executor (taskmask.py: a unit's kinds, targets and sectors cut to
    #                            its group's mission, a rule, not a weight) and the general gives "shoot" only to a
    #                            group with a shooter (a unit whose card has a missile range); False: neither, the
    #                            executor computes what it did to the bit (a checkpoint saved before it: True)
    task_mask_parts: str = MASK_PARTS  # ... which of the masks (taskmask.PARTS, a comma list; "all" every one, "none"):
    #                            withdraw_attack, withdraw_move, shoot_melee, shoot_target, attack_target, attack_move,
    #                            hold_target, hold_move, screen_move, shoot_volley (the executor's) and shoot_group
    #                            (the general's);
    #                            the default: those that cost nothing at minute 0 (docs "Task masks": the ablation)
    shoot_share: float = 0.5   # ... "shoot" for a group whose shooters are at least this share of its units (net.shoot_ok;
    #                            "at least one" let 61-66 % of the units under shoot be infantry without missiles)
    melee_stay: bool = True    # "withdraw" not for a group that fights in melee and has not wavered (net.melee_ok,
    #                            taskmask.fighting: some unit in melee, none wavering or worse); the teacher keeps the same
    #                            rule (a rule, not a weight: an older checkpoint gets it too). The human's battle against
    #                            g31: the whole infantry with the lord got "withdraw" in melee, 46 orders, those units
    #                            killed 0.03 men/s against 0.19; False: withdraw anywhere
    # --- the skeleton's simplifications (docs "Simplifications of the skeleton") ---
    tiles: str = "sectors"     # 1: the map tiles are the unit network's 16 x 16 sectors (8 features each) projected to
    #                            the card's width; "cnn" (the map CNN's 32 x 32 tiles, map_cnn.py) is not built yet
    effects: str = "off"       # 2: the effect cards are zeros ("off"); the effects' analysis is not built yet
    zone: str = "place"        # 5: the deployment zone is the area tools/nn/armies/place.py deploys an army in (the
    #                            arena's rule); "map" (the map's zone layers, map_input.py) is not built yet
    deploy: str = "net"        # 5: "net" the general picks the groups' places, a rule lays the units out; "script":
    #                            place.py's placement as before (the general still forms groups and gives missions)
    # --- re-planning ("Пересмотр замысла") ---
    replan_s: float = 30.0     # the plan is made again every this many seconds of battle ...
    replan_min_s: float = 3.0  # ... and on an event (a group broken, a lord fallen, the enemy near), but not sooner
    #                            than this after the last plan
    replan_near_m: float = 250.0  # ... the event "the enemy near": a standing enemy unit within this of one of our
    #                            groups' centres that was farther at the plan (teacher.HOLD_M: the teacher's waiting
    #                            groups attack together at once, not at the next 30 s plan); 0: no such event
    broken: float = 0.5        # a group is broken when more than this share of its units routs or is destroyed
    # --- the executor's task input ("Задание отряда") ---
    dist_m: float = 100.0      # a target's distance from the unit / this ...
    dist_cap: float = 20.0     # ... at most this
    # --- the architecture's repairs after the module atlas of g17r (build/diag_atlas2; docs "Tried and rejected") ---
    groups: str = "rule"       # the group collection: "rule" (groups.py: roles - shooters, melee, lord and heroes, fast -
    #                            then clusters by place within a role, at most `places` a side, both sides alike; the
    #                            slot attention's keys and queries take no part and learn nothing) or "net" (the learned
    #                            slot attention: the atlas found it degenerate - one group at the first plan, cards alike)
    group_r: float = 100.0     # ... "rule": units of a role join while their clusters' centres are closer than this, m
    lord_join_m: float = 60.0  # ... "rule": a lone lord (or hero) joins the nearest melee group whose centre is closer
    group_centre: bool = True  # ... "rule": a group card reads its units' values less their side's mean (how the group
    #                            differs from its army: the council's unit cards share a large common part)
    split_kinds: bool = True   # the executor's own mission kind table (exec_kinds; the group card detached): the
    #                            executor's epochs no longer move the general's kind choice (the atlas: -0.21 of the
    #                            withdraw share an update through the shared table); False: one table for both
    tile_feats: str = "v2"     # the tiles' numbers: "v1" the unit network's 8 sector numbers; "v2" the three always
    #                            zero in the simulator (the enemy seen earlier, own and enemy routs) replaced by own
    #                            shooters', the enemy's shooters' and the enemy's fast units' strength (tiles.extra)
    tile_norm: bool = True     # the tiles' numbers standardised per map (each number less its mean over the map's
    #                            tiles, / its spread, within +-5): the raw ones hardly move (spread 0.03-0.11)
    threat: bool = True        # the group's threat numbers (net.threat: the nearest enemy group's distance, the
    #                            strength ratio, both shooter shares, the share in contact; net.fight: the shares
    #                            under fire and in melee, by the state's flags) added to its card before the kind and
    #                            target choices (linear layers from zero)
    group_tiles: bool = True   # the tiles under every group's centre (net.group_tiles: the tiles' numbers within 1.5
    #                            tiles of it, weighted by distance) added to its card - ours before the kind and target
    #                            choices, theirs as the targets' keys (a linear layer from zero): the council's path
    #                            to the tiles was dead (the atlas: tiles zeroed, the kind's KL ~1e-7)
    task_group: bool = False   # the executor's task input carries the group card (the atlas: ignored, KL 0.003);
    #                            False: those columns are zeros (the input's width kept: older checkpoints load)

    def __post_init__(self):
        if self.tiles != "sectors":
            raise NotBuilt(f"general tiles {self.tiles!r}: only 'sectors' is built (the map CNN's tiles come later)")
        if self.effects != "off":
            raise NotBuilt(f"general effects {self.effects!r}: only 'off' is built (the effects' analysis comes later)")
        if self.zone != "place":
            raise NotBuilt(f"general zone {self.zone!r}: only 'place' is built (the map's zone layers come later)")
        if self.deploy not in ("net", "script"):
            raise ValueError(f"general deploy {self.deploy!r}: 'net' or 'script'")
        assert self.card % self.council_heads == 0, "the card's width must split into the heads"
        if not 1 <= self.kinds <= len(MISSION_KINDS):
            raise ValueError(f"general kinds {self.kinds}: 1..{len(MISSION_KINDS)} (config.MISSION_KINDS names them)")
        if self.kind_stick < 0:
            raise ValueError(f"general kind_stick {self.kind_stick}: >= 0")
        if not 0.0 <= self.shoot_share <= 1.0:
            raise ValueError(f"general shoot_share {self.shoot_share}: a share in [0, 1]")
        if self.target_grid < 1:
            raise ValueError(f"general target_grid {self.target_grid}: >= 1")
        if self.groups not in ("rule", "net"):
            raise ValueError(f"general groups {self.groups!r}: 'rule' or 'net'")
        if self.tile_feats not in ("v1", "v2"):
            raise ValueError(f"general tile_feats {self.tile_feats!r}: 'v1' or 'v2'")
        if self.group_r <= 0 or self.lord_join_m < 0:
            raise ValueError("general group_r > 0, lord_join_m >= 0")

    @property
    def task_dim(self):
        """The executor's task input a unit: its group card (the places' mix), its mission kind's card, its group's
        target (direction cos, sin from the unit's facing, distance, "none")."""
        return 2 * self.card + 4


def to_dict(cfg):
    return asdict(cfg)


# a checkpoint saved before the atlas's repairs has none of these fields: it plays as it was trained (these values);
# the run's options switch them (run.general_config: they are rules, not the shape - the new weights load fresh)
LEGACY = {"groups": "net", "split_kinds": False, "tile_feats": "v1", "tile_norm": False, "threat": False,
          "task_group": True}


_WARNED = []


def of_dict(d, warn=True):
    """A GeneralConfig of a checkpoint's dict (unknown fields ignored, missing ones default; a saved general without
    the atlas's repairs keeps its old behaviour: LEGACY, with a warning)."""
    names = {f.name for f in fields(GeneralConfig)}
    d = dict(d or {})
    old = [k for k in LEGACY if d and k not in d]
    for k in old:
        d[k] = LEGACY[k]
    if old and warn and not _WARNED:
        _WARNED.append(1)
        print("general: a checkpoint saved before the atlas's repairs - its old behaviour kept ("
              + ", ".join(f"{k}={LEGACY[k]}" for k in old) + "); the options --general-groups, --general-split-kinds, "
              "--general-tile-feats, --general-tile-norm, --general-threat, --general-task-group switch them", flush=True)
    return GeneralConfig(**{k: v for k, v in d.items() if k in names})

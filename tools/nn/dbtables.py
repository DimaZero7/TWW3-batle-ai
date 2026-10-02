"""Rows of the game's database tables (data/db.pack) that decide a fight.

A DB table is binary with no schema inside: a header (version), the number of
rows, then the rows one after another, each a fixed sequence of fields. The
field sequences below were worked out by hand on v9.0.1 build 50381
(30.09.2026): each layout reads its whole table to the last byte, every text
field is readable text and every flag is 0 or 1. A game update that changes a
table breaks this check loudly (`decode` raises), it never reads shifted numbers.

Field types: i int32, f float32, b flag (1 byte), s text (u16 length + UTF-8),
o optional text (flag, then text when the flag is 1).

Field names: the land_units and main_units names are the game's own (an earlier
decode with a schema, local archive research/evidence/units/empire-20260926);
names in INFERRED are ours, worked out from the values (docs/*/game/database.md);
fields whose meaning is not known are called f<index>.
"""
import struct

TYPES = "ifbso"


def _layout(text):
    """'name:t name:t ...' -> [(name, type)]."""
    out = []
    for item in text.split():
        name, t = item.split(":")
        assert t in TYPES, item
        out.append((name, t))
    return out


LAYOUTS = {
    "land_units": (54, _layout("""
        accuracy:i armour:s campaign_action_points:i category:s charge_bonus:i class:s
        historical_description_text:s key:s man_animation:o man_entity:s melee_attack:i melee_defence:i
        morale:i bonus_hit_points:i mount:o num_mounts:i primary_melee_weapon:s primary_missile_weapon:o
        rank_depth:i shield:s short_description_text:s spacing:s training_level:s articulated_record:o
        engine:o is_male:b visibility_spotting_range_min:f visibility_spotting_range_max:f attribute_group:o
        spot_dist_tree:i spot_dist_scrub:i reload:i selection_vo:s selected_vo_secondary:s
        selected_vo_tertiary:s hiding_scalar:f ground_stat_effect_group:o secondary_ammo:i primary_ammo:i
        damage_mod_flame:i damage_mod_magic:i num_engines:i damage_mod_physical:i damage_mod_missile:i
        damage_mod_all:i ai_usage_group:s can_skirmish:b can_brace:b mounted_draughts:b sync_locomotion:b
        capture_tier:s key_building_capture_tier:s score_capture_tier:s harmony_amplification_type:o
        first_person:o ar_unit_category:o healing_power:f spell_mastery:f infinite_secondary_ammo:b
        default_guard_mode:b""")),
    "main_units": (7, _layout("""
        additional_building_requirement:o campaign_cap:i caste:s create_time:i is_naval:b land_unit:s
        num_men:i multiplayer_cap:i multiplayer_cost:i naval_unit:o num_ships:i min_men_per_ship:i
        max_men_per_ship:i recruitment_cost:i unit:s upkeep_cost:i weight:o resource_requirement:o
        special_edition_mask:i in_encyclopedia:b audio_voiceover_culture:s ui_unit_group_land:s tier:i
        is_high_threat:b porthole_camera:s mount:o use_hitpoints_in_campaign:b porthole_composite_scene:o
        melee_cp:f missile_cp:f can_siege:b audio_voiceover_culture_override:s
        restrict_xp_gain_in_campaign:b audio_voiceover_actor_group:s food_cost:i has_spoken_vo:b
        is_monstrous:b multiplayer_qb_cap:i vo_is_dragon:b vo_is_dinosaur:b optional_ui_element:o
        barrier_health:i can_be_bribed:b point_allowance_weight:i is_renown:b""")),
    "battle_entities": (39, _layout("""
        key:s type:s walk_speed:f run_speed:f acceleration:f deceleration:f charge_speed:f
        f07:f f08:f f09:f radius:f collision_shape:s f12:f mass:f height:f f15:f f16:f hit_points:i
        f18:b f19:b f20:f f21:f f22:f size:s f24:f f25:s f26:s f27:f f28:s f29:i f30:i f31:f f32:b
        f33:f f34:f f35:i f36:o f37:o f38:f f39:f f40:f f41:b f42:o f43:b f44:b f45:o f46:f f47:f
        f48:i f49:s f50:b f51:f f52:f f53:f f54:f f55:o f56:b f57:b""")),
    "melee_weapons": (25, _layout("""
        bonus_v_large:i bonus_v_infantry:i key:s damage:i ap_damage:i f05:f f06:s f07:o
        splash_attack_target_size:o splash_attack_max_attacks:i splash_attack_power_multiplier:f
        f11:f f12:f f13:b contact_phase:o f15:i f16:i melee_attack_interval:f f18:o f19:b""")),
    "missile_weapons": (12, _layout("key:s f01:b default_projectile:s f03:o f04:b f05:b")),
    "projectiles": (53, _layout("""
        key:s category:s shot_type:s explosion_type:o spin_type:s projectile_number:i trajectory:s
        effective_range:i minimum_range:i max_elevation:i muzzle_velocity:f f11:f f12:f damage:i
        ap_damage:i f15:b f16:b f17:f base_reload_time:f calibration_distance:f calibration_area:f
        f21:f f22:f display:o f24:o f25:o f26:f f27:b contact_stat_effect:o f29:f f30:i f31:f f32:f
        homing_params:o f34:o f35:f f36:b f37:b f38:b f39:i penetration:o f41:i f42:b f43:b f44:b
        f45:b shots_per_volley:i f47:b f48:b f49:b f50:b f51:f f52:o f53:b f54:f f55:f f56:o
        f57:s""")),
    "unit_armour_types": (6, _layout("armour_value:i key:s f02:b material:s")),
    "unit_shield_types": (6, _layout("key:s f01:i f02:i missile_block_chance:i material:o")),
    "unit_attributes_to_groups_junctions": (2, _layout("attribute:s f01:b attribute_group:s")),
    "land_units_to_unit_abilites_junctions": (1, _layout("ability:s land_unit:s culture:s")),
    # Abilities (tools/nn/abilities.py), worked out 01.10.2026 on the same build.
    "special_ability_to_special_ability_phase_junctions": (2, _layout(
        "order:i special_ability:s phase:s target_self:b target_friends:b target_enemies:b")),
    "special_ability_phase_stat_effects": (3, _layout("phase:s value:f stat:s how:s")),
    "special_ability_phase_attribute_effects": (1, _layout("attribute:s phase:s effect:s")),
    # When the game fires a timed passive by itself (02.10.2026, the same build; no version marker):
    # Strength of the Penitent - losing_melee_combat; and when an effect switches off (out_of_melee,
    # morale_is_lower_than_half_of_base_morale for Frenzy).
    "special_ability_to_recharge_contexts": (None, _layout("context:s special_ability:s")),
    "special_ability_to_auto_deactivate_flags": (None, _layout("flag:s special_ability:s")),
}
# Tables too wide to decode whole (unit_special_abilities: version 74, ~80 fields, rows of 200-500
# bytes): only the fields right after the key, which sit at fixed offsets (decode_prefix).
PREFIXES = {
    "unit_special_abilities": (74, _layout("""
        key:s active_time:f recharge_time:f num_uses:i effect_range:f targets_own:b
        num_effected_friendly_units:i num_effected_enemy_units:i f08:b f09:f""")),
}
# Names we gave from the values, not from the game (why: docs/*/game/database.md).
INFERRED = {
    "battle_entities": {"type", "acceleration", "deceleration", "charge_speed", "radius", "collision_shape",
                        "height", "size"},
    "melee_weapons": {"bonus_v_large", "bonus_v_infantry", "splash_attack_target_size",
                      "splash_attack_max_attacks", "splash_attack_power_multiplier", "contact_phase",
                      "melee_attack_interval"},
    "projectiles": {"category", "shot_type", "explosion_type", "spin_type", "projectile_number", "trajectory",
                    "minimum_range", "max_elevation", "muzzle_velocity", "calibration_distance",
                    "calibration_area", "display", "contact_stat_effect", "homing_params", "penetration",
                    "shots_per_volley"},
    "unit_armour_types": {"material"},
    "unit_shield_types": {"missile_block_chance", "material"},
    "special_ability_to_special_ability_phase_junctions": {"order", "target_self", "target_friends",
                                                           "target_enemies"},
    "special_ability_phase_stat_effects": {"phase", "value", "stat", "how"},
    "special_ability_phase_attribute_effects": {"attribute", "phase", "effect"},
    "special_ability_to_recharge_contexts": {"context", "special_ability"},
    "special_ability_to_auto_deactivate_flags": {"flag", "special_ability"},
    "unit_special_abilities": {"active_time", "recharge_time", "num_uses", "effect_range", "targets_own",
                               "num_effected_friendly_units", "num_effected_enemy_units"},
}


def entry_name(table):
    return rf"db\{table}_tables\data__"


def header(b):
    """(version or None, offset of the row count)."""
    p, version = 0, None
    if b[:4] == b"\xfd\xfe\xfc\xff":                    # GUID marker: u16 length + UTF-16
        p = 6 + 2 * struct.unpack_from("<H", b, 4)[0]
    if b[p:p + 4] == b"\xfc\xfd\xfe\xff":               # version marker
        version = struct.unpack_from("<i", b, p + 4)[0]
        p += 8
    return version, p + 1                               # one more byte before the count


def _text(b, p):
    n = struct.unpack_from("<H", b, p)[0]
    raw = b[p + 2:p + 2 + n]
    if len(raw) != n:
        raise ValueError(f"text runs past the end at {p}")
    text = raw.decode("utf-8")
    if not text.isprintable():
        raise ValueError(f"not text at {p}: {text[:20]!r}")
    return text, p + 2 + n


def _flag(b, p):
    if b[p] not in (0, 1):
        raise ValueError(f"flag is {b[p]} at {p}")
    return bool(b[p]), p + 1


def read_field(b, p, t):
    if t == "i":
        return struct.unpack_from("<i", b, p)[0], p + 4
    if t == "f":
        return round(struct.unpack_from("<f", b, p)[0], 5), p + 4
    if t == "b":
        return _flag(b, p)
    if t == "s":
        return _text(b, p)
    if t == "o":
        present, p = _flag(b, p)
        return _text(b, p) if present else (None, p)
    raise KeyError(t)


def decode(b, table, layout=None):
    """Every row of a table as a dict; raises when the layout does not fit the bytes."""
    version, fields = layout or LAYOUTS[table]
    got, p = header(b)
    if got != version:
        raise ValueError(f"{table}: version {got}, the layout is for {version}")
    rows = struct.unpack_from("<I", b, p)[0]
    p += 4
    out = []
    for r in range(rows):
        row = {}
        for name, t in fields:
            try:
                row[name], p = read_field(b, p, t)
            except (ValueError, KeyError, struct.error, UnicodeDecodeError, IndexError) as e:
                raise ValueError(f"{table}: row {r}, field {name}: {e}") from None
        out.append(row)
    if p != len(b):
        raise ValueError(f"{table}: {len(b) - p} bytes left after {rows} rows")
    return out


def decode_prefix(b, table, keys, layout=None):
    """{key: first fields of its row} for a table read only up to PREFIXES[table] (the key first).

    The row of a key is found where the key is written as text and the fields after it read as
    the layout says: flags 0 or 1, times and counts in a sane range (-1 = none). A key found in no
    row, or in two rows that both read sanely, raises: never a guess."""
    version, fields = layout or PREFIXES[table]
    got, _ = header(b)
    if got != version:
        raise ValueError(f"{table}: version {got}, the prefix is for {version}")
    out = {}
    for key in keys:
        raw = key.encode("utf-8")
        mark = struct.pack("<H", len(raw)) + raw
        rows, at = [], b.find(mark)
        while at >= 0:
            try:
                row, p = {}, at
                for name, t in fields:
                    row[name], p = read_field(b, p, t)
                if _sane(row):
                    rows.append(row)
            except (ValueError, struct.error, UnicodeDecodeError, IndexError):
                pass
            at = b.find(mark, at + 1)
        if len(rows) != 1:
            raise ValueError(f"{table}: {len(rows)} rows read for {key}")
        out[key] = rows[0]
    return out


def _sane(row):
    """Numbers of a real row (not a key written inside another row): -1 or 0..10000."""
    for name, v in row.items():
        if isinstance(v, float) and not (v == -1 or 0 <= v <= 10000):
            return False
        if isinstance(v, int) and not isinstance(v, bool) and not -1 <= v <= 10000:
            return False
    return True


def read_tables(pack, tables):
    """{table: [row dicts]} from db.pack (needs Python 3.14+: compression.zstd)."""
    from tools.nn.gamedb import read_entries
    names = {entry_name(t): t for t in tables}
    raw = read_entries(pack, set(names))
    missing = set(names) - set(raw)
    assert not missing, f"not in {pack}: {sorted(missing)}"
    return {names[n]: decode(b, names[n]) for n, b in raw.items()}

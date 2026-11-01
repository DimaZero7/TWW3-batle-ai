"""The memreader probes' config (docs/ru/game/memreader.md): the memreader Plus pack pinned by SHA-256
(config/memreader.json) and the game code sites and structure offsets from a file OUTSIDE Git (build/mem/sites.json,
or the environment variable BAI_MEM_SITES) - the repository keeps only their names.

    dependency()           {pack, workshop_id, sha256}: the pack the launcher copies next to ours
    lua_config(path=None)  {sites = {name: {hi, lo, sig, expect, pattern}}, layout = {name: offset}, snapshot_max}
                           (the men's layout - src/apps/mem/entities.lua - and the morale's - src/apps/mem/morale.lua -
                           when the file has them):
                           the probe's config.mem (a relative address split hi * 2^20 + lo: the game's Lua numbers are
                           32-bit floats, exact only to 2^24)
"""
import hashlib
import json
import os
from pathlib import Path

from tools import config as project

PIN = project.CONFIG_DIR / "memreader.json"
SITES = ("proj_update", "ammo_left", "battle_ref", "battle_global")
LAYOUT = ("mgr_active", "p_shooter", "p_record", "p_state", "p_prev", "p_pos", "p_vel", "p_time", "s_feet",
          "r_category", "b_world", "w_battle", "w_proj", "i_unit", "u_ammo", "u_battle")
# the men of a unit (src/apps/mem/entities.lua): all of MEN_LAYOUT or none (a sites file without u_men: no men);
# MEN_OPTIONAL (the target's path) and the site dir_table (the facing vectors) only when the file has them
MEN_LAYOUT = ("u_men", "w_tick", "m_unit", "m_live", "m_pos", "m_vel", "m_face", "m_hp", "m_anim",
              "m_mode", "m_react", "m_mark", "m_fit", "m_flags", "m_brace", "m_speed")
MEN_OPTIONAL = ("m_target", "m_target_b", "m_carrier", "m_charge")
# the morale of a unit (src/apps/mem/morale.lua): all of MORALE_LAYOUT or none (a sites file without u_morale: no
# morale); u_fields {name: [offset, type]} the unit's own fields read beside it, when the file has them
MORALE_LAYOUT = ("u_morale", "o_unit", "o_state", "o_points", "o_target", "o_fx_n", "o_fx", "n_next", "n_id", "n_val",
                 "o_timed", "t_stride", "o_contacts", "c_id", "c_side", "o_surprise", "o_waver", "o_charge", "o_rally",
                 "o_bytes")
OPTIONAL_SITES = ("dir_table",)
SNAPSHOT_MAX = 300


def pin():
    return json.loads(PIN.read_text(encoding="utf-8"))


def dependency():
    p = pin()
    return {"pack": p["pack_name"], "workshop_id": p["workshop_id"], "sha256": p["sha256"]}


def sites_path():
    env = os.environ.get("BAI_MEM_SITES")
    return Path(env) if env else project.ROOT / pin()["sites_file"]


def _int(v):
    return int(v, 0) if isinstance(v, str) else int(v)


def split(rva):
    """(hi, lo) with rva = hi * 2^20 + lo, both exact in a 32-bit float."""
    rva = _int(rva)
    assert 0 <= rva < (1 << 44), rva
    return rva >> 20, rva & 0xFFFFF


def lua_config(path=None, exe=None):
    """The probe's config.mem from the sites file; checks every name the Lua side needs and (exe given) the game's
    build by the file's SHA-256."""
    path = Path(path or sites_path())
    if not path.is_file():
        raise FileNotFoundError(f"{path}: the sites file (outside Git) is missing - see docs/ru/game/memreader.md")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if exe is not None and raw.get("exe_sha256"):
        h = hashlib.sha256()
        with open(exe, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        if h.hexdigest() != raw["exe_sha256"]:
            raise ValueError(f"{exe}: another game build than the sites file's (exe_sha256) - redo the sites")
    sites = {}
    for name in SITES + tuple(n for n in OPTIONAL_SITES if n in raw["sites"]):
        s = raw["sites"][name]
        hi, lo = split(s["rva"])
        sites[name] = {"hi": hi, "lo": lo, "sig": s["sig"], "expect": s.get("expect", ""),
                       **({"pattern": s["pattern"]} if s.get("pattern") else {})}
    layout = {name: _int(raw["layout"][name]) for name in LAYOUT}
    if "u_men" in raw["layout"]:
        layout.update({name: _int(raw["layout"][name]) for name in MEN_LAYOUT})
        layout.update({name: _int(raw["layout"][name]) for name in MEN_OPTIONAL if name in raw["layout"]})
    if "u_morale" in raw["layout"]:
        layout.update({name: _int(raw["layout"][name]) for name in MORALE_LAYOUT})
        fields = raw["layout"].get("u_fields") or {}
        layout["u_fields"] = {name: [_int(off), str(kind)] for name, (off, kind) in fields.items()}
    if "form" in raw["layout"]:
        # the unit's formation and way (src/apps/mem/order.lua): {name: offset or a list of offsets}, as they are
        layout["form"] = {name: ([_int(x) for x in v] if isinstance(v, list) else _int(v))
                          for name, v in raw["layout"]["form"].items()}
    return {"sites": sites, "layout": layout, "snapshot_max": int(raw.get("snapshot_max", SNAPSHOT_MAX))}

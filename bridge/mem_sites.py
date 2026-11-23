"""The memreader probes' config (bridge/docs/apps/memreader.md): the memreader Plus pack pinned by SHA-256
(bridge/config/memreader.json) and the game code sites and structure offsets of the game's exe (memreader.json sites_file:
reverse/mem/sites_exe1103.json, the exe of 03.11.2026; the environment variable BAI_MEM_SITES only to try another file).

    dependency()           {pack, workshop_id, sha256}: the pack the launcher copies next to ours
    lua_config(path=None)  {sites = {name: {hi, lo, sig, expect, pattern}}, layout = {name: offset}, snapshot_max}
                           (the men's layout - bridge/lua/apps/mem/entities.lua - and the morale's - bridge/lua/apps/mem/morale.lua -
                           when the file has them):
                           the probe's config.mem (a relative address split hi * 2^20 + lo: the game's Lua numbers are
                           32-bit floats, exact only to 2^24)
    no_draw_config(path=None, exe=None)
                           the no-draw mode's config.no_draw (bridge/lua/apps/battle/no_draw.lua; bridge.build --no-draw):
                           {site = {name, hi, lo, at, pattern, before, after} of the site no_draw | None, why, frames =
                           the read-only frame counter (site frame_count) | None}; no site, or an exe of another build
                           than the file's: site None and why - the battle then goes on drawn (event v2_refused)
"""
import functools
import hashlib
import json
import os
from pathlib import Path

from bridge import config as project

PIN = project.CONFIG_DIR / "memreader.json"
SITES = ("proj_update", "ammo_left", "battle_ref", "battle_global")
LAYOUT = ("mgr_active", "p_shooter", "p_record", "p_state", "p_prev", "p_pos", "p_vel", "p_time", "s_feet",
          "r_category", "b_world", "w_battle", "w_proj", "i_unit", "u_ammo", "u_battle")
# the men of a unit (bridge/lua/apps/mem/entities.lua): all of MEN_LAYOUT or none (a sites file without u_men: no men);
# MEN_OPTIONAL (the target's path) and the site dir_table (the facing vectors) only when the file has them
MEN_LAYOUT = ("u_men", "w_tick", "m_unit", "m_live", "m_pos", "m_vel", "m_face", "m_hp", "m_anim",
              "m_mode", "m_react", "m_mark", "m_fit", "m_flags", "m_brace", "m_speed")
MEN_OPTIONAL = ("m_target", "m_target_b", "m_carrier", "m_charge")
# the morale of a unit (bridge/lua/apps/mem/morale.lua): all of MORALE_LAYOUT or none (a sites file without u_morale: no
# morale); u_fields {name: [offset, type]} the unit's own fields read beside it, when the file has them
MORALE_LAYOUT = ("u_morale", "o_unit", "o_state", "o_points", "o_target", "o_fx_n", "o_fx", "n_next", "n_id", "n_val",
                 "o_timed", "t_stride", "o_contacts", "c_id", "c_side", "o_surprise", "o_waver", "o_charge", "o_rally",
                 "o_bytes")
OPTIONAL_SITES = ("dir_table",)
SNAPSHOT_MAX = 300
# the no-draw mode (bridge/config/memreader.json: the ONE code patch allowed): the site's 5 bytes must be a call (e8) and
# the patch exactly the 5-byte nop, whatever the sites file says - another patch needs a new review, not a new file
NO_DRAW_SITE, FRAMES_SITE = "no_draw", "frame_count"
NOP5 = "0f 1f 44 00 00"


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


@functools.lru_cache(maxsize=4)
def exe_sha256(exe):
    h = hashlib.sha256()
    with open(exe, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def lua_config(path=None, exe=None):
    """The probe's config.mem from the sites file; checks every name the Lua side needs and (exe given) the game's
    build by the file's SHA-256."""
    path = Path(path or sites_path())
    if not path.is_file():
        raise FileNotFoundError(f"{path}: the sites file is missing - see bridge/docs/apps/memreader.md")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if exe is not None and raw.get("exe_sha256"):
        if exe_sha256(exe) != raw["exe_sha256"]:
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
        # the unit's formation and way (bridge/lua/apps/mem/order.lua): {name: offset or a list of offsets}, as they are
        layout["form"] = {name: ([_int(x) for x in v] if isinstance(v, list) else _int(v))
                          for name, v in raw["layout"]["form"].items()}
    return {"sites": sites, "layout": layout, "snapshot_max": int(raw.get("snapshot_max", SNAPSHOT_MAX))}


def _hex(text):
    return " ".join(str(text).lower().split())


def no_draw_config(path=None, exe=None):
    """config.no_draw without delay_ms / safety_ms (bridge/build.py adds them): the site no_draw of the sites file, or
    site None and why (the battle goes on drawn: the Lua refuses with no_site). A malformed site raises: the file needs a
    fix, not a battle."""
    path = Path(path or sites_path())
    if not path.is_file():
        return {"site": None, "why": f"no sites file {path.name}", "frames": None}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if exe is not None and raw.get("exe_sha256") and exe_sha256(exe) != raw["exe_sha256"]:
        return {"site": None, "why": f"the game's exe is another build than {path.name}'s (exe_sha256)", "frames": None}
    s = raw["sites"].get(NO_DRAW_SITE)
    if not s:
        return {"site": None, "why": f"{path.name} has no site {NO_DRAW_SITE}", "frames": None}
    before, after, pattern, at = _hex(s["expect"]), _hex(s["patch"]), _hex(s["pattern"]), int(s["pattern_at"])
    if len(before.split()) != 5 or before.split()[0] != "e8":
        raise ValueError(f"{path.name} {NO_DRAW_SITE}: expect must be the 5 bytes of a call (e8 ..), not {before!r}")
    if after != NOP5:
        raise ValueError(f"{path.name} {NO_DRAW_SITE}: the patch must be the 5-byte nop {NOP5!r} (memreader.json)")
    toks = pattern.split()
    if not (0 <= at <= len(toks) - 5) or any(t != "??" and t != b for t, b in zip(toks[at:at + 5], before.split())):
        raise ValueError(f"{path.name} {NO_DRAW_SITE}: pattern_at {at} does not point at the expected bytes")
    hi, lo = split(_int(s["rva"]) - at)
    site = {"name": NO_DRAW_SITE, "hi": hi, "lo": lo, "at": at, "pattern": pattern, "before": before, "after": after}
    frames = None
    f = raw["sites"].get(FRAMES_SITE)
    if f and f.get("ins_rva") and f.get("ins"):
        fhi, flo = split(f["rva"])
        ihi, ilo = split(f["ins_rva"])
        frames = {"hi": fhi, "lo": flo, "ins_hi": ihi, "ins_lo": ilo, "ins": _hex(f["ins"])}
    return {"site": site, "why": None, "frames": frames}

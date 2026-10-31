"""The memreader probes' config (docs/ru/game/memreader.md): the memreader Plus pack pinned by SHA-256
(config/memreader.json) and the game code sites and structure offsets from a file OUTSIDE Git (build/mem/sites.json,
or the environment variable BAI_MEM_SITES) - the repository keeps only their names.

    dependency()           {pack, workshop_id, sha256}: the pack the launcher copies next to ours
    lua_config(path=None)  {sites = {name: {hi, lo, sig, expect, pattern}}, layout = {name: offset}, snapshot_max}:
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
    for name in SITES:
        s = raw["sites"][name]
        hi, lo = split(s["rva"])
        sites[name] = {"hi": hi, "lo": lo, "sig": s["sig"], "expect": s.get("expect", ""),
                       **({"pattern": s["pattern"]} if s.get("pattern") else {})}
    layout = {name: _int(raw["layout"][name]) for name in LAYOUT}
    return {"sites": sites, "layout": layout, "snapshot_max": int(raw.get("snapshot_max", SNAPSHOT_MAX))}

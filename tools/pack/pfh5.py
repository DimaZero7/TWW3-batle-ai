"""Deterministic, uncompressed PFH5 mod pack writer and reader.

Format reference: RPFM rpfm_lib/src/files/pack/pack_versions/pfh5.rs.
Header: magic, pack type 3 (Mod), dependency count/size, file count,
index size, timestamp 0. Entries are sorted, so equal inputs give equal bytes.
"""
import struct

HEADER = struct.Struct("<4s6I")
MAGIC = b"PFH5"
PACK_TYPE_MOD = 3


def _check_dependencies(dependencies):
    dependencies = tuple(dependencies)
    if len(set(dependencies)) != len(dependencies):
        raise ValueError("duplicate pack dependencies")
    for name in dependencies:
        if (not name.endswith(".pack") or not name.isascii() or ".." in name
                or any(c in name for c in "/\\\x00")):
            raise ValueError(f"invalid pack dependency: {name!r}")
    return dependencies


def pack_files(files, dependencies=()):
    """files: {'script\\\\battle\\\\...lua': bytes}. Returns pack bytes."""
    dependencies = _check_dependencies(dependencies)
    dependency_index = b"".join(n.encode("ascii") + b"\0" for n in dependencies)
    index, payload = bytearray(), bytearray()
    for name, content in sorted(files.items()):
        if not name.isascii() or ".." in name or name.startswith(("/", "\\")):
            raise ValueError("pack paths must be relative ASCII paths")
        index += struct.pack("<IB", len(content), 0) + name.encode("ascii") + b"\0"
        payload += content
    header = HEADER.pack(MAGIC, PACK_TYPE_MOD, len(dependencies), len(dependency_index),
                         len(files), len(index), 0)
    return header + dependency_index + bytes(index) + bytes(payload)


def read_pack(blob):
    """Parses a pack written by pack_files. Returns ({name: bytes}, dependencies)."""
    magic, kind, deps, dep_size, count, index_size, timestamp = HEADER.unpack_from(blob)
    if (magic, kind, timestamp) != (MAGIC, PACK_TYPE_MOD, 0):
        raise ValueError("unexpected pack header")
    start = HEADER.size
    dependency_index = blob[start:start + dep_size]
    if dependency_index.count(b"\0") != deps or (dep_size and not dependency_index.endswith(b"\0")):
        raise ValueError("invalid dependency index")
    dependencies = tuple(n.decode("ascii") for n in dependency_index.split(b"\0")[:deps])
    pos, data = start + dep_size, start + dep_size + index_size
    entries = {}
    for _ in range(count):
        size, compressed = struct.unpack_from("<IB", blob, pos)
        pos += 5
        end = blob.index(0, pos)
        name = blob[pos:end].decode("ascii")
        if compressed or name in entries or data + size > len(blob):
            raise ValueError("invalid pack entry")
        entries[name] = blob[data:data + size]
        pos, data = end + 1, data + size
    if pos != start + dep_size + index_size or data != len(blob):
        raise ValueError("pack length mismatch")
    return entries, dependencies

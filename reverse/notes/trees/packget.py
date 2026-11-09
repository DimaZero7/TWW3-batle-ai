# extract named entries from a pack into out dir (zstd if compressed, like tools/nn/gamedb)
import struct, sys, os
from compression import zstd
D = r"C:/Program Files (x86)/Steam/steamapps/common/Total War WARHAMMER III/data"
pack, out, names = sys.argv[1], sys.argv[2], set(n.lower() for n in sys.argv[3:])
with open(os.path.join(D, pack), "rb") as f:
    magic, kind, pc, ps, fc, fs, ts = struct.unpack("<4s6I", f.read(28))
    ext = 20 if kind & 0x100 else 0
    f.read(ext); f.read(ps); idx = f.read(fs)
    pos, off = 0, 28 + ext + ps + fs
    for _ in range(fc):
        size = struct.unpack_from("<I", idx, pos)[0]
        pos += 4 + (4 if kind & 0x40 else 0)
        comp = idx[pos]; pos += 1
        end = idx.index(0, pos); name = idx[pos:end].decode("latin1"); pos = end + 1
        if name.lower() in names:
            f.seek(off); data = f.read(size)
            data = zstd.decompress(data[4:]) if comp else data
            p = os.path.join(out, name.replace("\\", "_"))
            open(p, "wb").write(data); print(p, len(data))
        off += size

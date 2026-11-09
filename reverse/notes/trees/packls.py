# list entries of PFH5 packs matching a regex (index only)
import struct, sys, re, glob, os
D = r"C:/Program Files (x86)/Steam/steamapps/common/Total War WARHAMMER III/data"
pat = re.compile(sys.argv[1], re.I)
for pack in sorted(glob.glob(D + "/*.pack")):
    with open(pack, "rb") as f:
        hdr = f.read(28)
        if hdr[:4] != b"PFH5": continue
        magic, kind, pc, ps, fc, fs, ts = struct.unpack("<4s6I", hdr)
        ext = 20 if kind & 0x100 else 0
        f.read(ext); f.read(ps); idx = f.read(fs)
        pos, off = 0, 28 + ext + ps + fs
        for _ in range(fc):
            size = struct.unpack_from("<I", idx, pos)[0]
            pos += 4 + (4 if kind & 0x40 else 0)
            comp = idx[pos]; pos += 1
            end = idx.index(0, pos); name = idx[pos:end].decode("latin1"); pos = end + 1
            if pat.search(name): print(os.path.basename(pack), name, size, comp)
            off += size

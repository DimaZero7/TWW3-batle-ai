# Who writes the morale effect name table 0x14406de00 (+ id*16): RIP-relative references into it, and nearby string literals.
import sys, numpy as np
sys.path.insert(0, r"reverse/tools")
import xref, tdis
T0, T1 = 0x14406de00, 0x14406de00 + 0x60 * 16
for secname in xref.secs:
    s = xref.secs[secname]
    if not (s.Characteristics & 0x20000000):   # executable
        continue
    buf = np.frombuffer(xref.data[s.PointerToRawData:s.PointerToRawData + s.SizeOfRawData], dtype=np.uint8)
    n = len(buf) - 4
    d = (buf[:n].astype(np.int64) | (buf[1:n+1].astype(np.int64) << 8) | (buf[2:n+2].astype(np.int64) << 16) | (buf[3:n+3].astype(np.int64) << 24))
    d = np.where(d >= 2**31, d - 2**32, d)
    base = xref.IB + s.VirtualAddress
    for tail in (0, 1, 4):
        tgt = base + np.arange(n) + 4 + tail + d
        idx = np.nonzero((tgt >= T0) & (tgt < T1))[0]
        for i in idx[:400]:
            print(secname, tail, hex(base + i), hex(int(tgt[i])), (int(tgt[i]) - T0) // 16, (int(tgt[i]) - T0) % 16)

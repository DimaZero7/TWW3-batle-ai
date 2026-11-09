"""vtrefs.py LO HI : rip-relative 'lea reg, [rip+X]' in the code with X in [LO, HI) (vtable starts written by
constructors); prints X and the referencing instruction addresses."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'tools'))
from xref import secs, data, IB
lo, hi = int(sys.argv[1], 16), int(sys.argv[2], 16)
s = secs['.udata']
buf = np.frombuffer(data[s.PointerToRawData:s.PointerToRawData + s.SizeOfRawData], dtype=np.uint8)
n = len(buf) - 4
d = (buf[:n].astype(np.int64) | (buf[1:n + 1].astype(np.int64) << 8) | (buf[2:n + 2].astype(np.int64) << 16)
     | (buf[3:n + 3].astype(np.int64) << 24))
d = np.where(d >= 2 ** 31, d - 2 ** 32, d)
base = IB + s.VirtualAddress
tgt = base + np.arange(n) + 4 + d
idx = np.nonzero((tgt >= lo) & (tgt < hi))[0]
out = {}
for i in idx:
    # lea r64,[rip+disp]: 48/4c 8d modrm(00 xxx 101) disp32 -> disp at +3
    if i >= 3 and buf[i - 2] == 0x8d and buf[i - 3] in (0x48, 0x4c) and (buf[i - 1] & 0xc7) == 0x05:
        out.setdefault(int(tgt[i]), []).append(base + i - 3)
for t in sorted(out):
    print(hex(t), ' '.join(hex(x) for x in out[t][:6]), len(out[t]))

# the unit task classes (the stack unit task + 0x848): lea-referenced tables whose destructor (slot 0) lies in
# 0x143260000-0x143290000; prints the boolean slots +0x20..+0x58 (+0x38: the 'attack' flag 0x142f59498 asks), the targets'
# count / getter (+0xb8 / +0xc0) and who references the table (the constructors). From the project's root:
#   reverse/ghidra/venv/Scripts/python reverse/notes/morale/task_vtables.py
import struct
import sys

import numpy as np

sys.path.insert(0, r'reverse/tools')
from tdis import pe, data, IB, read  # noqa: E402

TRUE, FALSE = 0x1408d3d30, 0x1408d3d10        # mov al, 1 / xor al, al
x = [s for s in pe.sections if s.Name.startswith(b'.xcode')][0]
lo = IB + x.VirtualAddress
hi = lo + x.Misc_VirtualSize
s = [s for s in pe.sections if s.Name.startswith(b'.udata')][0]
buf = np.frombuffer(data[s.PointerToRawData:s.PointerToRawData + s.SizeOfRawData], dtype=np.uint8)
n = len(buf) - 4
d = (buf[:n].astype(np.int64) | (buf[1:n + 1].astype(np.int64) << 8) | (buf[2:n + 2].astype(np.int64) << 16)
     | (buf[3:n + 3].astype(np.int64) << 24))
d = np.where(d >= 2 ** 31, d - 2 ** 32, d)
base = IB + s.VirtualAddress
t = base + np.arange(n) + 4 + d
m = (t >= lo) & (t < hi) & (t % 8 == 0)
refs = {}
for i in np.nonzero(m)[0]:
    if i >= 3 and buf[i - 2] == 0x8d and buf[i - 3] in (0x48, 0x4c):
        refs.setdefault(int(t[i]), []).append(base + i - 3)


def q(v, o):
    b = read(v + o, 8)
    return struct.unpack('<Q', b)[0] if b and len(b) == 8 else 0


for v in sorted(refs):
    if not 0x143260000 <= q(v, 0) < 0x143290000:
        continue
    flags = ' '.join('T' if q(v, o) == TRUE else '.' if q(v, o) == FALSE else '?' for o in range(0x20, 0x60, 8))
    print('%x  %s  +b8 %x  +c0 %x  refs %s' % (v, flags, q(v, 0xb8), q(v, 0xc0), ' '.join('%x' % r for r in refs[v][:3])))

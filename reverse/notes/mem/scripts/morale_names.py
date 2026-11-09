# The morale effects' name table (FUN_1430458fc: name = table 0x14406de00 + id * 0x10) read from the exe on disk.
import sys, struct
sys.path.insert(0, r"reverse/tools")
import tdis
T = 0x14406de00
for i in range(0, 0x40):
    b = tdis.read(T + i * 16, 16)
    if b is None:
        print(i, 'unmapped'); break
    a, b2 = struct.unpack('<QQ', b)
    s = tdis.cstr(a) if 0x140000000 <= a < 0x150000000 else None
    s2 = tdis.cstr(b2) if 0x140000000 <= b2 < 0x150000000 else None
    print(hex(i), hex(a), hex(b2), s, s2)

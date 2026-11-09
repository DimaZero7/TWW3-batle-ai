import sys, re, struct, collections
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tdis import pe, data, IB
s = [x for x in pe.sections if x.Name.rstrip(b'\0') == b'.udata'][0]
buf = data[s.PointerToRawData:s.PointerToRawData + s.SizeOfRawData]
base = IB + s.VirtualAddress
cnt = collections.Counter()
# mov r64, [rip+disp32] : (48|4c) 8b modrm(mod=00, rm=101) disp32  then  (48|49|4c|4d) 8b modrm(mod=10) 40 04 00 00
pat = re.compile(rb'[\x48\x4c]\x8b([\x05\x0d\x15\x1d\x25\x2d\x35\x3d])(....)[\x48\x49\x4c\x4d]\x8b[\x80-\xbf]\x40\x04\x00\x00', re.S)
for m in pat.finditer(buf):
    disp = struct.unpack('<i', m.group(2))[0]
    ip = base + m.start() + 7
    cnt[ip + disp] += 1
for g, n in cnt.most_common(15):
    print(hex(g), n)
print('sites:')
for m in pat.finditer(buf):
    disp = struct.unpack('<i', m.group(2))[0]
    ip = base + m.start() + 7
    if ip + disp == 0x144480d10: print(hex(base + m.start()))

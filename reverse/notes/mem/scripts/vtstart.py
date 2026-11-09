"""vtstart.py ADDR... : for a slot address inside a vtable, walk back to the vtable start (previous qword not a code
pointer) and print the start, the slot index of ADDR, and the slots 0..N as idx:target."""
import sys, os, struct
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'tools'))
from tdis import read, pe, IB
code = [(IB + s.VirtualAddress, IB + s.VirtualAddress + s.Misc_VirtualSize) for s in pe.sections
        if s.Characteristics & 0x20000000]
def is_code(v): return any(a <= v < b for a, b in code)
def q(a): return struct.unpack('<Q', read(a, 8))[0]
n = int(os.environ.get('N', '0'))
for x in sys.argv[1:]:
    a = int(x, 16); s = a
    while is_code(q(s - 8)): s -= 8
    print('slot %x: vtable %x idx 0x%x  col %x' % (a, s, a - s, q(s - 8)))
    for i in range(n):
        v = q(s + 8 * i)
        if not is_code(v): break
        print('  %03x: %x' % (8 * i, v))

# show disasm around each xref of given string addresses
import sys
sys.path.insert(0, r'reverse/tools')
from tdis import dis
from xref import xrefs
ts=[int(x,16) for x in sys.argv[2:]]; back=int(sys.argv[1])
for t,v in xrefs(ts).items():
    print('==', hex(t), [hex(x) for x in v][:10])
    for r in v[:6]:
        print('-- ref', hex(r))
        for l in dis(r-back, back+0x30): print('  ', l)

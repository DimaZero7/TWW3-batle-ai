# find lea/mov references to a VA in code
import sys
sys.path.insert(0,'reverse/tools')
from xref import xrefs
ts=[int(x,16) for x in sys.argv[1:]]
for t,v in xrefs(ts).items(): print(hex(t),[hex(x) for x in v][:40])

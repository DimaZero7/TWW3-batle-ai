import sys, numpy as np, pefile
import os
# the exe the tables (pdata.pkl, idx_battle.pkl), the Ghidra project and the notes' addresses belong to - 15.10.2026;
# another exe (the live Steam one): RE_EXE=<path>
p = os.environ.get('RE_EXE') or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ghidra', 'exe1015', 'Warhammer3.exe')
pe = pefile.PE(p, fast_load=True)
data = open(p,'rb').read()
IB=0x140000000
secs = {s.Name.rstrip(b'\0').decode(): s for s in pe.sections}
def xrefs(targets, secname='.udata', tail=0):
    s = secs[secname]
    buf = np.frombuffer(data[s.PointerToRawData:s.PointerToRawData+s.SizeOfRawData], dtype=np.uint8)
    n = len(buf)-4
    # int32 at each offset
    d = (buf[:n].astype(np.int64) | (buf[1:n+1].astype(np.int64)<<8) | (buf[2:n+2].astype(np.int64)<<16) | (buf[3:n+3].astype(np.int64)<<24))
    d = np.where(d>=2**31, d-2**32, d)
    base = IB + s.VirtualAddress
    nextip = base + np.arange(n) + 4 + tail
    res = {}
    tgt = nextip + d
    for t in targets:
        idx = np.nonzero(tgt == t)[0]
        res[t] = [base+i for i in idx]
    return res
if __name__=='__main__':
    ts=[int(x,16) for x in sys.argv[1:]]
    for t,v in xrefs(ts).items():
        print(hex(t), [hex(x) for x in v])

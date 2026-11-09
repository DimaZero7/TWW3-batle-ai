import sys, os, re, capstone; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tdis import md, read
lo=int(sys.argv[1],16); hi=int(sys.argv[2],16); disp=int(sys.argv[3],16); flt=sys.argv[4] if len(sys.argv)>4 else None
code=read(lo, hi-lo)
for ins in md.disasm(code, lo):
    pass
# linear sweep may desync; use pdata functions instead
import pickle, bisect
rf=pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','ghidra','exe1015','pdata.pkl'),'rb'))
IB=0x140000000
for b,e in rf:
    if IB+b<lo or IB+b>=hi: continue
    for ins in md.disasm(read(IB+b,e-b), IB+b):
        for op in ins.operands:
            if op.type==capstone.x86.X86_OP_MEM and op.mem.disp==disp and op.mem.base!=capstone.x86.X86_REG_RSP and op.mem.base!=capstone.x86.X86_REG_RBP:
                if flt is None or re.search(flt, ins.mnemonic):
                    print('%x %s %s | f=%x'%(ins.address, ins.mnemonic, ins.op_str, IB+b))

# scan all functions for float writes/updates to [reg+disp] for given disps (byte prefilter, then capstone)
import sys, pickle, struct, capstone
sys.path.insert(0, r'reverse/tools')
from tdis import read, IB
rf=pickle.load(open(r'reverse/ghidra/exe1015/pdata.pkl','rb'))
disps=[int(a,16) for a in sys.argv[1].split(',')]
lo=int(sys.argv[2],16) if len(sys.argv)>2 else 0
hi=int(sys.argv[3],16) if len(sys.argv)>3 else 1<<62
pats=[struct.pack('<I',d) for d in disps]
md=capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64); md.detail=True
for b,e in rf:
    f=IB+b
    if f<lo or f>=hi: continue
    code=read(f,e-b)
    if not code or not any(p in code for p in pats): continue
    for ins in md.disasm(code,f):
        if not ins.operands: continue
        op0=ins.operands[0]
        if op0.type==capstone.x86.X86_OP_MEM and op0.mem.disp in disps and op0.mem.base not in (capstone.x86.X86_REG_RSP,capstone.x86.X86_REG_RBP,capstone.x86.X86_REG_RIP):
            m=ins.mnemonic
            if True:
                print('%x f=%x %s %s'%(ins.address,f,m,ins.op_str))

# all instructions in [lo,hi) with any of the given disps
import sys, pickle, capstone
sys.path.insert(0, r'reverse/tools')
from tdis import read, IB
rf=pickle.load(open(r'reverse/ghidra/exe1015/pdata.pkl','rb'))
lo=int(sys.argv[1],16); hi=int(sys.argv[2],16); ds=set(int(x,16) for x in sys.argv[3:])
md=capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64); md.detail=True
for b,e in rf:
    f=IB+b
    if f<lo or f>=hi: continue
    c=read(f,e-b)
    if not c: continue
    for ins in md.disasm(c, f):
        for op in ins.operands:
            if op.type==capstone.x86.X86_OP_MEM and op.mem.disp in ds and op.mem.base!=capstone.x86.X86_REG_RIP:
                print('%x f=%x %s %s'%(ins.address,f,ins.mnemonic,ins.op_str)); break

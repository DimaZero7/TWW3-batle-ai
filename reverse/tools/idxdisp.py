# memory operands with index register and disp in [a,b]
import sys, pickle, capstone
sys.path.insert(0, r'reverse/tools')
from tdis import read, IB
rf=pickle.load(open(r'reverse/ghidra/exe1015/pdata.pkl','rb'))
lo=int(sys.argv[1],16); hi=int(sys.argv[2],16); a=int(sys.argv[3],16); b=int(sys.argv[4],16)
md=capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64); md.detail=True
for B,E in rf:
    f=IB+B
    if f<lo or f>=hi: continue
    code=read(f,E-B)
    if not code: continue
    for ins in md.disasm(code,f):
        for op in ins.operands:
            if op.type==capstone.x86.X86_OP_MEM and a<=op.mem.disp<=b and op.mem.index!=0:
                print('%x f=%x %s %s'%(ins.address,f,ins.mnemonic,ins.op_str))

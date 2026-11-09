# find functions where [base+A] (dword) and [base+B] (qword) are read with same base reg within K instrs
import sys, pickle, capstone
sys.path.insert(0, r'reverse/tools')
from tdis import read, IB
rf=pickle.load(open(r'reverse/ghidra/exe1015/pdata.pkl','rb'))
lo=int(sys.argv[1],16); hi=int(sys.argv[2],16); A=int(sys.argv[3],16); Bd=int(sys.argv[4],16); K=12
md=capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64); md.detail=True
for b,e in rf:
    f=IB+b
    if f<lo or f>=hi: continue
    code=read(f,e-b)
    if not code: continue
    L=list(md.disasm(code,f))
    for i,ins in enumerate(L):
        for op in ins.operands:
            if op.type==capstone.x86.X86_OP_MEM and op.mem.disp==A and op.mem.index==0 and op.size==4:
                base=op.mem.base
                for j in L[i+1:i+K]:
                    if any(o.type==capstone.x86.X86_OP_MEM and o.mem.disp==Bd and o.mem.base==base and o.size==8 for o in j.operands):
                        print('%x f=%x %s | %x %s'%(ins.address,f,ins.mnemonic+' '+ins.op_str,j.address,j.mnemonic+' '+j.op_str)); break

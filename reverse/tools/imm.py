# instructions in [lo,hi) with an immediate operand == value (any mnemonic), optional text filter
import sys, pickle, capstone, re
sys.path.insert(0, r'reverse/tools')
from tdis import read, IB
rf=pickle.load(open(r'reverse/ghidra/exe1015/pdata.pkl','rb'))
lo=int(sys.argv[1],16); hi=int(sys.argv[2],16); val=int(sys.argv[3],16); pat=sys.argv[4] if len(sys.argv)>4 else ''
md=capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64); md.detail=True
for b,e in rf:
    f=IB+b
    if f<lo or f>=hi: continue
    c=read(f,e-b)
    if not c: continue
    for ins in md.disasm(c,f):
        if any(op.type==capstone.x86.X86_OP_IMM and op.imm==val for op in ins.operands):
            t=ins.mnemonic+' '+ins.op_str
            if re.search(pat,t): print('%x f=%x %s'%(ins.address,f,t))

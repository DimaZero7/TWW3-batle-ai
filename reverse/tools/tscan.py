# find instructions referencing [reg+disp] with given immediate (test/and/bt) anywhere
import sys, pickle, struct, capstone
sys.path.insert(0, r'reverse/tools')
from tdis import read, IB
rf=pickle.load(open(r'reverse/ghidra/exe1015/pdata.pkl','rb'))
specs=[(int(a.split(':')[0],16),int(a.split(':')[1],16)) for a in sys.argv[1].split(',')]
lo=int(sys.argv[2],16); hi=int(sys.argv[3],16)
md=capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64); md.detail=True
for b,e in rf:
    f=IB+b
    if f<lo or f>=hi: continue
    code=read(f,e-b)
    for ins in md.disasm(code,f):
        ops=ins.operands
        if len(ops)!=2: continue
        if ops[0].type==capstone.x86.X86_OP_MEM and ops[1].type==capstone.x86.X86_OP_IMM:
            for d,v in specs:
                if ops[0].mem.disp==d and ops[1].imm==v and ops[0].mem.base not in (capstone.x86.X86_REG_RSP,capstone.x86.X86_REG_RBP):
                    print('%x f=%x %s %s'%(ins.address,f,ins.mnemonic,ins.op_str))

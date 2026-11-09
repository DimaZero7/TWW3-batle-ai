import sys, pickle, capstone
sys.path.insert(0, r'reverse/tools')
from tdis import read, IB
rf=pickle.load(open(r'reverse/ghidra/exe1015/pdata.pkl','rb'))
lo=int(sys.argv[1],16); hi=int(sys.argv[2],16); imm=int(sys.argv[3],16)
md=capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64); md.detail=True
for b,e in rf:
    f=IB+b
    if f<lo or f>=hi: continue
    for x in md.disasm(read(f,e-b), f):
        if x.mnemonic=='cmp' and len(x.operands)==2 and x.operands[1].type==capstone.x86.X86_OP_IMM and x.operands[1].imm==imm and x.operands[0].type==capstone.x86.X86_OP_MEM and x.operands[0].mem.base not in (capstone.x86.X86_REG_RSP,capstone.x86.X86_REG_RIP,capstone.x86.X86_REG_RBP):
            print('%x %x %s %s'%(f,x.address,x.mnemonic,x.op_str))

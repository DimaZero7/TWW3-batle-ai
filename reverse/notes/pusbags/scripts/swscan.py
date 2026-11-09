import sys, pickle, capstone
sys.path.insert(0, r'reverse/tools')
from tdis import read, IB
rf=pickle.load(open(r'reverse/ghidra/exe1015/pdata.pkl','rb'))
lo=int(sys.argv[1],16); hi=int(sys.argv[2],16); mn=int(sys.argv[3],16); mx=int(sys.argv[4],16)
md=capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64); md.detail=True
for b,e in rf:
    f=IB+b
    if f<lo or f>=hi: continue
    ins=list(md.disasm(read(f,e-b), f))
    for i,x in enumerate(ins):
        if x.mnemonic=='cmp' and len(x.operands)==2 and x.operands[1].type==capstone.x86.X86_OP_IMM and mn<=x.operands[1].imm<=mx and x.operands[0].type==capstone.x86.X86_OP_REG:
            nxt=ins[i+1:i+14]
            if any(n.mnemonic=='ja' for n in nxt[:4]) and any(n.mnemonic=='jmp' and n.operands[0].type==capstone.x86.X86_OP_REG for n in nxt):
                print('%x %x cmp %s'%(f,x.address,x.op_str))

# build index: disp -> [(insn_addr, func, text)] for functions in [lo,hi)
import sys, os, pickle, capstone
sys.path.insert(0, r'reverse/tools')
from tdis import read, IB
rf=pickle.load(open(r'reverse/ghidra/exe1015/pdata.pkl','rb'))
lo=int(sys.argv[1],16); hi=int(sys.argv[2],16); out=sys.argv[3]
md=capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64); md.detail=True
idx={}; calls={}
for b,e in rf:
    f=IB+b
    if f<lo or f>=hi: continue
    for ins in md.disasm(read(f,e-b), f):
        for op in ins.operands:
            if op.type==capstone.x86.X86_OP_MEM and op.mem.base not in (capstone.x86.X86_REG_RSP,capstone.x86.X86_REG_RIP) and op.mem.disp>=0x40:
                idx.setdefault(op.mem.disp,[]).append((ins.address,f,ins.mnemonic+' '+ins.op_str))
        if ins.mnemonic=='call' and ins.operands[0].type==capstone.x86.X86_OP_IMM:
            calls.setdefault(ins.operands[0].imm,[]).append((ins.address,f))
pickle.dump((idx,calls),open(out,'wb'))
print(len(idx))

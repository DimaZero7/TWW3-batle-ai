# find float/int loads from the entity record (entity+0x2d8 -> battle_entities row): offset -> functions
import sys, pickle, capstone, re
sys.path.insert(0, r'reverse/tools')
from tdis import read, IB
rf=pickle.load(open(r'reverse/ghidra/exe1015/pdata.pkl','rb'))
lo=int(sys.argv[1],16); hi=int(sys.argv[2],16); recoff=int(sys.argv[3],16) if len(sys.argv)>3 else 0x2d8
md=capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64); md.detail=True
res={}
for b,e in rf:
    f=IB+b
    if f<lo or f>=hi: continue
    regs={}
    for ins in md.disasm(read(f,e-b), f):
        ops=ins.operands
        if ins.mnemonic=='mov' and len(ops)==2 and ops[0].type==capstone.x86.X86_OP_REG and ops[1].type==capstone.x86.X86_OP_MEM and ops[1].mem.disp==recoff:
            regs[ops[0].reg]=ins.address; continue
        for op in ops:
            if op.type==capstone.x86.X86_OP_MEM and op.mem.base in regs and ins.address-regs[op.mem.base]<40 and op.mem.index==0:
                res.setdefault(op.mem.disp,set()).add(f)
        if ops and ops[0].type==capstone.x86.X86_OP_REG and ops[0].reg in regs and ins.mnemonic not in ('cmp','test'):
            regs.pop(ops[0].reg,None)
for d in sorted(res): print(hex(d), ' '.join(hex(x) for x in sorted(res[d])[:8]), len(res[d]))

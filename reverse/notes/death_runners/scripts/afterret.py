# after each call to target: track rax (and regs it is copied to) for 60 insns, print mem reads off them
import sys, pickle, capstone
sys.path.insert(0, r'reverse/tools')
from tdis import read, IB
from capstone import x86
idx,calls=pickle.load(open(sys.argv[1],'rb'))
t=int(sys.argv[2],16)
md=capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64); md.detail=True
for a,f in calls.get(t,[]):
    regs={x86.X86_REG_RAX}; out=[]
    for k,ins in enumerate(md.disasm(read(a+5,400),a+5)):
        if k>60 or ins.mnemonic in('ret','call','jmp'): break
        ops=ins.operands
        for i,op in enumerate(ops):
            if op.type==capstone.x86.X86_OP_MEM and op.mem.base in regs:
                out.append('%x %s %s'%(ins.address,ins.mnemonic,ins.op_str))
        if ins.mnemonic=='mov' and len(ops)==2 and ops[0].type==1 and ops[1].type==1 and ops[1].reg in regs:
            regs.add(ops[0].reg)
        elif len(ops)>=1 and ops[0].type==1 and ops[0].reg in regs and ins.mnemonic not in('test','cmp'):
            regs.discard(ops[0].reg)
    print('== %x in %x'%(a,f)); [print('   '+o) for o in out]

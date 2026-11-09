import sys, pickle, capstone
sys.path.insert(0, r'reverse/tools')
from tdis import read, IB
rf=pickle.load(open(r'reverse/ghidra/exe1015/pdata.pkl','rb'))
lo=int(sys.argv[1],16); hi=int(sys.argv[2],16); imm=int(sys.argv[3],16); disps=set(int(x,16) for x in sys.argv[4].split(','))
md=capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64); md.detail=True
for b,e in rf:
    f=IB+b
    if f<lo or f>=hi: continue
    hit=[];ds=set()
    for x in md.disasm(read(f,e-b), f):
        for op in x.operands:
            if op.type==capstone.x86.X86_OP_MEM and op.mem.disp in disps: ds.add(op.mem.disp)
        if x.mnemonic=='cmp' and len(x.operands)==2 and x.operands[1].type==capstone.x86.X86_OP_IMM and x.operands[1].imm==imm: hit.append(x.address)
    if hit and ds==disps: print('%x'%f, [hex(h) for h in hit])

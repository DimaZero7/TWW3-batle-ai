import sys, pickle, capstone
sys.path.insert(0, r'reverse/tools')
from tdis import read, IB
rf=pickle.load(open(r'reverse/ghidra/exe1015/pdata.pkl','rb'))
lo=int(sys.argv[1],16); hi=int(sys.argv[2],16); want=[set(int(v,16) for v in g.split('|')) for g in sys.argv[3].split(',')]
md=capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64); md.detail=True
for b,e in rf:
    f=IB+b
    if f<lo or f>=hi: continue
    seen=set()
    for x in md.disasm(read(f,e-b), f):
        if x.mnemonic in('cmp','sub','add') and len(x.operands)==2 and x.operands[1].type==capstone.x86.X86_OP_IMM and x.operands[0].type==capstone.x86.X86_OP_REG:
            seen.add(x.operands[1].imm & 0xffffffff)
    if all(seen & g for g in want): print('%x'%f, sorted(hex(v) for v in seen if v<0x100)[:40])

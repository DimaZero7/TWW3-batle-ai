# functions with disp A and B and an immediate/lea stride S
import sys, pickle, capstone, re
sys.path.insert(0, r'reverse/tools')
from tdis import read, IB
rf=pickle.load(open(r'reverse/ghidra/exe1015/pdata.pkl','rb'))
lo=int(sys.argv[1],16); hi=int(sys.argv[2],16); ds=[int(x,16) for x in sys.argv[3].split(',')]; S=sys.argv[4]
md=capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64); md.detail=True
for b,e in rf:
    f=IB+b
    if f<lo or f>=hi: continue
    code=read(f,e-b)
    if not code: continue
    seen=set(); st=0
    for ins in md.disasm(code,f):
        for op in ins.operands:
            if op.type==capstone.x86.X86_OP_MEM: seen.add(op.mem.disp)
        if re.search(r'\b'+S+r'\b', ins.op_str): st+=1
    if all(d in seen for d in ds) and st: print('%x size=%x stride_hits=%d'%(f,e-b,st))

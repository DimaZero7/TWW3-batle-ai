# functions in [lo,hi) that both read disp A and write dword disp B
import sys, pickle, capstone
sys.path.insert(0, r'reverse/tools')
from tdis import read, IB
rf=pickle.load(open(r'reverse/ghidra/exe1015/pdata.pkl','rb'))
lo=int(sys.argv[1],16); hi=int(sys.argv[2],16); A=int(sys.argv[3],16); B=int(sys.argv[4],16)
md=capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64); md.detail=True
for b,e in rf:
    f=IB+b
    if f<lo or f>=hi: continue
    c=read(f,e-b)
    if not c: continue
    ra=[];wb=[]
    for ins in md.disasm(c,f):
        for i,op in enumerate(ins.operands):
            if op.type==capstone.x86.X86_OP_MEM:
                if op.mem.disp==A: ra.append(ins.address)
                if op.mem.disp==B and i==0 and ins.mnemonic=='mov': wb.append(ins.address)
    if ra and wb: print('%x A=%s B=%s'%(f,' '.join('%x'%x for x in ra[:4]),' '.join('%x'%x for x in wb[:4])))

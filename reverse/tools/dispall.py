# all instructions in .udata battle range with given disp (any base/index), optional mnemonic filter
import sys, pickle, capstone
sys.path.insert(0, r'reverse/tools')
from tdis import read, IB
rf=pickle.load(open(r'reverse/ghidra/exe1015/pdata.pkl','rb'))
lo=int(sys.argv[1],16); hi=int(sys.argv[2],16); disp=int(sys.argv[3],16); pat=sys.argv[4] if len(sys.argv)>4 else ''
md=capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64); md.detail=True
import re
for b,e in rf:
    f=IB+b
    if f<lo or f>=hi: continue
    for ins in md.disasm(read(f,e-b), f):
        for op in ins.operands:
            if op.type==capstone.x86.X86_OP_MEM and op.mem.disp==disp:
                t=ins.mnemonic+' '+ins.op_str
                if re.search(pat,t): print('%x f=%x %s'%(ins.address,f,t))

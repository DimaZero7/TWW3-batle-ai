# calls to attribute check function with edx=imm in given list
import sys, pickle, capstone
sys.path.insert(0, r'reverse/tools')
from tdis import read, IB
rf=pickle.load(open(r'reverse/ghidra/exe1015/pdata.pkl','rb'))
tgt=int(sys.argv[1],16); want=set(int(x,16) for x in sys.argv[2:])
md=capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64); md.detail=True
for b,e in rf:
    f=IB+b
    if f<0x140100000 or f>=0x143400000: continue
    last={}
    try: code=read(f,e-b)
    except: continue
    if not code: continue
    hist=[]
    for ins in md.disasm(code, f):
        hist.append(ins)
        if ins.mnemonic in ('call','jmp') and ins.operands and ins.operands[0].type==capstone.x86.X86_OP_IMM and ins.operands[0].imm==tgt:
            for p in reversed(hist[-8:-1]):
                if p.mnemonic=='mov' and p.op_str.startswith('edx, '):
                    try: v=int(p.op_str.split(', ')[1],16)
                    except: break
                    if v in want: print('%x f=%x edx=%x'%(ins.address,f,v))
                    break

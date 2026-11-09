import sys, pickle, capstone, re
sys.path.insert(0, r'reverse/tools')
from tdis import read, IB
rf=pickle.load(open(r'reverse/ghidra/exe1015/pdata.pkl','rb'))
lo=int(sys.argv[1],16); hi=int(sys.argv[2],16); imm=sys.argv[3]
md=capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
for B,E in rf:
    f=IB+B
    if f<lo or f>=hi: continue
    code=read(f,E-B)
    if not code: continue
    L=list(md.disasm(code,f))
    for i,ins in enumerate(L):
        if ins.mnemonic=='cmp' and re.match(r'^e\w+|^r\d+d', ins.op_str) and ins.op_str.endswith(', '+imm):
            reg=ins.op_str.split(',')[0]
            for p in L[max(0,i-4):i]:
                if p.mnemonic=='mov' and p.op_str.startswith(reg+', dword ptr [') and re.search(r'\+ (4|0x14|0xc)\]',p.op_str):
                    print('%x f=%x %s | %s'%(ins.address,f,p.op_str,ins.op_str))

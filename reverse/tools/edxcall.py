import sys, pickle, capstone, re, collections
sys.path.insert(0, r'reverse/tools')
from tdis import read, IB
rf=pickle.load(open(r'reverse/ghidra/exe1015/pdata.pkl','rb'))
lo=int(sys.argv[1],16); hi=int(sys.argv[2],16); imm=sys.argv[3]
md=capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
cnt=collections.defaultdict(list)
for B,E in rf:
    f=IB+B
    if f<lo or f>=hi: continue
    code=read(f,E-B)
    if not code: continue
    L=list(md.disasm(code,f))
    for i,ins in enumerate(L):
        if ins.mnemonic=='mov' and ins.op_str=='edx, '+imm:
            for n in L[i+1:i+6]:
                if n.mnemonic=='call':
                    cnt[n.op_str].append('%x'%f); break
for t,v in sorted(cnt.items(), key=lambda x:-len(x[1])): print(t, len(v), ' '.join(sorted(set(v)))[:200])

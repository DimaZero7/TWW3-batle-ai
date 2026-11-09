# print 14 instructions before each call to target inside [lo,hi)
import sys, pickle, capstone
sys.path.insert(0, r'reverse/tools')
from tdis import read, IB
idx,calls=pickle.load(open(sys.argv[1],'rb'))
t=int(sys.argv[2],16); n=int(sys.argv[3]) if len(sys.argv)>3 else 14
md=capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
for a,f in calls.get(t,[]):
    ins=list(md.disasm(read(f,a-f+5),f))
    print('== call at %x in %x'%(a,f))
    for i in ins[-n:]: print('   %x %s %s'%(i.address,i.mnemonic,i.op_str))

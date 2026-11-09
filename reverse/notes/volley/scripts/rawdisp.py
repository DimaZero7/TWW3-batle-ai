# find instructions anywhere in .udata whose disp32 == value and that write memory
import sys, re, struct, capstone
sys.path.insert(0, r'reverse/tools')
from tdis import pe, data, IB
s=[x for x in pe.sections if x.Name.startswith(b'.udata')][0]
code=data[s.PointerToRawData:s.PointerToRawData+s.SizeOfRawData]; base=IB+s.VirtualAddress
md=capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
for a in sys.argv[1:]:
    d=int(a,16); pat=struct.pack('<i',d)
    seen=set()
    for m in re.finditer(re.escape(pat), code):
        for back in range(2,5):
            st=m.start()-back
            ins=list(md.disasm(code[st:st+16], base+st))
            if not ins: continue
            i=ins[0]
            if i.size>=back+4 and hex(d) in i.op_str and i.op_str.split(',')[0].find('ptr')>=0 and i.mnemonic.startswith('mov'):
                if i.address not in seen:
                    seen.add(i.address); print(a, hex(i.address), i.mnemonic, i.op_str)

import sys, struct, pickle, capstone
sys.path.insert(0, r'reverse/tools')
from tdis import read
idx,calls=pickle.load(open(r'reverse/ghidra/exe1015/idx_battle.pkl','rb'))
md=capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64); md.detail=True
q=lambda a: struct.unpack('<Q',read(a,8))[0]
seen=set()
for ins_a,f in calls.get(0x143250f9c,[]):
    for ins in md.disasm(read(ins_a,40), ins_a):
        if ins.mnemonic=='lea' and 'rip' in ins.op_str:
            vt=ins.address+ins.size+ins.operands[1].mem.disp
            if vt in seen: break
            seen.add(vt)
            s={o:q(vt+o) for o in (0x60,0x68,0x70,0x80)}
            print('f=%x vt=%x'%(f,vt), ' '.join('%x:%x%s'%(o,v,'' if v in (0x1408d3d10,0x1408d3ca0) else '*') for o,v in s.items()))
            break

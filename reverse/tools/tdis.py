import sys, pefile, capstone
import os
# the exe the tables (pdata.pkl, idx_battle.pkl), the Ghidra project and the notes' addresses belong to - 15.10.2026;
# another exe (the live Steam one): RE_EXE=<path>
p = os.environ.get('RE_EXE') or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ghidra', 'exe1015', 'Warhammer3.exe')
pe = pefile.PE(p, fast_load=True)
data = open(p,'rb').read()
IB=0x140000000
def read(va, n):
    rva = va-IB
    for s in pe.sections:
        if s.VirtualAddress <= rva < s.VirtualAddress+max(s.Misc_VirtualSize,s.SizeOfRawData):
            off = s.PointerToRawData + rva - s.VirtualAddress
            return data[off:off+n]
def cstr(va):
    b = read(va, 300)
    if b is None: return None
    e = b.find(b'\0')
    s = b[:e]
    if len(s)>=3 and all(32<=c<127 for c in s): return s.decode()
    return None
md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64); md.detail=True
def dis(va, n):
    out=[]
    for ins in md.disasm(read(va,n), va):
        cm=''
        if ins.op_str.find('rip')>=0:
            for op in ins.operands:
                if op.type==capstone.x86.X86_OP_MEM and op.mem.base==capstone.x86.X86_REG_RIP:
                    t = ins.address+ins.size+op.mem.disp
                    s = cstr(t)
                    cm = ' ; %x'%t + (' "%s"'%s if s else '')
                    if s is None and ins.mnemonic.startswith(('movss','mulss','addss','subss','divss','comiss','ucomiss','maxss','minss')):
                        import struct; cm+=' =%g'%struct.unpack('<f',read(t,4))[0]
                    if s is None and ins.mnemonic.startswith(('movsd','mulsd','addsd','subsd','divsd','comisd')):
                        import struct; cm+=' =%g'%struct.unpack('<d',read(t,8))[0]
        out.append('%x: %-8s %s%s'%(ins.address, ins.mnemonic, ins.op_str, cm))
    return out
if __name__=='__main__':
    va=int(sys.argv[1],16); n=int(sys.argv[2],0) if len(sys.argv)>2 else 200
    print('\n'.join(dis(va,n)))

import sys, struct, re, capstone
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tdis import read, md, data, pe, IB
s = [x for x in pe.sections if x.Name.startswith(b'.udata')][0]
code = data[s.PointerToRawData:s.PointerToRawData+s.SizeOfRawData]
base = IB+s.VirtualAddress
def find_disp(disp, mn_filter=None, size=4):
    pat = struct.pack('<i', disp) if size==4 else struct.pack('<b', disp)
    res=[]
    for m in re.finditer(re.escape(pat), code):
        i=m.start()
        cands=[]
        for back in range(2,11):
            st=i-back
            ins = next(md.disasm(code[st:st+16], base+st), None)
            if ins is None or ins.size != back+size + (ins.size-back-size if False else 0): pass
            if ins is None: continue
            if ins.size < back+size: continue
            ok=any(op.type==capstone.x86.X86_OP_MEM and op.mem.disp==disp for op in ins.operands)
            if ok and (mn_filter is None or re.search(mn_filter, ins.mnemonic)):
                cands.append((ins.address, ins.mnemonic, ins.op_str))
        res.extend(cands)
    return res
if __name__=='__main__':
    d=int(sys.argv[1],16); f=sys.argv[2] if len(sys.argv)>2 else None
    r=find_disp(d,f)
    for a,mn,op in r: print('%x: %s %s'%(a,mn,op))
    print(len(r))

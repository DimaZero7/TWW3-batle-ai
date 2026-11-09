import sys,struct
sys.path.insert(0,'reverse/tools')
from tdis import read,cstr
for a in sys.argv[1:]:
    va=int(a,16)
    rec=struct.unpack('<6Q',read(va-8,48))
    # name is first field of the record (state ptr = record + 8)
    nm=cstr(rec[0]) if rec[0]>0x140000000 else None
    vt=rec[1]
    fn=struct.unpack('<8Q',read(vt,64)) if 0x140000000<vt<0x150000000 else ()
    print(a,'name',nm,'vt %x'%vt,' '.join('%x'%f for f in fn))

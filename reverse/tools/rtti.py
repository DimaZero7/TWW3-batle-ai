import sys, struct
sys.path.insert(0, r'reverse/tools')
from tdis import read, IB
for a in sys.argv[1:]:
    v=int(a,16)
    col=struct.unpack('<Q',read(v-8,8))[0]
    try:
        b=read(col,24); sig,off,cd,td=struct.unpack('<IIIi',b[:16])
        name=read(IB+td+16,120).split(b'\0')[0]
        print(a, hex(col), name)
    except Exception as e: print(a, hex(col), 'no rtti', e)

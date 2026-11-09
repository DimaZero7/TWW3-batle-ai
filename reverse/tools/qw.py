import sys, struct
sys.path.insert(0, r'reverse/tools')
from tdis import read, dis
a=int(sys.argv[1],16); lo=int(sys.argv[2]); hi=int(sys.argv[3])
for i in range(lo,hi):
    p=struct.unpack('<Q',read(a+8*i,8))[0]
    try:
        L=dis(p,0x20)[:2]; t=' | '.join(l.split(':',1)[1].strip() for l in L)
    except Exception: t=''
    print('%x [%+d] %x  %s'%(a+8*i,i,p,t[:110]))

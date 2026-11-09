import sys, os, re; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tdis import dis
from fn import bounds
a=int(sys.argv[1],16); s,e=bounds(a)
L=dis(s,e-s); getter=None
for l in L:
    m=re.search(r'lea\s+r8, \[rip.*; ([0-9a-f]+)$', l)
    if m: getter=int(m.group(1),16)
    m=re.search(r'lea\s+rdx, .*"([^"]+)"', l)
    if m and getter:
        g=dis(getter,0x20); off=[x for x in g[:5] if 'rax +' in x]
        print(m.group(1), '|', off[0].split(':',1)[1].strip() if off else g[1]); getter=None

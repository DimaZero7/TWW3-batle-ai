import re,json
L=open('reverse/notes/morale_kv.asm').read().splitlines()
cur=None;out=[]
for i,l in enumerate(L):
  m=re.search(r'lea\s+rcx, \[rsi \+ (0x[0-9a-f]+)\]',l)
  if m: cur=int(m.group(1),16)
  if 'add      rcx, 0x18' in l: cur=0x18
  m=re.search(r'"([^"]+)"',l)
  if m and "lea      rdx" in l:
    if m.group(1)=="morale_base": cur=0x18
    c=None
    for j in range(i,min(i+8,len(L))):
      c=re.search(r'call\s+(0x[0-9a-f]+)',L[j])
      if c: break
    out.append((cur,m.group(1),c.group(1) if c else '?'))
kv={}
for l in open('reverse/notes/morale/kv_morale.txt',encoding='utf8').read().splitlines()[2:]:
  r=json.loads(l); kv[r['key']]=r['value']
with open('reverse/notes/morale/kv_offsets.txt','w') as f:
  for o,n,c in out: f.write('%s %s %s %s\n'%(hex(o+0x60),n,c,kv.get(n)))
print(len(out))

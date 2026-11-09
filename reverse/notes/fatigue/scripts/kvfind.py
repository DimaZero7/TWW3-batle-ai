import sys,collections
sys.path.insert(0,r'reverse/tools')
from dispfind import find_disp
from fn import bounds
names="idle idle_in_building idle_rain idle_snow limbering ready shooting working walking walking_artillery walking_horse_artillery running running_artillery_horse running_cavalry running_cavalry_light charging combat climbing_walls climbing_ladders threshold_fresh threshold_active threshold_winded threshold_tired threshold_very_tired threshold_exhausted threshold_max gradient_shallow gradient_steep gradient_very_steep".split()
off={0x78+0x68*i:n for i,n in enumerate(names)}
out=open(sys.argv[1],'w')
for o,n in sorted(off.items()): out.write('%x %s\n'%(o,n))
cnt=collections.defaultdict(set)
for o,n in off.items():
    for a,mn,op in find_disp(o):
        if 'rsp' in op or 'rbp' in op: continue
        cnt[bounds(a)[0]].add(n)
res=sorted(cnt.items(), key=lambda x:-len(x[1]))
for f,s in res[:80]:
    if len(s)<2: break
    out.write('%x %d %s\n'%(f,len(s),' '.join(sorted(s))))

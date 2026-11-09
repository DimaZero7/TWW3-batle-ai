import sys, os, pickle, bisect, json
D=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
from tdis import dis, IB
rf = pickle.load(open(os.path.join(D,'..','ghidra','exe1015','pdata.pkl'),'rb'))
NP=os.path.join(D,'names.json')
names = json.load(open(NP)) if os.path.exists(NP) else {}
def bounds(va):
    rva=va-IB; i=bisect.bisect_right(rf,(rva,1<<40))-1
    return IB+rf[i][0], IB+rf[i][1]
if __name__=='__main__':
    for a in sys.argv[1:]:
        va=int(a,16); s,e=bounds(va)
        print('=== func %x-%x (%s)'%(s,e,names.get('%x'%s,'')))
        for l in dis(s, e-s):
            for k,v in names.items():
                if k in l.split(';')[0] and ('call' in l or 'jmp' in l): l+='  <'+v+'>'
            print(l)

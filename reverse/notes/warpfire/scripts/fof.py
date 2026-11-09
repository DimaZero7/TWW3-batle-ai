import sys,pickle,bisect
rf=pickle.load(open(r'reverse/ghidra/exe1015/pdata.pkl','rb'))
IB=0x140000000
rf=sorted(rf)
st=[b for b,e in rf]
for a in sys.argv[1:]:
    x=int(a,16)-IB; i=bisect.bisect_right(st,x)-1
    print(a, hex(IB+rf[i][0]), hex(IB+rf[i][1]))

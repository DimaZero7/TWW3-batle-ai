import sys, pickle
idx,calls=pickle.load(open(sys.argv[1],'rb'))
for a in sys.argv[2:]:
    d=int(a,16)
    fs=sorted(set(f for _,f in calls.get(d,[])))
    print(a, ' '.join('%x'%f for f in fs))

import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tdis import read, IB, pe, data
import struct
# use .pdata-like runtime function table: find exception directory
import pefile
pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_EXCEPTION']])
rf = [(e.struct.BeginAddress, e.struct.EndAddress) for e in pe.DIRECTORY_ENTRY_EXCEPTION]
import bisect, pickle
rf.sort()
pickle.dump(rf, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','ghidra','exe1015','pdata.pkl'),'wb'))
print(len(rf))
for a in sys.argv[1:]:
    rva=int(a,16)-IB
    i=bisect.bisect_right(rf,(rva,1<<40))-1
    print(a, hex(IB+rf[i][0]), hex(IB+rf[i][1]))

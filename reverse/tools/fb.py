import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fn import bounds
print(' '.join('%x'%bounds(int(a,16))[0] for a in sys.argv[1:]))

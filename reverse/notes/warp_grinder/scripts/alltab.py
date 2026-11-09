import sys, json
sys.path.insert(0, r'reverse/tools')
import db
names=[n for n in dir(db) if not n.startswith('_')]
print(names)

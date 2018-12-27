''' read token_sparse.csv file int np array of bits '''
import os
import numpy as np

# count number of lines in a file, time and space efficient
def __file_len(fname):
    with open(fname) as f:
        for i, _ in enumerate(f):
            pass
    return i + 1

# size
N_DOCS = len([name for name in os.listdir('xml')])
N_TOKENS = __file_len('token_vector.tkn')

# allocate memory
X = np.zeros((N_DOCS, N_TOKENS), dtype=np.bool)

with open('token_sparse.csv', 'r') as sparsefile:
    l = sparsefile.readline()
    while l:
        docid, tokid = map(int, l.split(','))
        X[docid-1][tokid] = True
        l = sparsefile.readline()

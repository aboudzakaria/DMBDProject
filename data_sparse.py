''' read data_sparse.csv file int np array '''
import os
import numpy as np


def __file_len(fname):
    '''count number of lines in a file, time and space efficient'''
    with open(fname) as f:
        for i, _ in enumerate(f):
            pass
    return i + 1


# size
N_DOCS = len([name for name in os.listdir('xml')])
N_TOKENS = __file_len('token_df_X2.tkn')

if __name__ == "__main__":
    print("size = ({0},{1})".format(N_DOCS, N_TOKENS))

# allocate memory
data_boolean = np.zeros((N_DOCS, N_TOKENS), dtype=np.bool)
data_tf = np.zeros((N_DOCS, N_TOKENS), dtype=np.int16)
data_tfidf = np.zeros((N_DOCS, N_TOKENS), dtype=np.float)

with open('data_sparse.csv', 'r') as sparsefile:
    l = sparsefile.readline()
    while l:
        splt = l.split(',')
        docid, tokid = map(int, splt[:2])
        boolean = bool(splt[2])
        tf = splt[3]
        tfidf = float(splt[4])

        data_boolean[docid-1][tokid] = boolean
        data_tf[docid-1][tokid] = tf
        data_tfidf[docid-1][tokid] = tfidf

        l = sparsefile.readline()

from sklearn.model_selection import train_test_split
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

no_text = np.genfromtxt('no_text.csv', dtype=str, delimiter='.')[
    :, 0].astype(int) - 1
traincsv = np.genfromtxt('train.csv', dtype=str, delimiter=',', skip_header=True)[
    :, [0, 2]].astype(int)
traincsv[:, 0] -= 1
ix = np.invert(np.isin(traincsv[:, 0], no_text))
trainind = traincsv[ix, 0]

testcsv = np.genfromtxt('test.csv', dtype=str, delimiter=',', skip_header=True)[
    :, 0].astype(int)
testcsv[:] -= 1
ixx = np.invert(np.isin(testcsv[:], no_text))
testind = testcsv[ixx]

X_b = data_boolean[trainind]
X_tf = data_tf[trainind]
X_tfidf = data_tfidf[trainind]
y = traincsv[ix, 1]

X_train_b, X_validate_b, y_train_b, y_validate_b = train_test_split(X_b, y)
X_train_tf, X_validate_tf, y_train_tf, y_validate_tf = train_test_split(
    X_tf, y)
X_train_tfidf, X_validate_tfidf, y_train_tfidf, y_validate_tfidf = train_test_split(
    X_tfidf, y)

X_test_b = data_boolean[testind]
X_test_tf = data_tf[testind]
X_test_tfidf = data_tfidf[testind]

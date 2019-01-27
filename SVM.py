from data_sparse import *
from sklearn import svm
import sys

kernels = ['rbf', 'linear', 'poly']
gammas = [0.1, 1, 10, 100]
Cs = [0.1, 1, 10, 100, 1000]
degrees = [1, 2, 3, 4, 5, 6]

for kernel in kernels:
    for gamma in gammas:
        for C in Cs:
            for degree in degrees:
                clf = svm.SVC(kernel=kernel, gamma=gamma, C=C, degree=degree).fit(
                    X_train_tfidf, y_train_tfidf)
                pred = clf.predict(X_validate_tfidf)
                accuracy = (y_validate_tfidf == pred).mean() * 100
                print("kernel={0},gamma={1},C={2},degree={3} => accuracy={4}%".format(
                    kernel, str(gamma), str(C), str(degree), str(accuracy)))
                sys.stdout.flush()

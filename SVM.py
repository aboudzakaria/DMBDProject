from data_sparse import *
from sklearn import svm
from sklearn.metrics import f1_score
import sys

gammas = [0.1, 1, 10, 100]
Cs = [0.1, 1, 10, 100]
degrees = [1, 2, 3, 4]

for gamma in gammas:
    for C in Cs:
        clf = svm.SVC(kernel='rbf', gamma=gamma, C=C).fit(
            X_train_tfidf, y_train_tfidf)
        pred = clf.predict(X_validate_tfidf)
        accuracy = (y_validate_tfidf == pred).mean() * 100
        f1 = f1_score(y_validate_tfidf, pred)
        print("kernel={0},gamma={1},C={2} => f1={3},accuracy={4}%".format(
            'linear', str(gamma), str(C), str(f1), str(accuracy)))
        sys.stdout.flush()    


for C in Cs:
    clf = svm.SVC(kernel='linear', C=C).fit(
        X_train_tfidf, y_train_tfidf)
    pred = clf.predict(X_validate_tfidf)
    accuracy = (y_validate_tfidf == pred).mean() * 100
    f1 = f1_score(y_validate_tfidf, pred)
    print("kernel={0},C={1} => f1={2},accuracy={3}%".format(
        'linear', str(C), str(f1), str(accuracy)))
    sys.stdout.flush()

for degree in degrees:
    for gamma in gammas:
        for C in Cs:
            clf = svm.SVC(kernel='poly', C=C, degree=degree).fit(
                X_train_tfidf, y_train_tfidf)
            pred = clf.predict(X_validate_tfidf)
            accuracy = (y_validate_tfidf == pred).mean() * 100
            f1 = f1_score(y_validate_tfidf, pred)
            print("kernel={0},gamma={1},C={2},degree={3} => f1={4},accuracy={5}%".format(
                'poly', str(gamma), str(C), str(degree), str(f1), str(accuracy)))
            sys.stdout.flush()

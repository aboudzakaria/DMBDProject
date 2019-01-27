import sys
import pickle
import numpy as np
from data_sparse import *
from sklearn.metrics import f1_score
coco = 0

from sklearn.neighbors import KNeighborsClassifier

def U_dis(d1,d2):
    global coco
    if coco % 100 == 0:
        print(int(coco/100))
        sys.stdout.flush()
    coco += 1
    assert len(d1) == len (d2), "d1 and d2 don't have the same size!"
    L = len(d1)
    return sum( [ (d1[i]-d2[i])**2 for i in range(L)] )

def sc_KNN(X_train,X_test,y_train,y_test, k):
    neigh = KNeighborsClassifier(n_neighbors=k,algorithm='kd_tree' )
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    """print(y_pred)
    print(y_test)
    print(y_pred == y_test)"""
    
    return f1_score(y_test, y_pred)

if __name__ == '__main__':
    __debug = False
    ks = [i for i in range(1, 21)]
    
    #get data here
    X_train = X_tfidf
    X_test = X_validate_tfidf
    y_train = y_train_tfidf
    y_test = y_validate_tfidf
    if __debug:
        ks = [i for i in range(1, 9)]
        X_train = X_train[:100,:]
        X_test = X_test[:100,:]
        y_train = y_train[:100]
        y_test = y_test[:100]
       
    print("X_train shape", X_train.shape)
    print("X_test shape", X_test.shape)
    print("y_train shape", y_train.shape)
    print("y_test shape", y_test.shape)
    
    for k in ks:
        print('K= ',k,' ',sc_KNN(X_train,X_test,y_train,y_test,k))

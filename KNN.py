import sys
import pickle
import numpy as np
from data_sparse import *
coco = 0

def U_dis(d1,d2):
    global coco
    if coco % 100 == 0:
        print(int(coco/100))
        sys.stdout.flush()
    coco += 1
    assert len(d1) == len (d2), "d1 and d2 don't have the same size!"
    L = len(d1)
    return sum( [ (d1[i]-d2[i])**2 for i in range(L)] )

def knn_accuracy(y_train, y_test, dist, ks):
    sortedind = dist.argsort(axis=0)
    # majority vote
    for k in ks:
        count = 0
        for i in range(y_test.size):
            knearest = sortedind[:k, i]
            y_pred = np.bincount(y_train[knearest]).argmax()
            if y_pred == y_test[i]:
                count += 1
        print("k={0}, accuracy={1}%".format(k, count/y_test.size))
        sys.stdout.flush()

if __name__ == '__main__':
    ks = [i for i in range(1, 21)]
    
    #get data here
    X_train = X_train_tf
    X_test = X_validate_tf
    y_train = y_train_tf
    y_test = y_validate_tf
    
    print("X_train shape", X_train.shape)
    print("X_test shape", X_test.shape)
    print("y_train shape", y_train.shape)
    print("y_test shape", y_test.shape)
        
    train_test_dis = [ [ U_dis(X_train[i],X_test[j]) for j in range(y_test.size) ] for i in range(y_train.size) ]
    print('finished calc distance')
    knn_accuracy(y_train, y_test, train_test_dis, ks)

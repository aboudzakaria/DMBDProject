import sys
from data_sparse import *
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

if __name__ == '__main__':
    __debug = False
    
    #get data here
    X_train = X_train_tfidf
    X_test = X_validate_tfidf
    y_train = y_train_tfidf
    y_test = y_validate_tfidf
    if __debug:
        X_train = X_train[:100,:]
        X_test = X_test[:100,:]
        y_train = y_train[:100]
        y_test = y_test[:100]
    print("\nUsing Logistic Regression .. \n")
    print("X_train shape", X_train.shape)
    print("X_test shape", X_test.shape)
    print("y_train shape", y_train.shape)
    print("y_test shape", y_test.shape)
    print("\n")
    logreg = LogisticRegression()
    logreg.fit(X_train,y_train)
    y_pred = logreg.predict(X_test)

print('Accuracy: {:.3f}'.format(logreg.score(X_test, y_test)))
f1 = f1_score(y_test, y_pred) , (y_pred == y_test).mean()
print("\nF1 Score:",f1[0])
cf_mat = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cf_mat)
sys.stdout.flush()

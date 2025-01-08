from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from Evaluation import evaluation
import random as rn

def Model_KNN(train_data,train_target,test_data,test_target):
    scaler = StandardScaler()
    scaler.fit(train_data)

    X_train = scaler.transform(train_data)
    X_test = scaler.transform(test_data)
    classifier = KNeighborsClassifier(n_neighbors=5)


    classifier.fit(X_train.tolist(), train_target.tolist())
    # Predict Output
    out = classifier.predict(X_test.tolist())
    predicted = np.round(out)

    Eval = evaluation(predicted.reshape(-1, 1), test_target.reshape(-1, 1))


    return np.asarray(Eval).ravel(), predicted



from sklearn.svm import SVC  # "Support Vector Classifier"
import numpy as np
from Evaluation import evaluation


def Model_SVM(train_data, train_target, test_data, test_target):
    clf = SVC(kernel='linear')
    clf.fit(train_data, train_target[:, 0])
    Y_pred = clf.predict(test_data.tolist())
    pred = np.asarray(Y_pred)
    pred= np.resize(pred, (65, 2))
    Eval = evaluation(pred, test_target)
    return Eval, pred




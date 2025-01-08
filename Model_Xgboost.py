import numpy as np
from xgboost import XGBClassifier
from Evaluation import evaluation


def Model_Xgboost(train_data, train_target, test_data, test_target):
    model = XGBClassifier(learning_rate=0.01,objective="binary:logistic", random_state=42, eval_metric="auc")
    pred = np.zeros(test_target.shape).astype('int')
    for i in range(test_target.shape[1]):
        model.fit(train_data, train_target[:, i], epochs=100)
        pred[:, i] = model.predict(test_data)
    predict = np.round(pred).astype('int')
    Eval = evaluation(predict, test_target)
    return np.asarray(Eval).ravel()




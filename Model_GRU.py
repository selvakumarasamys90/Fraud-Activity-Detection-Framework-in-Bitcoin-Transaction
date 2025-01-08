# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from Evaluation import evaluation
plt.style.use('fivethirtyeight')
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU


def Model_GRU(X_train, y_train, X_test, Y_test):
    Train_Data = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    Test_Data = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    # The GRU architecture
    regressorGRU = Sequential()
    # First GRU layer with Dropout regularisation
    regressorGRU.add(GRU(units=10, return_sequences=True, input_shape=(1, Train_Data.shape[2]), activation='tanh'))
    regressorGRU.add(Dropout(0.2))
    # The output layer
    regressorGRU.add(Dense(y_train.shape[1]))
    # Compiling the RNN
    regressorGRU.compile(optimizer='adam',
                         loss='mean_squared_error')
    Train_Data = np.asarray(Train_Data).astype(np.float32)
    TestX = np.asarray(Test_Data).astype(np.float32)
    regressorGRU.fit(Train_Data, y_train, epochs=5)
    pred = regressorGRU.predict(TestX)
    pred = np.reshape(pred, (pred.shape[0], pred.shape[2]))
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    Eval = evaluation(pred, Y_test)
    return Eval, pred

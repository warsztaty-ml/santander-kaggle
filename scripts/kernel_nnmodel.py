# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import gc
import sklearn
import sklearn.metrics
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Conv1D, BatchNormalization, Flatten
from keras.callbacks import Callback, EarlyStopping
from keras import metrics
from keras import losses
from keras import activations

BATCH_SIZE = 128
EPOCH_NUMBER = 10
VERBOSE = 2

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))
import psutil

process = psutil.Process(os.getpid())

# Any results you write to the current directory are saved as output.
INPUT_DIR = "../input/"

model = Sequential()


def auc_roc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)


def prepare_network(input_size):
    print("Długość wektora wyjściowego: " + str(input_size))

    model.add(Conv1D(filters=20, kernel_size=2, input_shape=(input_size, 1), activation=activations.relu))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.5))
    # model.add(Dense(256, kernel_initializer='normal', activation=activations.relu))
    # model.add(Dense(128, kernel_initializer='normal', activation=activations.relu))
    # model.add(Dense(64, kernel_initializer='normal', activation=activations.relu))
    model.add(Dense(32, kernel_initializer='normal', activation=activations.relu))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation=activations.sigmoid))
    print(model.summary())
    model.compile(optimizer='adam', loss=losses.binary_crossentropy, metrics=[auc_roc])


def network_train(x_train, y_train):
    early_stopping = EarlyStopping(monitor='auc_roc', patience=3, verbose=0, mode='max', restore_best_weights=True)
    model.fit(x_train, y_train, BATCH_SIZE, EPOCH_NUMBER, VERBOSE, callbacks=[early_stopping], validation_split=0.2,
              shuffle=True)
    # model.fit(x_train, y_train, BATCH_SIZE, EPOCH_NUMBER, VERBOSE)


def network_test(x_test):
    return model.predict(x_test)


def data_preprocessing(df, is_train):
    target = 'target'
    y = None
    df_out = df.drop('ID_code', axis=1, inplace=False)
    scaler = StandardScaler()
    if is_train:
        y = df[target].values
        df_out.drop('target', axis=1, inplace=True)
    X = scaler.fit_transform(df_out)
    X = X.reshape(len(df), len(df_out.columns), 1)
    return X, y, len(df_out.columns)


def data_postprocessing(df_test, pred):
    df_sub = pd.DataFrame(df_test)
    df_sub['target'] = pred
    return df_sub[["ID_code", "target"]]


def train_data(train):
    print("Przeprowadzenie preprocessingu i nauki na danych")
    x_train, y_train, input_size = data_preprocessing(train, True)

    prepare_network(input_size)
    network_train(x_train, y_train)


def test_data(test):
    print("Przeprowadzenie predykcji i postprocesingu na danych")
    x_test, _, _ = data_preprocessing(test, False)
    result = network_test(x_test)
    test_output = data_postprocessing(test, result)
    return test_output


def main():
    print("Wczytywanie danych treningowych")
    train = pd.read_csv(INPUT_DIR + 'train.csv')
    print("Rozpoczęcie procesu nauki")
    train_data(train)
    del train
    print("Rozpoczęcie procesu testowania")
    test = pd.read_csv(INPUT_DIR + 'test.csv')
    submission = test_data(test)
    print("Zapis odpowiedzi do pliku")
    submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
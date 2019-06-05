# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import gc
import sklearn
import sklearn.metrics
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Conv1D, BatchNormalization, Flatten
from keras.callbacks import Callback, EarlyStopping
from keras import metrics
from keras import losses
from keras import activations
import argparse

BATCH_SIZE = 128
EPOCH_NUMBER = 100
VERBOSE = 2
NUMBER_NETWORKS = 30

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import psutil

process = psutil.Process(os.getpid())

models = list()
scaler = StandardScaler()


def auc_roc(y_true, y_pred):
    return tf.py_func(roc_auc_score_FIXED, (y_true, y_pred), tf.double)


def roc_auc_score_FIXED(y_true, y_pred):
    if len(np.unique(y_true)) == 1:  # bug in roc_auc_score
        return accuracy_score(y_true, np.rint(y_pred))
    return roc_auc_score(y_true, y_pred)


def prepare_network(input_size):
    print("Długość wektora wyjściowego: " + str(input_size))
    for i in range(NUMBER_NETWORKS):
        model = Sequential()
        model.add(Conv1D(filters=32, kernel_size=1, input_shape=(input_size, 1), activation=activations.relu))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(1, activation=activations.sigmoid))
        model.compile(optimizer='adam', loss=losses.binary_crossentropy, metrics=[auc_roc])
        models.append(model)
    print(models[0].summary())


def network_train(model, x_train, y_train, x_valid, y_valid):
    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(y_train),
                                                      y_train)
    early_stopping = EarlyStopping(monitor='auc_roc', patience=5, verbose=0, mode='max', restore_best_weights=True)
    model.fit(x_train, y_train, BATCH_SIZE, EPOCH_NUMBER, VERBOSE, callbacks=[early_stopping],
              validation_data=(x_valid, y_valid),
              shuffle=True, class_weight=class_weights)
    # model.fit(x_train, y_train, BATCH_SIZE, EPOCH_NUMBER, VERBOSE)


def network_test(model, x_test):
    return model.predict(x_test)


def data_preprocessing(df, is_train):
    target = 'target'
    y = None
    if is_train:
        y = df[target].values
        df_out = df.drop('target', axis=1, inplace=False)
    else:
        df_out = df.drop('ID_code', axis=1, inplace=False)
    X = scaler.fit_transform(df_out)
    X = X.reshape(len(df), len(df_out.columns), 1)
    return X, y, len(df_out.columns)


def data_postprocessing(df_test, pred):
    df_sub = pd.DataFrame(df_test)
    df_sub['target'] = pred
    return df_sub[["ID_code", "target"]]


def train_data(train):
    print("Przeprowadzenie preprocessingu i nauki na danych")
    df_0 = train[train['target'] == 0]
    df_1 = train[train['target'] > 0]

    prepare_network(len(train.columns) - 1)

    for i in range(NUMBER_NETWORKS):
        train_0, valid_0 = train_test_split(df_0, test_size=0.2)
        train_1, valid_1 = train_test_split(df_1, test_size=0.2)
        train = pd.concat([train_0, train_1]).sample(frac=1)
        valid = pd.concat([valid_0, valid_1]).sample(frac=1)
        x_train, y_train, input_size = data_preprocessing(train, True)
        x_valid, y_valid, input_size = data_preprocessing(valid, True)
        network_train(models[i], x_train, y_train, x_valid, y_valid)


def test_data(test):
    x_test, _, _ = data_preprocessing(test, False)
    predictions = None
    for i in range(NUMBER_NETWORKS):
        result = network_test(models[i], x_test)
        if predictions is None:
            predictions = result
        else:
            predictions = result + predictions
    predictions = predictions / NUMBER_NETWORKS
    test_output = data_postprocessing(test, predictions)
    return test_output


def main():
    parser = argparse.ArgumentParser \
        (description='Performs binary classification using convolutional neural network')
    parser.add_argument('--train_data', help='Input train data csv', default='../data/train.csv', type=str)
    parser.add_argument('--data', help='Input data csv', default='../data/test.csv', type=str)
    parser.add_argument('--output', help='output path', default='cnn_submission.csv', type=str)
    print("Wczytywanie danych treningowych")
    args = parser.parse_args()
    train = pd.read_csv(args.train_data)
    df_train = train
    X = df_train.iloc[:, 2:]
    Y = df_train['target']
    df_train = pd.DataFrame(X, columns=df_train.columns[2:])
    df_train['target'] = pd.Series(Y)
    df_train = df_train.sample(frac=1)
    train_data(df_train)
    del train
    print("Rozpoczęcie procesu testowania")
    df_test = pd.read_csv(args.data)
    submission = test_data(df_test)
    print("Zapis odpowiedzi do pliku")
    submission.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()

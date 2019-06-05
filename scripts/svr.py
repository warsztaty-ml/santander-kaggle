import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import sklearn.metrics
from sklearn.metrics import roc_auc_score
import processing as pc
import argparse
from sklearn import svm
# Any results you write to the current directory are saved as output.
INPUT_DIR = "../input/"

def train_data_SVR(train, clf, version=1):
    print("Przeprowadzenie preprocessingu danych trenigowych")
    if version == 1:
        x_train, y_train, input_size = pc.data_preprocessing_with_undersampled(train, True)
    else:
        x_train, y_train, input_size = pc.data_preprocessing_with_undersampled2(train, True)

    print("Rozpoczęcie procesu nauki")
    clf.fit(x_train, y_train)


def test_data_SVR(test, clf, version=1):
    print("Przeprowadzenie preprocessingu danych testowych")
    if version == 1:
        x_test_solo, _, _ = pc.data_preprocessing_with_undersampled(test, False)
    else:
        x_test_solo, _, _ = pc.data_preprocessing_with_undersampled2(test, False)

    print("Rozpoczęcie procesu testowania")
    pred_solo = clf.predict(x_test_solo)
    print("Przeprowadzenie postprocessingu danych testowych")
    test_output = pc.data_postprocessing(test, pred_solo)

    return test_output


###############################################################################
def args():
    """Parses command line arguments given to the script using argparse
    """
    parser = argparse.ArgumentParser(description='Performs binary classification using the SVM algorithm')

    parser.add_argument('--kernel', help='Kernel', default='rbf', type=str)
    parser.add_argument('--eps', help='Epsilon', default=0.1, type=float)
    parser.add_argument('--gamma', help='Gamma', default='auto', type=str)
    parser.add_argument('--c', help='C', default=1, type=float)
    parser.add_argument('--wsp', help='Data', default=1, type=float)
    parser.add_argument('--train_data', help='Input train data csv', default='../data/train.csv', type=str)
    parser.add_argument('--data', help='Input data csv', default='../data/test.csv', type=str)
    parser.add_argument('--output', help='output path', default='svm_submission.csv', type=str)

    args = parser.parse_args()
    if args.gamma != 'auto':
        args.gamma = float(args.gamma)

    print(args)
    return args.kernel, args.eps, args.gamma, args.c, args.wsp, args.train_data, args.data, args.output


def main():
    kernel, eps, gamma, c, wsp, train_data, data, output = args()
    clf = svm.SVR(kernel=kernel, gamma=gamma, C=c, epsilon=eps)
    print("Wczytywanie danych treningowych")
    train = pd.read_csv(train_data)
    num_rows = int(train.shape[0] * wsp)
    train = train.head(num_rows)
    print("Rozpoczęcie procesu nauki")
    train_data_SVR(train, clf)
    del train
    print("Wczytywanie danych testowych")
    test = pd.read_csv(data)
    submission = test_data_SVR(test, clf)
    print("Zapis odpowiedzi do pliku")
    submission.to_csv(output, index=False)


def roc_auc_score(y, pred):
    return sklearn.metrics.roc_auc_score(y, pred)


def main2():
    kernel, eps, gamma, c, wsp, train_data, data, output = args()
    clf = svm.SVR(kernel=kernel, gamma=gamma, C=c, epsilon=eps)
    print("Wczytywanie danych")
    data = pd.read_csv(train_data)
    num_rows = data.shape[0]
    num_rows = int(num_rows * wsp)
    k = int(num_rows * 9 / 10)
    train = data.head(k)
    test = data.tail(num_rows - k)
    train_data_SVR(train, clf, 2)

    y_test = test[["target"]]
    submission = test_data_SVR(test, clf, 2)
    pred = submission[["target"]]
    print("Otrzymany rezultat:", roc_auc_score(np.array(y_test, dtype=np.float64), pred))


if __name__ == '__main__':
    main()

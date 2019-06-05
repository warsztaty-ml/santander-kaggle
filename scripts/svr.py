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

def train_data_SVR(train, clf):
    print("Przeprowadzenie preprocessingu danych trenigowych")
    x_train, y_train, input_size = pc.data_preprocessing_with_undersampled(train, True)
    print("Rozpoczęcie procesu nauki")
    clf.fit(x_train, y_train)


def test_data_SVR(test, clf):
    print("Przeprowadzenie preprocessingu danych testowych")
    x_test_solo, _, _ = pc.data_preprocessing_with_undersampled(test, False)
    print("Rozpoczęcie procesu testowania")
    pred_solo = clf.predict(x_test_solo)
    print("Przeprowadzenie postprocessingu danych testowych")
    test_output = pc.data_postprocessing(test, pred_solo)

    return test_output


###############################################################################

def main():
    print("Wczytywanie danych treningowych")
    train = pd.read_csv(INPUT_DIR + 'train.csv')
    print("Rozpoczęcie procesu nauki")
    # train_data(train)
    train_data_SVR(train)
    del train
    print("Rozpoczęcie procesu testowania")
    test = pd.read_csv(INPUT_DIR + 'test.csv')
    # submission = test_data(test)
    submission = test_data_SVR(test)
    print("Zapis odpowiedzi do pliku")
    submission.to_csv('submission.csv', index=False)


def roc_auc_score(y, pred):
    return sklearn.metrics.roc_auc_score(y, pred)


def args():
    """Parses command line arguments given to the script using argparse
    """
    parser = argparse.ArgumentParser(description='Performs binary classification using the SVM algorithm')

    parser.add_argument('--kernel', help='Kernel', default='rbf', type=str)
    parser.add_argument('--eps', help='Epsilon', default=0.1, type=float)
    parser.add_argument('--gamma', help='Gamma', default='auto', type=str)
    parser.add_argument('--c', help='C', default=1, type=float)
    parser.add_argument('--wsp', help='Data', default=1, type=float)

    args = parser.parse_args()
    if args.gamma != 'auto':
        args.gamma = float(args.gamma)

    print(args)
    return args.kernel, args.eps, args.gamma, args.c, args.wsp


def main2():
    kernel, eps, gamma, c, wsp = args()
    clf = svm.SVR(kernel=kernel, gamma=gamma, C=c, epsilon=eps)
    print("Wczytywanie danych")
    data = pd.read_csv(INPUT_DIR + 'train.csv')
    num_rows = data.shape[0]
    num_rows = int(num_rows * wsp)
    k = int(num_rows * 9 / 10)
    train = data.head(k)
    test = data.tail(num_rows - k)
    train_data_SVR(train, clf)

    y_test = test[["target"]]
    submission = test_data_SVR(test, clf)
    pred = submission[["target"]]
    print("Otrzymany rezultat:", roc_auc_score(np.array(y_test, dtype=np.float64), pred))


if __name__ == '__main__':
    main2()

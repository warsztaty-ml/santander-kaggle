import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import sklearn.metrics
from sklearn.metrics import roc_auc_score
import processing as pc
from sklearn import svm
# Any results you write to the current directory are saved as output.
INPUT_DIR = "../input/"

clf = svm.SVR()

def train_data_SVR(train):
    print("Przeprowadzenie preprocessingu i nauki na danych")
    x_train, y_train, input_size = pc.data_preprocessing_with_undersampled(train, True)
    clf.fit(x_train, y_train)


def test_data_SVR(test):
    print("Przeprowadzenie predykcji i postprocesingu na danych")
    x_test_solo, _, _ = pc.data_preprocessing_with_undersampled(test, False)
    pred_solo = clf.predict(x_test_solo)
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


def main2():
    print("Wczytywanie danych treningowych")
    data = pd.read_csv(INPUT_DIR + 'train.csv')
    num_rows = data.shape[0]
    # num_rows = int( num_rows / 1000)
    k = int(num_rows * 9 / 10)
    train = data.head(k)
    test = data.tail(num_rows - k)
    print("Rozpoczęcie procesu nauki")
    train_data_SVR(train)

    print("Rozpoczęcie procesu testowania")
    y_test = test[["target"]]
    submission = test_data_SVR(test)
    pred = submission[["target"]]
    print(roc_auc_score(np.array(y_test, dtype=np.float64), pred))


if __name__ == '__main__':
    main()

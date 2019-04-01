import numpy as np
import pandas as pd
import itertools
from sklearn import mixture
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score
import processing

DATA_DIR = '../data/'
IS_UNDERSAMPLED = True

IS_GRID_SEARCH = True
IS_TRAINING = True

component_numbers = [5, 10, 15, 20]
reg_covars = [1e-2, 1e-3, 1e-5, 1e-6]


# using Bayes' theorem
# calculates probability of Y on condition X
# Y stands for predicted class
# Xi stands for variable i, i = 0,...,199
# Likelihood of every Xi|(Y=0) and Xi|(Y=1) is calculated using gaussian mixture
# all is calculated in logs, so we avoid multiplying small numbers
class GaussMixNaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self, component_number=1, reg_covar=1e-06):
        self.component_number = component_number
        self.reg_covar = reg_covar

    def fit(self, x, y):
        print("Training started")
        self.log_probY = np.log(np.bincount(y) / len(y))
        self.log_prob_Xi_on_Y = [[mixture.GaussianMixture(n_components=self.component_number, reg_covar=self.reg_covar)
                                  .fit(x[y == i, j:j + 1])
                                  .score_samples for j in range(x.shape[1])]
                                    for i in range(len(self.log_probY))]
        print("Training finished")

    def predict_proba(self, x):
        print("Predicting probabilities started")
        # shape of log_likelihood before summing
        shape = (len(self.log_probY), x.shape[1], x.shape[0])
        log_likelihood = np.sum([[self.log_prob_Xi_on_Y[i][j](x[:, j:j + 1])
                                  for j in range(shape[1])]
                                 for i in range(shape[0])], axis=1).T
        log_joint = self.log_probY + log_likelihood
        print("Predicting probabilities finished")
        return np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True))

    def predict(self, x):
        return self.predict_proba(x).argmax(axis=1)


clf = GaussMixNaiveBayes(component_number=component_numbers[0], reg_covar=reg_covars[0])


def load_and_process_data(is_training):
    if is_training:
        print("Loading train data")
        data = pd.read_csv(DATA_DIR + 'train.csv')
    else:
        print("Loading test data")
        data = pd.read_csv(DATA_DIR + 'test.csv')

    if IS_UNDERSAMPLED and is_training:
        x_data, y_data, col_nr = processing.data_preprocessing_with_undersampled(data, True)
    else:
        x_data, y_data, col_nr = processing.data_preprocessing(data, is_training)

    x_data = x_data.values.astype('float64')
    y_data = y_data.astype('int32')
    return x_data, y_data, col_nr


def hyperparameter_grid_search():
    global clf
    x_data, y_data, col_nr = load_and_process_data(True)

    col_names = ['component_number', 'reg_covar', 'AUC']
    rows = []

    for cn, reg_cov in itertools.product(component_numbers, reg_covars):
            clf = GaussMixNaiveBayes(component_number=cn, reg_covar=reg_cov)
            clf.fit(x_data, y_data)
            auc = roc_auc_score(y_data, clf.predict_proba(x_data)[:, 1])

            print('component number: {}, reg_covar: {}, auc: {}'.format(cn, reg_cov, auc))
            rows.append([cn, reg_cov, auc])


    search_results = pd.DataFrame(data=rows, columns=col_names)
    search_results = search_results.sort_values(by=['AUC'], ascending=False)

    search_results.to_csv('{}gauss_mix_grid_search.csv'.format(DATA_DIR), index=False)


def main():
    if IS_GRID_SEARCH:
        hyperparameter_grid_search()
    else:
        x_data, y_data, col_nr = load_and_process_data(IS_TRAINING)
        if IS_TRAINING:
            clf.fit(x_data, y_data)
            print('AUC: {}'.format(roc_auc_score(y_data, clf.predict_proba(x_data)[:, 1])))
        else:
            x_train, y_train, col_nr = load_and_process_data(True)
            clf.fit(x_train, y_train)

            pred = clf.predict_proba(x_data)[:, 1]

            id_codes = ['test_{}'.format(i) for i in range(x_data.shape[0])]
            df = pd.DataFrame(data=id_codes, columns=["ID_code"])
            results = processing.data_postprocessing(df, pred)

            results.to_csv(DATA_DIR + 'gaussian_mixture_results.csv', index=False)


if __name__ == '__main__':
    main()

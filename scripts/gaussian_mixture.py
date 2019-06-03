import numpy as np
import pandas as pd
import itertools
from sklearn import mixture
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import argparse
import data_processor
import processing

#Example parameters for script
#--train_data=train.csv --data=test.csv --data_balancing=o --component_number=3 --reg_covar=1e-3 --is_only_training=true

IS_UNDERSAMPLED = False
IS_OVERSAMPLED = False
IS_GRID_SEARCH = False
IS_TRAINING = False
SEED = None
TrainData = None
TestData = None
DataProcessor = None

component_numbers = [5, 10, 15, 20]
reg_covars = [1e-2, 1e-4]


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


def create_data_processor(is_undersampled, is_oversampled, seed=0):
    global DataProcessor
    if is_undersampled:
        DataProcessor = data_processor.DataProcessor(TrainData).with_scaling().with_undersampling(seed)
    else:
        if is_oversampled:
            DataProcessor = data_processor.DataProcessor(TrainData).with_scaling().with_oversampling(seed)
        else:
            DataProcessor = data_processor.DataProcessor(TrainData).with_scaling()


def process_data(is_training):
    if is_training:
        x_data, y_data, col_nr = DataProcessor.process_train()
    else:
        x_data, y_data, col_nr = DataProcessor.process_test(TestData)

    x_data = x_data.values.astype('float64')
    if is_training:
        y_data = y_data.astype('int32')
    return x_data, y_data, col_nr


def hyperparameter_grid_search():
    x_data, y_data, _ = process_data(True)
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1, random_state=1)

    col_names = ['component_number', 'reg_covar', 'AUC_train', 'AUC_val']
    rows = []

    for cn, reg_cov in itertools.product(component_numbers, reg_covars):
        clf = GaussMixNaiveBayes(component_number=cn, reg_covar=reg_cov)
        clf.fit(x_train, y_train)
        auc_train = roc_auc_score(y_train, clf.predict_proba(x_train)[:, 1])
        auc_val = roc_auc_score(y_val, clf.predict_proba(x_val)[:, 1])

        print('component number: {}, reg_covar: {}, train auc: {}, val auc: {}'
              .format(cn, reg_cov, auc_train, auc_val))
        rows.append([cn, reg_cov, auc_train, auc_val])

    search_results = pd.DataFrame(data=rows, columns=col_names)
    search_results = search_results.sort_values(by=['AUC_val'], ascending=False)

    search_results.to_csv('gauss_mix_grid_search.csv', index=False)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser\
        (description='Performs binary classification using Bayes classificator with gaussian mixture')
    parser.add_argument('--train_data', help='Input train data csv', default='../data/train.csv', type=str)
    parser.add_argument('--data', help='Input data csv', default='../data/test.csv', type=str)
    parser.add_argument('--seed', help='Random seed', default=12415, type=int)
    parser.add_argument('--data_balancing', help='Specify if you want to use undersampling(U) or oversampling(O)',
                        default='oversampling', type=str)
    parser.add_argument('--is_only_training',
                        help='True if you want to just train the model and receive feedback (model is not saved)',
                        default='false', type=str2bool)
    parser.add_argument('--component_number', help='The number of mixture components', default=5, type=int)
    parser.add_argument('--reg_covar', help='Non-negative regularization added to the diagonal of covariance',
                        default=1e-2, type=float)
    parser.add_argument('--is_grid_search', help='If True, performs grid search (on fixed parameters in code, ' +
                                              '--component_number and --reg_covar parameters don\'t matter)',
                        default='false', type=str2bool)
    parser.add_argument('--output', help='output path', default='gauss_mix_output.csv', type=str)
    args = parser.parse_args()

    print(args)
    print('\n')
    global TrainData, TestData, SEED, IS_UNDERSAMPLED, IS_OVERSAMPLED, IS_TRAINING, IS_GRID_SEARCH

    TrainData = pd.read_csv(args.train_data)
    TestData = pd.read_csv(args.data)
    SEED = args.seed
    IS_TRAINING = args.is_only_training
    IS_GRID_SEARCH = args.is_grid_search

    balancing = args.data_balancing.lower()
    IS_OVERSAMPLED = False
    IS_UNDERSAMPLED = False
    if balancing == 'u' or balancing == 'undersampling':
        IS_UNDERSAMPLED = True
    if balancing == 'o' or balancing == 'oversampling':
        IS_OVERSAMPLED = True

    create_data_processor(IS_UNDERSAMPLED, IS_OVERSAMPLED, SEED)
    if IS_GRID_SEARCH:
        hyperparameter_grid_search()
    else:
        clf = GaussMixNaiveBayes(component_number=args.component_number, reg_covar=args.reg_covar)
        x_train, y_train, _ = process_data(True)
        clf.fit(x_train, y_train)
        print('Training AUC: {}'.format(roc_auc_score(y_train, clf.predict_proba(x_train)[:, 1])))
        if not IS_TRAINING:
            x_test, _, _ = process_data(False)
            pred = clf.predict_proba(x_test)[:, 1]
            pred = np.round(pred)

            id_codes = ['test_{}'.format(i) for i in range(x_test.shape[0])]
            df = pd.DataFrame(data=id_codes, columns=["ID_code"])
            results = processing.data_postprocessing(df, pred)

            results.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()

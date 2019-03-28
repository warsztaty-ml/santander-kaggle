import numpy as np
import pandas as pd
from sklearn import mixture


DATA_DIR = '../data/'

component_number = 5

clf = mixture.GaussianMixture(n_components=component_number, covariance_type='full', verbose=1)


# Bayesian information criterion for given data (the lower the better) - model must be trained first
def calculate_bic(data):
    return clf.bic(data)


def calculate_score(x, y):
    y2 = np.array(clf.predict(x))

    col_names = ['cluster_id', 'class', 'score', 'number of points']
    rows = []

    for i in range(component_number):
        cluster_indices = np.where(y2 == i)
        cluster = y[cluster_indices]
        class_score = np.sum(cluster == 1)/len(cluster)
        cluster_class = 1
        if class_score < 0.5:
            class_score = 1 - class_score
            cluster_class = 0
        rows.append(np.array([i, cluster_class, class_score, len(cluster)]))

    clusters = pd.DataFrame(data=rows, columns=col_names)

    print(clusters)

    score = (clusters['score']*clusters['number of points']).sum()/clusters['number of points'].sum()
    return score


def gauss_mix_train():
    print("Loading train data")
    train = pd.read_csv(DATA_DIR + 'train.csv')
    y_train = np.array(train[['target']]).flatten()
    train.drop(['ID_code', 'target'], axis=1, inplace=True)

    print("Training started")
    clf.fit(train)
    results = calculate_score(train, y_train)
    print("Training finished. \nTraining score:\n{}".format(results))

    bic = calculate_bic(train)
    print("Bayesian information criterion: {}".format(bic))


def gauss_mix_test():
    print("Loading test data")
    test = pd.read_csv(DATA_DIR + 'test.csv')
    test.drop(['ID_code'], axis=1, inplace=True)

    bic = calculate_bic(test)
    print("Bayesian information criterion: {}".format(bic))

    print("Testing started")
    results = clf.predict(test)
    print("Testing finished")

    results.to_csv(DATA_DIR + 'gaussian_mixture_results.csv', index=False)


def main():
    gauss_mix_train()
    # gauss_mix_test()


if __name__ == '__main__':
    main()
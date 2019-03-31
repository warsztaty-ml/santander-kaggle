import numpy as np
import pandas as pd
import itertools
from sklearn import mixture
import processing

DATA_DIR = '../data/'
IS_UNDERSAMPLED = True

IS_GRID_SEARCH = True
IS_TRAINING = True

component_numbers = [10, 15, 20, 25, 30, 35]
reg_covars = [1e-5, 5e-5, 1e-6, 5e-6, 1e-7]

clf = mixture.GaussianMixture(n_components=component_numbers[0], reg_covar=reg_covars[0], verbose=1)


# Bayesian information criterion for given data (the lower the better) - model must be trained first
def calculate_bic(data):
    return clf.bic(data)


# for every component of gaussian_mixture, calculates to which class that component belongs and the accuracy of that
def calculate_component_info(x, y, component_number):
    y2 = np.array(clf.predict(x))

    col_names = ['component_id', 'class', 'accuracy', 'number of points']
    rows = []

    for i in range(component_number):
        component_indices = np.where(y2 == i)
        component = y[component_indices]
        class_accuracy = np.sum(component == 1) / len(component)
        component_class = 1
        if class_accuracy < 0.5:
            class_accuracy = 1 - class_accuracy
            component_class = 0
        rows.append(np.array([i, component_class, class_accuracy, len(component)]))

    components = pd.DataFrame(data=rows, columns=col_names)
    components['class'] = components['class'].astype(np.int32)
    components['component_id'] = components['component_id'].astype(np.int32)
    components['number of points'] = components['number of points'].astype(np.int64)
    print(components)

    accuracy = (components['accuracy'] * components['number of points']).sum() / components[
        'number of points'].sum()

    return components, accuracy


def classify(x, component_info):
    y = np.array(clf.predict(x))

    classes = []
    component_class_array = component_info['class'].values
    for component_index in y:
        classes.append(component_class_array[component_index])

    return classes


def calculate_accuracy(x, y, component_info):
    y2 = classify(x, component_info)
    return np.sum(y == y2)/len(y2)


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

    return x_data, y_data, col_nr


def gauss_mix_train(x_train, y_train, component_number):
    print("Training started")
    clf.fit(x_train)
    component_info, accuracy = calculate_component_info(x_train, y_train, component_number)
    print("Training finished. \nTraining accuracy:\n{}".format(accuracy))

    bic = calculate_bic(x_train)
    print("Bayesian information criterion: {}".format(bic))
    return component_info, accuracy, bic


def gauss_mix_test(x_data, component_info):
    bic = calculate_bic(x_data)
    print("Bayesian information criterion: {}".format(bic))

    print("Testing started")
    pred = classify(x_data, component_info)
    print("Testing finished")

    id_codes = ['test_{}'.format(i) for i in range(x_data.shape[0])]
    df = pd.DataFrame(data=id_codes, columns=["ID_code"])

    results = processing.data_postprocessing(df, pred)

    results.to_csv(DATA_DIR + 'gaussian_mixture_results.csv', index=False)


def hyperparameter_grid_search():
    global clf
    x_data, y_data, col_nr = load_and_process_data(True)
    best_accuracy = -1

    col_names = ['component_number', 'reg_covar', 'accuracy', 'bic']
    rows = []

    for cn, reg_cov in itertools.product(component_numbers, reg_covars):
            clf = mixture.GaussianMixture(n_components=cn, reg_covar=reg_cov, verbose=1)
            component_info, accuracy, bic = gauss_mix_train(x_data, y_data, cn)
            print('component number: {}, reg_covar: {}, accuracy: {}, bic: {}'.format(cn, reg_cov, accuracy, bic))
            rows.append([cn, reg_cov, accuracy, bic])

    search_results = pd.DataFrame(data=rows, columns=col_names)
    search_results.sort_values(by=['accuracy'], ascending=False)

    search_results.to_csv('{}gauss_mix_grid_search.csv'.format(DATA_DIR), index=False)


def main():
    if IS_GRID_SEARCH:
        hyperparameter_grid_search()
    else:
        x_data, y_data, col_nr = load_and_process_data(IS_TRAINING)
        if IS_TRAINING:
            gauss_mix_train(x_data, y_data, component_numbers[0])
        else:
            x_train, y_train, col_nr = load_and_process_data(True)
            component_info, accuracy, bic = gauss_mix_train(x_train, y_train, component_numbers[0])
            gauss_mix_test(x_data, component_info)


if __name__ == '__main__':
    main()

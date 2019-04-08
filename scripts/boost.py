#!/usr/bin/env python3
import xgboost as xgb
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from data_processor import DataProcessor
from processing import data_postprocessing
from sklearn.decomposition import PCA
import csv
import json
import os
import math

def args():
    """Parses command line arguments given to the script using argparse
    """
    parser = argparse.ArgumentParser(description='Performs binary classification using the xgboost algorithm')

    parser.add_argument('input_data', help='Input data csv', type=str)
    parser.add_argument('--seed', help='Random seed', default=None, type=int)
    parser.add_argument('--infer', help='Do inference, not training', type=str)
    parser.add_argument('--ratio', help='Train/test split ratio', default=0.8, type=float)
    parser.add_argument('--results', help='Csv to append training results to', default='results.csv', type=str)
    parser.add_argument('-m', help='Model file, read during inference, written after training', type=str)

    args = parser.parse_args()
    print(args)
    return args.input_data, args.seed, args.infer, args.ratio, args.results, args.m

        

def exp_eta(initial, k, min_eta):
    def exp_eta_internal(epoch, epoch_count):
        lrate = initial * math.exp(-k * epoch) + min_eta
        print(lrate)
        return lrate
    return exp_eta_internal

def main():
    input_data_path, seed, file_to_infer, train_test_ratio, result_file, model_file = args()

    train_data = pd.read_csv(input_data_path)
    train, test = train_test_split(train_data, train_size=train_test_ratio, random_state=seed)
    data_processor = DataProcessor(train).with_scaling()#.with_oversampling()

    train_X, train_y, _ = data_processor.process_train()
    test_X, test_y, _ = data_processor.process_data(test)

    pca_components = 200
    pca = PCA(n_components=pca_components)
    train_X_pca = pca.fit_transform(train_X)
    test_X_pca= pca.transform(test_X)

    train = xgb.DMatrix(data=train_X_pca, label=train_y)
    test = xgb.DMatrix(data=test_X_pca, label=test_y)


    if file_to_infer is not None:
        bst = xgb.Booster()
        bst.load_model(model_file)
        data_to_infer = pd.read_csv(file_to_infer)

        data_X, _, _ = data_processor.process_test(data_to_infer)

        data_matrix = xgb.DMatrix(data=data_X)

        labels = bst.predict(data_matrix)
        submission = data_postprocessing(data_to_infer, labels)

        submission.to_csv("submission.csv", index=False)

    else:
        gamma = 0.4
        eta = 0.3
        k = 0.005
        min_eta = 0.01
        scpw = 0.11
        depth = 2


        param = {
                    'max_depth': depth, 
                    'n_estimators': 100,
                    'eta':eta, 
                    'k': k,
                    'silent':0, 
                    'gamma': gamma,
                    'nthread': 4,
                    'objective':'binary:logistic', 
                    'eval_metric':['logloss', 'auc'],
                    'min_child_weight': 1,
                    'random_state': seed,
                    'scale_pos_weight': scpw
        }

        watchlist = [(train, 'train'), (test, 'eval')]
        num_round = 10000

        history = {}
        bst = xgb.train(param, train, num_round, watchlist, early_stopping_rounds=1000, maximize=True, evals_result=history, callbacks = [xgb.callback.reset_learning_rate(exp_eta(eta, k, min_eta))])

        preds = bst.predict(test)
        labels = test.get_label()
        print('error=%f' % (sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]) / float(len(preds))))

        auc = roc_auc_score(labels, preds)
        print(auc)

        bst.save_model(model_file)

        clf = xgb.XGBClassifier(**param)
        print(clf.get_xgb_params())
        params = sorted([(k,v) for k,v in clf.get_xgb_params().items()], key=lambda tup: tup[0])
        
        print_header = not os.path.isfile(result_file)
        header = [p[0] for p in params]
        header.extend(["pca_components", "model", "auc"])
        values = [p[1] for p in params]
        values.extend([pca_components, model_file, auc])

        with open(result_file, 'a') as f:
            writer = csv.writer(f)
            if print_header:
                writer.writerow(header)
            writer.writerow(values)






    # train_data = load_csv(train_data_path)



if __name__ == "__main__":
    main()
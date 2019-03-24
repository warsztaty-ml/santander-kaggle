#!/usr/bin/env python3
import xgboost as xgb
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import RandomOverSampler, ADASYN


def args():
    """Parses command line arguments given to the script using argparse
    """
    parser = argparse.ArgumentParser(description='Performs binary classification using the xgboost algorithm')

    parser.add_argument('input_data', help='Input data csv', type=str)
    parser.add_argument('--seed', help='Random seed', default=None, type=int)
    parser.add_argument('--infer', help='Do inference, not training', action='store_true')
    parser.add_argument('--ratio', help='Train/test split ratio', default=0.8, type=float)
    parser.add_argument('-m', help='Model file, read during inference, written after training', type=str)

    args = parser.parse_args()
    print(args)
    return args.input_data, args.seed, args.infer, args.ratio, args.m


def load_csv(path, train_test_ratio, seed, load_labels):
    df = pd.read_csv(path)
    if load_labels:
        label = df.loc[:, 'target']
        data = df.loc[:, 'var_0':]
        train_x, test_x, train_y, test_y =  train_test_split(data, label, train_size=train_test_ratio, random_state=seed)
        columns = train_x.columns
        train_x, train_y = RandomOverSampler(random_state=seed).fit_resample(train_x, train_y)
        # test_x, test_y = RandomOverSampler(random_state=seed).fit_resample(test_x, test_y)
        return (xgb.DMatrix(data=pd.DataFrame(train_x, columns=columns), label=train_y), xgb.DMatrix(data=pd.DataFrame(test_x, columns=columns), label=test_y))
    else:
        data = df.loc[:, 'var_0':]
        return xgb.DMatrix(data=data)
        

def main():
    input_data_path, seed, do_infer, train_test_ratio, model_file = args()

    input_data = load_csv(input_data_path, train_test_ratio, seed, ~do_infer)

    if(do_infer):
        pass
    else:
        train, test = input_data
        param = {'max_depth':2, 'eta':1, 'silent':0, 'objective':'binary:logistic'}

        watchlist = [(test, 'eval'), (train, 'train')]
        num_round = 100

        bst = xgb.train(param, train, num_round, watchlist)

        preds = bst.predict(test)
        labels = test.get_label()
        print('error=%f' % (sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]) / float(len(preds))))

        print(roc_auc_score(labels, preds))

        bst.save_model(model_file)


    # train_data = load_csv(train_data_path)



if __name__ == "__main__":
    main()
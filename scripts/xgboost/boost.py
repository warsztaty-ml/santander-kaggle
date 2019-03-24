#!/usr/bin/env python3
import xgboost as xgb
import numpy as np
import pandas as pd
import argparse


def args():
    """Parses command line arguments given to the script using argparse
    """
    parser = argparse.ArgumentParser(description='Performs binary classification using the xgboost algorithm')

    parser.add_argument('train_data', help='Train data csv', type=str)
    parser.add_argument('test_data', help='Test data csv', type=str)

    args = parser.parse_args()
    return args.train_data, args.test_data


def load_csv(path, is_train=True):
    """Loads and returns csv file at the given path as a xgb matrix
    
    Arguments:
        path        -- path of the csv file that is loaded
        is_train  -- information whether the given file is a training file and label data needs to be loaded
    
    Returns:
        xgb.DMatrix -- matrix with the loaded data
    """

    df = pd.read_csv(path)
    if is_train:
        label = df.loc[:, 'target']
        data = df.loc[:, 'var_0':]
        return xgb.DMatrix(data=data, label=label)
    else:
        data = df.loc[:, 'var_0':]
        return xgb.DMatrix(data=data)

def main():
    train_data_path, test_data_path = args()

    train_data = load_csv(train_data_path)
    test_data = load_csv(test_data_path, is_train=False)



if __name__ == "__main__":
    main()
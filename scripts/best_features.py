import pandas as pd


def biggest_variation(num_of_features=50):
    '''Returns array of column names with biggest variation'''
    train = pd.read_csv('../data/train.csv')
    train_features = train.drop(columns=['ID_code', 'target'])
    feature_names = list(train_features.columns.values)
    train_describe = train_features.describe().T
    train_describe = train_describe.drop(columns=['count'])
    train_describe.insert(loc=0, column='feature', value=feature_names)
    train_describe = train_describe[['feature', 'std']]
    train_describe = train_describe.sort_values('std', ascending=False)
    return train_describe[0:num_of_features].as_matrix(columns=['feature'])


if __name__ == '__main__':
    print(biggest_variation(10))

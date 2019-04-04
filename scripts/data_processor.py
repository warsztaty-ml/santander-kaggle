import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

class DataProcessor():

    def __init__(self, train_data):
        self.target = 'target'
        self.id_column = "ID_code"

        self.scaler = None
        self.sampler = None

        self.train_data = self.__remove_columns(train_data, self.id_column)

    def with_scaling(self):
        self.scaler = StandardScaler()
        X, _ = self.__xy_split(self.train_data)
        self.scaler.fit(X)
        return self

    def with_undersampling(self, seed = 0):
        self.sampler = RandomUnderSampler(random_state=seed)
        return self

    def with_oversampling(self, seed = 0):
        self.sampler = RandomOverSampler(random_state=seed)
        return self


    def process_train(self):
        X, y = self.__xy_split(self.train_data)
        columns = X.columns

        if self.sampler is not None:
            X, y = self.sampler.fit_resample(X, y)
            X = pd.DataFrame(X, columns=columns)
            y = np.array(y, dtype=np.float64)

        if self.scaler is not None:
            X = self.scaler.transform(X)
            X = pd.DataFrame(X, columns=columns)

        return X, y, len(columns)

    def process_data(self, data):
        X, y = self.__xy_split(data)
        columns = X.columns

        if self.scaler is not None:
            X = self.scaler.transform(X)
            X = pd.DataFrame(X, columns=columns)

        return X, y, len(columns)

    def process_test(self, data):
        X = data
        columns = X.columns

        if self.scaler is not None:
            X = self.scaler.transform(X)
            X = pd.DataFrame(X, columns=columns)

        return X, None, len(columns)

    def __remove_columns(self, data, columns):
        return data.drop(columns, axis=1, inplace=False)

    def __xy_split(self, data):
        return self.__remove_columns(data, self.target), np.array(data[self.target], dtype=np.float64)
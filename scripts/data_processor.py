import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler



class DataProcessor():
    """Refactor of processing.py to reuse scaler from train data when scaling test data
    """
    def __init__(self, train_data):
        """Constructor, used as a base for using the builder methods
        
        Arguments:
            train_data  -- Dataframe with full training data without any preprocessing done
        """

        self.target = 'target'
        self.id_column = "ID_code"

        self.scaler = None
        self.sampler = None

        self.train_data = self.__remove_columns(train_data, self.id_column)


    def with_scaling(self):
        """Builder method to add data scaling to processor. Can be used in method chaining.
        
        Returns:
            Instance of the processor it is called on
        """

        self.scaler = StandardScaler()
        X, _ = self.__xy_split(self.train_data)
        self.scaler.fit(X)
        return self

    def with_undersampling(self, seed = 0):
        """Builder method to add undersampling. Overwrites oversampling. Can be used in method chaining.
        
        Returns:
            Instance of the processor it is called on
        """
        self.sampler = RandomUnderSampler(random_state=seed)
        return self

    def with_oversampling(self, seed = 0):
        """Builder method to add oversampling. Overwrites undersampling. Can be used in method chaining.
        
        Returns:
            Instance of the processor it is called on
        """
        self.sampler = RandomOverSampler(random_state=seed)
        return self


    def process_train(self):
        """Performs enabled data processing tasks for the train dataset.
        
        Returns:
            DataFrame with processed training data + array with training labels 
                + input size (number of columns of processed training data without target or ID columns)
        """

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
        """Performs enabled data processing tasks for a given dataset. This is used for example after splitting training data into train/val datasets,
        then this method can be used to process the val dataset.

        Arguments:
            data -- Dataframe with full data without any preprocessing done. Should also have labels.
        
        Returns:
            DataFrame with processed data + array with labels + input size (number of columns of processed data without target or ID columns)
        """
        data = self.__remove_columns(data, self.id_column)
        X, y = self.__xy_split(data)
        columns = X.columns

        if self.scaler is not None:
            X = self.scaler.transform(X)
            X = pd.DataFrame(X, columns=columns)

        return X, y, len(columns)

    def process_test(self, data):
        """Performs enabled data processing tasks for a given test dataset. This should be used for the test set from kaggle.

        Arguments:
            data -- Dataframe with full data without any preprocessing done. It should NOT have any labels.
        
        Returns:
            DataFrame with processed data + None (representing the labels, is kept for compatibility reasons) 
                + input size (number of columns of processed data without target or ID columns)
        """
        data = self.__remove_columns(data, self.id_column)
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
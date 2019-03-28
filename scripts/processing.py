import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter


def data_preprocessing(df, is_train):
    target = 'target'
    y = None
    df_out = df.drop('ID_code', axis=1, inplace=False)
    scaler = StandardScaler()
    if is_train:
        y = np.array(df[target], dtype=np.float64)
        df_out.drop('target', axis=1, inplace=True)
    df_out[df_out.columns] = scaler.fit_transform(df_out[df_out.columns])

    return df_out, y, len(df_out.columns)


def data_preprocessing_with_undersampled(df, is_train):
    target = 'target'
    y = None
    df_out = df.drop('ID_code', axis=1, inplace=False)
    scaler = StandardScaler()
    if is_train:
        print(df_out.shape)
        rus = RandomUnderSampler(random_state=0)
        X_resampled, y_resampled = rus.fit_resample(df_out, df_out[target])
        print(X_resampled.shape, sorted(Counter(y_resampled).items()))
        df_out = pd.DataFrame(X_resampled, columns=df_out.columns)
        y = np.array(df_out[target], dtype=np.float64)
        df_out.drop('target', axis=1, inplace=True)

    df_out[df_out.columns] = scaler.fit_transform(df_out[df_out.columns])

    return df_out, y, len(df_out.columns)


def data_postprocessing(df_test, pred):
    df_sub = pd.DataFrame(df_test)
    df_sub['target'] = pred
    return df_sub[["ID_code", "target"]]
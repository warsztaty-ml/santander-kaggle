#https://stackoverflow.com/questions/29432629/correlation-matrix-using-pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('../data/test.csv')

##Corr Matrix
#corr = df.corr()
#import seaborn as sns
#f, ax = plt.subplots(figsize=(10, 8))

#sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
#            square=True, ax=ax)

#plt.show()
##End Corr Matrix


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
train = pd.read_csv('../data/train.csv')
target = train['target']
train = train.drop(["ID_code", "target"], axis=1)
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train)         
PCA_train_x = PCA(.95).fit_transform(train_scaled)

print(PCA_train_x.shape)
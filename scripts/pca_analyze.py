#https://stackoverflow.com/questions/29432629/correlation-matrix-using-pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('C:\\Users\\mabien\\source\\repos\\santander-kaggle\\data/test.csv')
df = df.drop(columns=['ID_code'])
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i + 1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

def cout_outliers(data, mult=2):
    q1, q3 = np.percentile(data,[25,75])
    iqr = q3 - q1
    lower_bound = q1 - (mult * iqr) 
    upper_bound = q3 + (mult * iqr)
    
    return sum((x<lower_bound or x>upper_bound) for x in data)

##Corr Matrix
#corr = df.corr()
#import seaborn as sns
#f, ax = plt.subplots(figsize=(10, 8))

#sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
#            square=True, ax=ax)

#plt.show()
#plt.clf()

#print("Top Absolute Correlations")

#res = get_top_abs_correlations(df, 10)

#f= open("../docs/mostcorrelatedfeatures.txt","w+")
#f.write(pd.DataFrame(res).to_latex())
#f.close()
#End Corr Matrix

## Check Outliers
outliersPerColumn = []
for column in df.columns:
    data = df[column]
    outliersPerColumn.append(cout_outliers(data))

plt.plot(outliersPerColumn)
plt.ylabel("Liczba outliers w kolumnie")
plt.xlabel("Numer kolumny")
plt.savefig("outliers.eps")
plt.savefig("outliers.png")
##end Outliers

#from sklearn.decomposition import PCA
#from sklearn.preprocessing import StandardScaler
#train = pd.read_csv('../data/train.csv')
#target = train['target']
#train = train.drop(["ID_code", "target"], axis=1)
#scaler = StandardScaler()
#train_scaled = scaler.fit_transform(train)         
#PCA_train_x = PCA(.95).fit_transform(train_scaled)

#print(PCA_train_x.shape)
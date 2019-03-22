import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def compute_feature_distribution(df1, df2, label1, label2, features, file_name):
    i = 0
    plt.figure()
    fig, ax = plt.subplots(5,5,figsize=(18,22))

    for feature in features:
        i += 1
        print("Computing feature {}. distribution...".format(feature))
        plt.subplot(5,5,i)
        sns.kdeplot(df1[feature], bw=0.5,label=label1)
        sns.kdeplot(df2[feature], bw=0.5,label=label2)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)
        plt.tick_params(axis='y', which='major', labelsize=6)
    plt.savefig('{}/{}.eps'.format('.', file_name), format='eps', dpi=1000)

plt.style.use('seaborn')

train = pd.read_csv('../data/train.csv')
train_features = train.drop(columns=['ID_code', 'target'])

test = pd.read_csv('../data/test.csv')
nulls_in_train = train.isnull().sum()
nulls_in_test = test.isnull().sum()
print('Number of nulls found in train dataset ', nulls_in_train[nulls_in_train > 0].count())
print('Number of nulls found in test dataset ', nulls_in_test[nulls_in_test > 0].count())

target = train['target']

total_count = target.count()
ones_count = target[target == 1].count()
print('Positive class count', ones_count, '(', ones_count / total_count, '%)')
print('Negative class count', total_count - ones_count, '(', 1 - ones_count / total_count, '%)')

sns.set_style('whitegrid')
sns.countplot(target)

plt.ylabel('liczba przykładów')
plt.xlabel('etykieta')
plt.savefig('{}/{}.eps'.format('.', 'classes'), format='eps', dpi=1000)

feature_names = list(train_features.columns.values)
train_describe = train_features.describe().T
train_describe = train_describe.drop(columns=['count'])
train_describe.insert(loc=0, column='feature', value=feature_names)

print(train_describe.to_latex(index=False, longtable=True))

print('Stats:')
print('Max value:', train_describe[['max']].max()[0])
print('Min value:', train_describe[['min']].min()[0])

print('Mean max:', train_describe[['mean']].max()[0])
print('Mean min:', train_describe[['mean']].min()[0])

print('Mean std:', train_describe[['std']].max()[0])
print('Mean std:', train_describe[['std']].min()[0])


t0 = train.loc[train['target'] == 0]
t1 = train.loc[train['target'] == 1]
features = train.columns.values[2:27]

print("Computing train set features distribution (0 vs 1)...")
compute_feature_distribution(t0, t1, '0', '1', features, 'feature_distribution')

print("Computing features distribution (train set vs test set)...")
compute_feature_distribution(train, test, 'zestaw treningowy', 'zestaw testowy', features, "feature_distribution_train_vs_test")


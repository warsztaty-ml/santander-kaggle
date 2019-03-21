import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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

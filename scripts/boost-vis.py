#!/usr/bin/env python3
import matplotlib.pyplot as plt
import json

with open('test_long.json', 'r') as f:
    data = json.load(f)

plt.rcParams.update({'font.size': 22})

# results = [0.7101437842809597, 0.76278175313734, 0.8319161001644174, 
# 0.8547590407318241, 0.8699588926420821, 0.8734111473888255, 
# 0.8772173512484902, 0.8754645437642232, 0.8775629417943722]

# x = [10, 20, 50, 75, 100, 120, 150, 180, 200]

# plt.scatter(x, results, marker='x')

# plt.plot(data['train']['auc'], label='Training AUC')
# plt.plot(data['eval']['auc'], label='Validation AUC')
# plt.xlabel('Iteration')
# plt.ylabel('AUC')

plt.plot(data['train']['logloss'], label='Training Loss')
plt.plot(data['eval']['logloss'], label='Validation Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.legend(loc='best')
plt.show()
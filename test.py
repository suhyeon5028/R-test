import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

bream_length = pd.read_csv("bream_length.csv", header=None)
bream_weight = pd.read_csv("bream_weight.csv", header=None)
smelt_length = pd.read_csv("smelt_length.csv", header=None)
smelt_weight = pd.read_csv("smelt_weight.csv", header=None)

length = np.concatenate((bream_length, smelt_length),axis=None)
weight = np.concatenate((bream_weight, smelt_weight),axis=None)

# plt.scatter(bream_length, bream_weight)
# plt.scatter(smelt_length, smelt_weight)
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()

fish_data = [[l, w] for l, w in zip(length, weight)]

fish_target = [1] * 35 + [0]*14

input_arr = np.array(fish_data)
target_arr = np.array(fish_target)
index = np.arange(49)

np.random.seed(42)
np.random.shuffle(index)
train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]
test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]

plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(test_input[:, 0], test_input[:, 1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

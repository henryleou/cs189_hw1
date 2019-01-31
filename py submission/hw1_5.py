import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from scipy import io
import random
import sys
import sklearn.metrics
from sklearn.metrics import accuracy_score
from scipy.sparse import vstack
import save_csv

# Problem 5a

np.random.seed(10)

spam = io.loadmat("data/spam_data.mat")
num_samples = len(spam["training_data"])
index_arr = np.random.choice(num_samples, num_samples, replace=False)

def helper_kfold(sq, z):
    average = len(sq) / int(z)
    output = []
    last = 0
    while last < len(sq):
        output.append((int(last), int(last+average)))
        last = last + average
    return output

collect_validation_acc = []
#c_values = list(range(1, 100, 10))
c_values = [2**(i) for i in range(-5, 5, 1)]
all_train_data = spam['training_data'][index_arr]
all_train_labels = spam['training_labels'][index_arr].ravel()

### Method of partitioning into 5 parts
k = 5
num = list(range(0, all_train_data.shape[0]))
partition = helper_kfold(num, k)
print(partition)

for i in c_values:
    k_acc = []
    for j in np.arange(0, k, 1):
        
        # get paritions of data - valid and train
        val_num = partition[j]
        val_data = all_train_data[val_num[0]:val_num[1]]
        val_labels = all_train_labels[val_num[0]:val_num[1]]        
        t_data = np.concatenate([all_train_data[x:y] for x, y in partition if 
                                 (x, y) != val_num])
        t_labels = np.concatenate([all_train_labels[x:y] for x, y in partition if 
                                   (x, y) != val_num])
        
        # clf functions
        clf = svm.LinearSVC(C = i)
        clf.fit(t_data, t_labels)
        k_acc.append(clf.score(val_data, val_labels))
    
    print("C = {}".format(i), "Average Accuracy: {}".format(np.mean(k_acc)))
    collect_validation_acc.append(np.mean(k_acc))
    
max_num = collect_validation_acc.index(max(collect_validation_acc))
print("The C that gives maximum accuracy: {}".format(c_values[max_num]))
print("Accuracy: {}".format(max(collect_validation_acc)))

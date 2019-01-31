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

np.random.seed(420)
cifar10 = io.loadmat("data/cifar10_data.mat")
num_samples = len(cifar10["training_data"])
index_arr = np.random.choice(num_samples, num_samples, replace=False)
cifar10_validation_set_data = cifar10["training_data"][index_arr][:5000]
cifar10_validation_set_labels = cifar10["training_labels"][index_arr][:5000]
cifar10_training_set_data = cifar10["training_data"][index_arr][5000:]
cifar10_training_set_labels = cifar10["training_labels"][index_arr][5000:]

#Problem 3c
collect_train_acc = []
collect_validation_acc = []
trainings = [100, 200, 500, 1000, 2000, 5000]
for i in trainings:
    t_data = cifar10_training_set_data[:i, :-1]
    t_labels = cifar10_training_set_labels[:i, -1:].ravel()
    val_data = cifar10_validation_set_data[:, :-1]
    val_labels = cifar10_validation_set_labels[:, -1:].ravel()
    
    clf = svm.SVC(kernel='linear')
    clf.fit(t_data, t_labels)
    val_acc = clf.score(val_data, val_labels)
    collect_validation_acc.append(1-val_acc)

plt.plot(trainings, collect_validation_acc,
         label='Validation data error', marker='x')
plt.xlabel('Sample Size')
plt.ylabel('Error Rate')
plt.legend(loc='upper right', frameon=False)
plt.show()

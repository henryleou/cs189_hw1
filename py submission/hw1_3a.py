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

np.random.seed(418)
mnist = io.loadmat("data/mnist_data.mat")
num_samples = len(mnist["training_data"])
index_arr = np.random.choice(num_samples, num_samples, replace=False)

mnist_validation_set_data = mnist["training_data"][index_arr][:10000]
mnist_validation_set_labels = mnist["training_labels"][index_arr][:10000]
mnist_training_set_data = mnist["training_data"][index_arr][10000:]
mnist_training_set_labels = mnist["training_labels"][index_arr][10000:]

#Problem 3a
#collect_train_acc = []
collect_validation_acc = []
trainings = [100, 200, 500, 1000, 2000, 5000, 10000]
for i in trainings:
    t_data = mnist_training_set_data[:i, :-1]
    t_labels = mnist_training_set_labels[:i, -1:].ravel()
    val_data = mnist_validation_set_data[:, :-1]
    val_labels = mnist_validation_set_labels[:, -1:].ravel()
    clf = svm.LinearSVC()
    clf.fit(t_data, t_labels)
    val_acc = clf.score(val_data, val_labels)

    collect_validation_acc.append(1 - val_acc)
    
#plt.plot(trainings, collect_train_acc,
#         label='Training data error', marker='x')  
plt.plot(trainings, collect_validation_acc,
         label='Validation data error', marker='x')
plt.xlabel('Sample Size')
plt.ylabel('Error Rate')
plt.legend(loc='upper right', frameon=False)
plt.show()

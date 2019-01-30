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

np.random.seed(10)

# Problem 6 Kaggle cifar10 set
C = 0.1
cifar10 = io.loadmat("data/cifar10_data.mat")
num_samples = len(cifar10["training_data"])
index_arr = np.random.choice(num_samples, num_samples, replace=False)

all_train_labels = cifar10["training_labels"][index_arr][5000:]
all_train_data = cifar10["training_data"][index_arr][5000:]
all_val_labels = cifar10["training_labels"][index_arr][:5000]
all_val_data = cifar10["training_data"][index_arr][:5000]

#t_data = vstack([all_train_data, all_val_data])
#t_labels = np.append(all_train_labels.ravel(), all_val_labels.ravel())

test_data = cifar10["test_data"]
train_data = cifar10["training_data"][index_arr]
train_labels = cifar10["training_labels"][index_arr].ravel()
print("num test data" + str(len(test_data)))

num_train = 5000
t_data = train_data[:num_train]
t_labels = train_labels[:num_train]
val_data = train_data[num_train:num_train+1000]
val_labels = train_labels[num_train:num_train+1000]
print("Complete processing data")

clf = svm.SVC(C=C, kernel="poly", degree=2, verbose = False, max_iter = 100000)
#clf = svm.SVC(C=C, kernel="linear", verbose = True, max_iter = 100000)
clf.fit(t_data, t_labels)
print("Training complete.")
print("Accuracy: {}".format(clf.score(val_data, val_labels)))
t_predict = clf.predict(test_data)

save_csv.results_to_csv(t_predict)

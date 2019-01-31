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

# Problem 4a

collect_train_acc = []
collect_validation_acc = []
sample_size = 60000
# c_values = [10**(i) for i in range(-10, 5, 1)] #list(range(1, 100))
# found that the optimal values should be between -6 and -4
c_values = [10**(j) for j in np.arange(-6, -4, 0.5)] 
val_data = mnist_validation_set_data[:, :-1]
val_labels = mnist_validation_set_labels[:, -1:].ravel()
train_data = mnist_training_set_data[:sample_size, :-1]
train_labels = mnist_training_set_labels[:sample_size, -1:].ravel()
for i in c_values:
    clf = svm.LinearSVC(C=i)
    clf.fit(train_data, train_labels)
    val_acc = clf.score(val_data, val_labels)
    collect_validation_acc.append(val_acc)
    print("C =", i, ": ", val_acc)
maximum_c = collect_validation_acc.index(max(collect_validation_acc))

print("The C value for maximum accuracy:{}".format(c_values[maximum_c]))
print()
print("Accuracy of the maximum C:{}".format(max(collect_validation_acc)))

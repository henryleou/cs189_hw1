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

#Problem 6 Kaggle Mnist set
np.random.seed(420)
C = 1e-05
mnist = io.loadmat("data/mnist_data.mat")
num_samples = len(mnist["training_data"])
index_arr = np.random.choice(num_samples, num_samples, replace=False)

all_train_labels = mnist["training_labels"][index_arr][10000:]
all_train_data = mnist["training_data"][index_arr][10000:]
all_val_labels = mnist["training_labels"][index_arr][:10000]
all_val_data = mnist["training_data"][index_arr][:10000]
t_data = vstack([all_train_data, all_val_data])
t_labels = np.append(all_train_labels.ravel(), all_val_labels.ravel())
test = mnist["test_data"]
#clf = svm.SVC(C=C, kernel="poly", degree=2)
clf = svm.SVC(C=C, kernel="linear")
clf.fit(t_data, t_labels)
print(clf.score(all_val_data, all_val_labels))
t_predict = clf.predict(test)

save_csv.results_to_csv(t_predict)
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

#Problem 6 Kaggle Spam set 
np.random.seed(421)
C = 8
spam = io.loadmat("data/spam_data.mat")
num_samples = len(spam["training_data"])
index_arr = np.random.choice(num_samples, num_samples, replace=False)

spam_20 = int(spam["training_data"][index_arr].shape[0] * 0.2)
all_train_labels = spam["training_labels"][index_arr][spam_20:]
all_train_data = spam["training_data"][index_arr][spam_20:]
all_val_labels = spam["training_labels"][index_arr][:spam_20]
all_val_data = spam["training_data"][index_arr][:spam_20]
t_data = vstack([all_train_data, all_val_data])
t_labels = np.append(all_train_labels.ravel(), all_val_labels.ravel())
test = spam["test_data"]
clf = svm.SVC(C=C, kernel="poly", degree=2)
clf.fit(t_data, t_labels)
print(clf.score(all_val_data, all_val_labels))
t_predict = clf.predict(test)

# Save, override
save_csv.results_to_csv(t_predict)

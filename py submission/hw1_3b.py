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

np.random.seed(419)
spam = io.loadmat("data/spam_data.mat")
num_samples = len(spam["training_data"])
index_arr = np.random.choice(num_samples, num_samples, replace=False)

spam_20 = int(spam["training_data"][index_arr].shape[0] * 0.2)
spam_validation_set_data = spam["training_data"][index_arr][:spam_20]
spam_validation_set_labels = spam["training_labels"][index_arr][:spam_20]
spam_training_set_data = spam["training_data"][index_arr][spam_20:]
spam_training_set_labels = spam["training_labels"][index_arr][spam_20:]

#problem 3b
#collect_train_acc = []
collect_validation_acc = []
trainings = [100, 200, 500, 1000, 2000, 4138]
for i in trainings:
    
    # get train and valid data
    t_data = spam_training_set_data[:i]
    t_labels = spam_training_set_labels[:i].ravel()
    val_data = spam_validation_set_data
    val_labels = spam_validation_set_labels#.ravel()
    
    # train model
    clf = svm.LinearSVC()
    #clf = svm.SVC(kernel="linear")
    clf.fit(t_data, t_labels)
    
    # predict and cal accuracy
    val_predict = clf.predict(val_data)
    val_acc = clf.score(val_data, val_labels)

    collect_validation_acc.append(1-val_acc)
    print(val_acc)
    
#plt.plot(trainings, collect_train_acc,
#          label='Training data accuracy', marker='x')  
plt.plot(trainings, collect_validation_acc,
         label='Validation data error', marker='x')
plt.xlabel('Sample Size')
plt.ylabel('Error Rate')
plt.legend(loc='upper right', frameon=False)
plt.show()

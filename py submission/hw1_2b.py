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
#np.random.seed(0)
spam = io.loadmat("data/spam_data.mat")
num_samples = len(spam["training_data"])
index_arr = np.random.choice(num_samples, num_samples, replace=False)

spam_20 = int(spam["training_data"][index_arr].shape[0] * 0.2)
spam_validation_set_data = spam["training_data"][index_arr][:spam_20]
spam_validation_set_labels = spam["training_labels"][index_arr][:spam_20]
spam_training_set_data = spam["training_data"][index_arr][spam_20:]
spam_training_set_labels = spam["training_labels"][index_arr][spam_20:]
print("validation for spam data:{}".format(spam_validation_set_data.shape))
print("validation for spam labels:{}".format(spam_validation_set_labels.shape))
print("training for spam data:{}".format(spam_training_set_data.shape))
print("training for spam labels:{}".format(spam_training_set_labels.shape))

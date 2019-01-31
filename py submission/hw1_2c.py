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
print("validation for cifar10 data:{}".format(cifar10_validation_set_data.shape))
print("validation for cifar10 labels:{}".format(cifar10_validation_set_labels.shape))
print("training for cifar10 data:{}".format(cifar10_training_set_data.shape))
print("training for cifar10 labels:{}".format(cifar10_training_set_labels.shape))

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
print("validation for mnist data:{}".format(mnist_validation_set_data.shape))
print("validation for mnist labels:{}".format(mnist_validation_set_labels.shape))
print("training for mnist data:{}".format(mnist_training_set_data.shape))
print("training for mnist labels:{}".format(mnist_training_set_labels.shape))
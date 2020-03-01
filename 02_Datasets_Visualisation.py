# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 11:54:23 2019

@author: smile
"""
import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris
iris = load_iris()


print(iris.feature_names)  # like: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
print(iris.target_names)  # output labels like: ['setosa' 'versicolor' 'virginica']

print(iris.data[0])  # actual set of features like:  [5.1 3.5 1.4 0.2]
print(iris.target[0])  # current label

# iterate through all of the features:
for i in range(len(iris.target)):
    print("Example: %d, features: %s, label: %s" % (i, iris.data[i], iris.target[i]))

# save test index for later tests, we will not use it for training
test_idx = [0, 50, 100]

# training data
train_target = np.delete(iris.target, test_idx)  # removing 3 samples from the training labels set
train_data = np.delete(iris.data, test_idx, axis=0)  # removing 3 samples from the training data set

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# classifier training starts
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)

# out result
print(test_target)  # print this as example and compare to the next line
print(clf.predict(test_data))  # print out result and compare with previous line

# viz code
import matplotlib

vis_out = tree.plot_tree(clf.fit(iris.data, iris.target))
matplotlib.pyplot.show(vis_out)
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn import tree

print("Lets begin...")

key_features = [[140, 1], [130, 1], [150, 0], [170, 0]]  # set features here
labels = ["apple", "apple", "orange", "orange"]  # set of labels here

classifier = tree.DecisionTreeClassifier()  # just define tree, it doesn't know anything yet
classifier = classifier.fit(key_features, labels) # merge features and labels

print(classifier.predict([[100, 0]])) # print output prediction result
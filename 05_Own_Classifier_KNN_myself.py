import random
from scipy.spatial import distance


def euc(a, b):
    return distance.euclidean(a, b)  # distance between data points like Pitagorian


# own classifier here:
class ScrappyKNN():
    def fit(self, X_train, y_train):  # for training
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        self.X_test = X_test
        predictions = []  # empty list for future appends
        for row in X_test:  # going many times for every given row of data
            label = self.closest(row)  # calculate answer based on classifier giving a row of test data
            predictions.append(label)  # append it to predictions list
        return predictions

    def closest(self, row):  # defining a KNN classifier logic
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):  # for every i in training X data, start from 1 because 0 is used already
            dist = euc(row, self.X_train[i])  # find a dist between given row and X_train[i]
            if dist < best_dist:  # find a min and assign variables if min is found
                best_dist = dist
                best_index = i
        return self.y_train[best_index]  # return value of y_train with best index


from sklearn.datasets import load_iris
iris = load_iris()

X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)  # half of the labels and features is test

# define classifier #1
#from sklearn import tree
#clf = tree.DecisionTreeClassifier()

# define classifier #2
#from sklearn.neighbors import KNeighborsClassifier
#clf = KNeighborsClassifier()

# define classifier (own one)
clf = ScrappyKNN()

# train classifier
clf.fit(X_train, y_train)

#predict output
predictions = clf.predict(X_test)

# print them just to see ast example
labels = iris.target_names
name = labels[predictions]  # gets name of the label[0, 1, 2] as prediction is a number
print(name)
print(predictions)

# some stats on that
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))
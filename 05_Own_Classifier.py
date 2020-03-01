import random
from scipy.spatial import distance


def euc(a,b):
    return distance.euclidean(a,b)


# own classifier here:
class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        self.X_test = X_test
        predictions = []  # empty list for future appends
        for row in X_test:
            label = random.choice(self.y_train)  # select random from training labels data
            predictions.append(label)  # append it to predictions list
        return predictions


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
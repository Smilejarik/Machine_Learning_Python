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
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()

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
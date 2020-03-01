import csv
import pickle
from sklearn.externals import joblib


def read_txt(points):
    array_bids = []  # data as list of lists [[.], [.], ... [.]]
    array_trends = []  # data as list with same!!! length as above
    with open('C:\playground\\bids_log.txt', "r") as bids_file:
        lines = bids_file.readlines()
        print("Lines: {} \n and len: {}".format(lines, len(lines)))
        for line in lines:
            #print(len(eval(line)))
            array_bids.append(eval(line)[int(points)*(-1):])  # use last X "points" for training
        print("array bids: {} \n and len: {}".format(array_bids, len(array_bids)))
        bids_file.close()
    with open('C:\playground\\trends_log.txt', "r") as trends_file:
        lines = trends_file.readlines()
        print("Lines: {} \n and len: {}".format(lines, len(lines)))
        for line in lines:
            line = line.strip()
            if "Up" in line:
                trend_nr = 2
            elif "Down" in line:
                trend_nr = 1
            elif "No trend" in line:
                trend_nr = 0
            array_trends.append(trend_nr)
        print("array trends: {} \n and len {}".format(array_trends, len(array_trends)))
        trends_file.close()

    magic_here(array_bids, array_trends)


def magic_here(array_bids, array_trends):
    X = array_bids
    y = array_trends

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)  # half of the labels and features is test

    # define classifier #1
    from sklearn import tree
    clf = tree.DecisionTreeClassifier()

    # define classifier #2
    #from sklearn.neighbors import KNeighborsClassifier
    #clf = KNeighborsClassifier()

    # train classifier
    clf.fit(X_train, y_train)

    #predict output
    predictions = clf.predict(X_test)

    # print them just to see ast example
    labels = ["No trend", "Down", "Up"]
    trends_names = []
    for pred in predictions:
        trends_names.append(labels[pred])  # gets name of the label[0, 1, 2] as prediction is a number

    print(trends_names)
    print(predictions)

    # some stats on that
    from sklearn.metrics import accuracy_score
    print(accuracy_score(y_test, predictions))
    accuracy_array.append(round(accuracy_score(y_test, predictions), 3))


    # Save the model as a pickle in a file
    joblib.dump(clf, 'C:/playground/Model_DCTree_01.pkl')

    # Load the model from the file
    #clf_from_joblib = joblib.load('C:/playground/Model_KNN_01.pkl')

    # Use the loaded model to make predictions
    #clf_from_joblib.predict(X_test)


def write_csv(accuracy):
    with open("C:\playground\\accuracy.csv", "a+", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(accuracy)
    f.close()


if __name__ == "__main__":
    points = 200  # take last X points from the dataset, (to take all 200 should be 200)
    while points <= 200:  # this will define max nr of points to train, if your set is 200 points, can be 200
        accuracy_array = []
        counter = 0
        while counter < 1:  # repeat nr of times, nr of accuracy to add for one set of points
            read_txt(points)
            counter += 1
        print(accuracy_array)
        write_csv(accuracy_array)
        points += 20

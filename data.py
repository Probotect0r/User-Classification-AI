import csv
import numpy as np
from sklearn import cross_validation

def load_training(size):
    training_file = open('kaggle_AI_training_data.csv')
    reader = csv.reader(training_file, delimiter=',')
    d = []
    t = []
    data = {
        "training_data": [],
        "training_targets": [],
        "test_data": [],
        "test_targets": []
    }

    reader = csv.reader(training_file, delimiter=',')
    for row in reader:
        length = len(row)
        d.append(row[1:length-1])
        t.append(row[-1])
    data["training_data"], data["test_data"], data["training_targets"], data["test_targets"] = cross_validation.train_test_split(d, t, test_size=size, random_state=0)

    data["training_data"] = np.array(data["training_data"]).astype(np.float)
    data["training_targets"] = np.array(data["training_targets"]).astype(np.float)
    data["test_data"] = np.array(data["test_data"]).astype(np.float)
    data["test_targets"] = np.array(data["test_targets"]).astype(np.float)
    return data

def load_test():
    #import the training data from the csv
    training_file = open("kaggle_AI_training_data.csv")
    test_file = open("kaggle_AI_test_data.csv")

    reader = csv.reader(training_file, delimiter=",")
    reader2 = csv.reader(test_file, delimiter=",")
    d = []
    t = []
    data = {
        "training_data": [],
        "training_targets": [],
        "test_data": [],
        "ids": []
    }
    for row in reader:
        length = len(row)
        data["training_data"].append(row[1:length-1])
        data["training_targets"].append(row[-1])

    for row in reader2:
        data["ids"].append(row[0])
        length = len(row)
        data["test_data"].append(row[1:length])

    data["training_data"] = np.array(data["training_data"]).astype(np.float)
    data["training_targets"] = np.array(data["training_targets"]).astype(np.float)
    data["test_data"] = np.array(data["test_data"]).astype(np.float)

    return data

def write_predictions(ids, test_predicted_targets):
    prediction_file = open("predictions.csv", "w")
    writer = csv.writer(prediction_file, delimiter=",")
    writer.writerow(["id", "prediction"])
    for i in range(len(ids)):
        writer.writerow([ids[i], test_predicted_targets[i]])

def calculate_error(test_predicted_targets, test_targets):
    num_wrong = 0
    for i in range(len(test_targets)):
        if test_targets[i] != test_predicted_targets[i]:
            num_wrong += 1

    print "The number of wrong predictions: {}".format(num_wrong)

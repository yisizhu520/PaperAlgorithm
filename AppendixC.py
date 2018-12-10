import csv as csv
import sys as sys
import random as random
from random import shuffle
import math as math
from collections import defaultdict
from operator import itemgetter
import copy as copy
import time as time
import datetime as datetime
from math import sqrt


# this function is for importing data
def getdata(file_name):
    with open(file_name, newline='') as f:
        rowdata = []
        reader = csv.reader(f)
        for row in reader:
            for i in range(1, len(row)):
                row[i] = float(row[i])
            rowdata.append(row)
    return rowdata


def distance(x1, x2):
    distance = sqrt(sum((x1 - x2) ** 2 for x1, x2 in zip(x1, x2)))
    return distance


def normalize(data):
    for j in range(len(data)):
        for i in range(1, len(data[j])):
            data[j][i] = (data[j][i] - 1.0) / (10.0 - 1.0)
    return data


# this function is to create a stratified folded data set from a normal datase
def stratify(data, folds):
    # building a dictionary to hold all data by class which is in data[0][0]
    classes = {}
    # splitting data into classes
    for d in data:
        if d[0] not in classes:
            classes[d[0]] = []
            classes[d[0]].append(d)
    else:
        classes[d[0]].append(d)
    # n-fold stratified samples
    data = []
    for r in range(folds):
        data.append([])
        # spreading the classes evenly into all data sets
    for key, items in classes.items():
        for i in range(len(items)):
            data[i % folds].append(items[i])
    return data


# testing the prediction performance
def get_accuracy(antibodies, test_data, self_class, non_self_class):
    correct = 0.0
    incorrect = 0.0
    for x in test_data:
        yhat = predict(antibodies, x, self_class, non_self_class)
        if x[0] == yhat:
            correct += 1
            # print("correct")
        else:
            incorrect += 1
        # print("incorrect")
    accuracy = correct / float(len(test_data))
    return accuracy


def generate_random_antibody(data, parameters):
    # format: [[center], radius]
    radius = parameters["radius"]
    center = []
    for i in range(1, len(data[0])):
        center.append(random.uniform(0, 1))
    return [center, radius]


def train_population(training_set, population_size, parameters, self_class, non_self_class):
    antibodies = []
    self_class = [x for x in training_set if x[0] == self_class]
    while len(antibodies) < population_size:
        proposed_antibody = generate_random_antibody(data, parameters)
        flagged = False
        for x in self_class:
            if distance(proposed_antibody[0], x[1:]) < proposed_antibody[1]:
                flagged = True
        if flagged == False:
            antibodies.append(proposed_antibody)
        return antibodies


def predict(antibodies, x, self_class, non_self_class):
    for a in antibodies:
        if distance(a[0], x[1:]) < a[1]:
            return non_self_class
    return self_class


# testing the prediction performance
def optimized_get_accuracy(antibodies, test_data, self_class, non_self_class):
    correct = 0.0
    incorrect = 0.0
    for x in test_data:
        yhat = optimized_predict(antibodies, x, self_class, non_self_class)
        if x[0] == yhat:
            correct += 1
            # print("correct")
        else:
            incorrect += 1
            # print("incorrect")
    accuracy = correct / float(len(test_data))
    return accuracy


def optimized_train_population(training_set, population_size, parameters, self_class, non_self_class):
    antibodies = []
    original_self_class = [x for x in training_set if x[0] == self_class]
    while len(antibodies) < population_size:
        self_class = original_self_class  # this allows the selection aboveto happen only once
        proposed_antibody = generate_random_antibody(training_set, parameters)
    # select the self class points in each dimension that could be contained in by the proposed antibody
    for i in range(1, len(self_class[0])):
        self_class = [s for s in self_class if s[i] > (proposed_antibody[0][i - 1] - proposed_antibody[1]) and s[i] < (
                    proposed_antibody[0][i - 1] + proposed_antibody[1])]
        # if the self_class list is empty then add the antibody, since thereare no points in the self class contained by the hyper-cube containing the hyper-sphere
        if len(self_class) == 0:
            antibodies.append(proposed_antibody)
        # check whether the self points selected are actually contained by the hypersphere and not only the hyper cube
        else:
            flagged = False
            for s in self_class:
                if distance(proposed_antibody[0], s[1:]) < proposed_antibody[1]:
                    flagged = True
            if flagged == False:  # if there are no points that are within the hyper-sphere then add the antibody to the population
                antibodies.append(proposed_antibody)
    return antibodies


def optimized_predict(antibodies, x, self_class, non_self_class):
    # select the antibodies that could contain the point
    # for every dimension in the antibody center:
    for i in range(len(antibodies[0][0])):
        antibodies = [a for a in antibodies if x[i + 1] > (a[0][i] - a[1]) and x[i + 1] < (a[0][i] + a[1])]

    # further filter the set of antibodies
    for a in antibodies:
        if distance(a[0], x[1:]) < a[1]:
            return non_self_class
    return self_class
    # if the set of antibodies is filtered down to zero, then we know that the points is outside of the non-self class, there for it is self
    if len(antibodies) == 0:
        return self_class


# note: this script wiill be coded to only use the Breast Cancer Wisconsin Data Set
original_data = getdata("C:/Users/Brian/Documents/IPythonNotebooks/datasets/cancer.csv")
parameters = {}
parameters["radius"] = 0.93
print("population size \t accuracy")
for population_size in range(100, 1050, 50):
    # building a balanced data set
    data = []
    for c in ["benign", "malignant"]:
        class_data = [d for d in original_data if d[0] == c]
        data = data + class_data[:int(float(500) / 2.0)]
    data = normalize(data)
    data = stratify(data, 10)
    accuracy = 0.0
    for st in range(1):
        test_set = data[st % len(data)]
        validation_set = data[(st + 1) % len(data)]
        training_set = []
        for tsp in range(len(data) - 2):
            training_set = training_set + data[(st + 2 + tsp) % len(data)]
        best_r = 0
        max_accuracy = 0.0
        # find the optimal value for the radius of the antibodies
        for r in range(1, 100, 10):
            print(r / 100.0)
            parameters = {}
            parameters["radius"] = float(r) / 100.0
            antibodies = train_population(training_set, 1000, parameters, "benign", "malignant")
            accuracy = get_accuracy(antibodies, validation_set, "benign", "malignant")
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                best_r = float(r) / 100.0
        parameters = {}
        parameters["radius"] = best_r
        antibodies = train_population(training_set, population_size, parameters, "benign", "malignant")
        accuracy = accuracy + get_accuracy(antibodies, test_set, "benign", "malignant")
    print(population_size, " \t ", accuracy / 1.0)
print("")

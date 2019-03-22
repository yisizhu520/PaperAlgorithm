from algorithm.base import *
import math
from operator import itemgetter

def distance(x1, x2):
    if len(x1) != len(x2):
        return False
    else:
        distance = 0
        for i in range(len(x1)):
            distance = distance + ((float(x1[i]) - float(x2[i])) ** 2)
        distance = math.sqrt(distance)
        return distance


# testing the prediction performance
def test_accuracy(antibodies, test_data):
    error_count = 0
    correct_count = 0
    for x in test_data:
        yhat = predict(antibodies, x)
        if x[0] != yhat:
            error_count = error_count + 1
        else:
            correct_count = correct_count + 1
    return float(correct_count) / float(len(test_data))


# vote for the best classification
def predict(antibodies, x):
    distances = []
    for a in antibodies:
        d = distance(x[1], a[1])
        if d <= a[2]:
            return a[0]
        else:
            distances.append([a[0], d])
    distances.sort(key=itemgetter(1))
    return distances[0][0]


def get_class_labels(data):
    classes = []
    for i in data:
        if i[0] not in classes:
            classes.append(i[0])
    return classes


def get_class_antibody_dict(antibodies):
    classes_antibody_dict = {}
    for a in antibodies:
        if a[0] not in classes_antibody_dict:
            classes_antibody_dict[a[0]] = []
            classes_antibody_dict[a[0]].append(a)
        else:
            classes_antibody_dict[a[0]].append(a)
    return classes_antibody_dict




from algorithm.base import *
import math
from operator import itemgetter


def separate(data):
    labels = []
    dataa = []
    for i in data:
        labels.append(i[0])
        dataa.append(i[1:])
    return [labels, dataa]


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


def get_evaluation_indicator(antibodies, test_data, class_label):
    TP, TN, FP, FN = 0, 0, 0, 0
    SE, SP, TPR, FPR = 0, 0, 0, 0
    for x in test_data:
        if x[0] == class_label:
            yhat = predict(antibodies, x)
            if x[0] == yhat:
                TP += 1
            else:
                FN += 1
        else:
            yhat = predict(antibodies, x)
            if yhat == class_label:
                FP += 1
            else:
                TN += 1

    # print (TP, TN, FP, FN)
    if float(TP + FP) != 0:
        precision = float(TP) / float(TP + FP)
    else:
        precision = 0.0
    if float(TP + FN) != 0:
        recall = float(TP) / float(TP + FN)
    else:
        recall = 0.0
    if float(FP + TN) != 0:
        FPR = float(FP) / float(FP + TN)
        SP = float(TN) / float(FP + TN)
    else:
        FPR = 0.0
        SP = 0.0
    if (precision + recall) != 0:
        fmeasure = 2 * ((precision * recall) / (precision + recall))
    else:
        fmeasure = 0.0

    SE = TPR = recall
    gmean = math.sqrt(SE * SP)
    return {
        'fmeasure': fmeasure,
        'gmean': gmean,
        'TPR': TPR,
        'FPR': FPR,
    }

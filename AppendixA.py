import csv as csv
import datetime as datetime
import math as math
from operator import itemgetter
from os import listdir
from random import choice
from random import shuffle


# this function is for importing data
def getdata(file_name):
    with open(file_name, 'r') as f:
        rowdata = []
        reader = csv.reader(f)
        for row in reader:
            rowdata.append(row)
    return rowdata


def proportion_per_class(data):
    prop = {}
    for d in data:
        if d[0] not in prop.keys():
            prop[d[0]] = 1
        else:
            prop[d[0]] = prop[d[0]] + 1
    for k in prop:
        prop[k] = float(prop[k]) / float(len(data))
    return prop


def get_class_labels(data):
    classes = []
    for i in data:
        if i[0] not in classes:
            classes.append(i[0])
    return classes


def normalize(data):
    # cycling through each feature, but not the class label
    for i in range(1, len(data[0])):
        lowest = 100000000000000000
        highest = -10000000000000000
        for j in data:
            if float(j[i]) < lowest:
                lowest = float(j[i])
            if float(j[i]) > highest:
                highest = float(j[i])
        # now that we have the highest and lowest values, we can calculate the normalized value
        for j in data:
            if highest == lowest:
                j[i] = 0.5
            else:
                j[i] = (float(j[i]) - lowest) / (highest - lowest)
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


def distance(x1, x2, parameters):
    if len(x1) != len(x2):
        return False
    else:
        distance = 0
        for i in range(1, len(x1)):
            distance = distance + ((float(x1[i]) - float(x2[i])) ** 2)
        distance = math.sqrt(distance)
        return distance


# testing the prediction performance
def test_fmeasure(antibodies, test_data, class_label, parameters):
    TP, TN, FP, FN = 0, 0, 0, 0
    for x in test_data:
        if x[0] == class_label:
            yhat = predict(antibodies, x, parameters)
            if x[0] == yhat:
                TP += 1
            else:
                FN += 1
        else:
            yhat = predict(antibodies, x, parameters)
            if yhat == class_label:
                FP += 1
            else:
                TN += 1

    # print TP, TN, FP, FN
    if float(TP + FP) != 0:
        precision = float(TP) / float(TP + FP)
    else:
        precision = 0.0
    if float(TP + FN) != 0:
        recall = float(TP) / float(TP + FN)
    else:
        recall = 0.0
    if (precision + recall) != 0:
        fmeasure = 2 * ((precision * recall) / (precision + recall))
    else:
        fmeasure = 0.0
    # return "TP: " + str(TP)+ " TN: "+ str(TN)+ " FP: "+ str(FP)+ " FN: " + str(FN)
    return [precision, recall, fmeasure]


# testing the prediction performance
def test_accuracy(antibodies, test_data, parameters):
    error_count = 0
    correct_count = 0
    for x in test_data:
        yhat = predict(antibodies, x, parameters)
        if x[0] != yhat:
            error_count = error_count + 1
        else:
            correct_count = correct_count + 1
        # print "predicted: ", yhat, " actual: ", x[0]
        if len(test_data) == 0:
            print('test data is empty')
            return 0
    return float(correct_count) / float(len(test_data))


# vote for the best classification
def predict(antibodies, x, parameters):
    distances = []
    for a in antibodies:
        d = distance(x, a[0], parameters)
        if d <= a[1]:
            return a[0][0]
        else:
            distances.append([a[0][0], d])
    distances.sort(key=itemgetter(1))
    return distances[0][0]


def error_count(antibody, training_set, parameters):
    error_count = 0
    class_data = [i for i in training_set if i[0] != antibody[0][0]]
    for t in class_data:
        # print('distance--%d', distance(t, antibody[0], parameters))
        if distance(t, antibody[0], parameters) <= antibody[1]:
            error_count = error_count + 1
    return error_count


def generate_population(training_set, classes, size, parameters):
    antibodies = []
    # select random antibodies from the self class, and add with a radius of 0
    for c in classes:
        class_data = [i for i in training_set if i[0] == c]
        if len(class_data) == 0:
            continue
        num_of_antibodies = int(float(size) / float(len(classes)))
        for i in range(num_of_antibodies):
            proposed_antibody = [choice(class_data), 0.0]
            antibodies.append(proposed_antibody)
    # expand the antibodies by a step size until it misclassify a non-self point
    for a in antibodies:
        changed = True
        while changed:
            if error_count(a, training_set, parameters) > 0:
                a[1] = a[1] - parameters["step_size"]
                changed = False
            else:
                a[1] = a[1] + parameters["step_size"]
                changed = True
    return antibodies


# structure of antibody: [ [class, x1, x2, x3,... ], radius]
files = [f for f in listdir("data/")]
original_data = []
for f in files:
    original_data = original_data + getdata("data/" + f)
classes = get_class_labels(original_data)
set_size = 1000
parameters = {}
parameters["step_size"] = 0.05
parameters["c"] = 0.08
parameters["d"] = 2
print("Fig 2. Classification accuracy with data set size held at 1000 samples")
print("set_size \t popsize \t accuracy")
for pop_size in range(100, 1100, 50):
    average_accuracy = 0.0
    # building a balanced data set
    data = []
    for c in classes:
        class_data = [d for d in original_data if d[0] == c]
        shuffle(class_data)
        data = data + class_data[:int(float(1000) / float(len(classes)))]
    data = normalize(data)
    data = stratify(data, 10)
    for st in range(len(data)):
        test_set = data[st % len(data)] + data[(st + 1) % len(data)] + data[(st + 2) % len(data)]
        training_set = []
        for tsp in range(len(data) - 1):
            training_set = training_set + data[(st + 3 + tsp) % len(data)]
        antibodies = generate_population(training_set, classes, pop_size, parameters)
        accuracy = test_accuracy(antibodies, test_set, parameters)
        average_accuracy = average_accuracy + accuracy
    print("1000 \t ", pop_size, " \t ", average_accuracy / 10.0)
print("")
print("")

print("Fig 3. Classification accuracy and data set size 200-1000")
# average accuracy over sample size, three training methods, x=# of flows, y = average accuracy,
# three training methods, three lines
print("set_size \t popsize \t accuracy")
for sample_size in range(100, 1100, 50):
    average_accuracy = 0.0

    # building a balanced data set
    data = []
    for c in classes:
        class_data = [d for d in original_data if d[0] == c]
        shuffle(class_data)
        data = data + class_data[:int(float(sample_size) / float(len(classes)))]

    data = normalize(data)
    data = stratify(data, 10)
    for st in range(len(data)):
        test_set = data[st % len(data)] + data[(st + 1) % len(data)] + data[(st + 2) % len(data)]
        training_set = []
        for tsp in range(len(data) - 1):
            training_set = training_set + data[(st + 3 + tsp) % len(data)]
        antibodies = generate_population(training_set, classes, 1000, parameters)
        accuracy = test_accuracy(antibodies, test_set, parameters)
        average_accuracy = average_accuracy + accuracy
    print(sample_size, " \t 1000 \t ", average_accuracy / 10.0)
print("")
print("")

print("Fig 4. Classification time with data set size at 1000")
print("set_size \t popsize \t classification time")
for pop_size in range(100, 1100, 50):
    average_time = 0.0

    # building a balanced data set
    data = []
    for c in classes:
        class_data = [d for d in original_data if d[0] == c]
        shuffle(class_data)
        data = data + class_data[:int(float(1000) / float(len(classes)))]
    data = normalize(data)
    data = stratify(data, 10)

    for st in range(10):
        test_set = data[st % len(data)] + data[(st + 1) % len(data)] + data[(st + 2) % len(data)]
        training_set = []
        for tsp in range(len(data) - 1):
            training_set = training_set + data[(st + 3 + tsp) % len(data)]
        antibodies = generate_population(training_set, classes, pop_size, parameters)
        t1 = datetime.datetime.now()
        accuracy = test_accuracy(antibodies, test_set, parameters)
        t2 = datetime.datetime.now()
        average_time = float(average_time) + float(datetime.timedelta.total_seconds(t2 - t1))
    print(" 1000 \t ", pop_size, average_time / 10.0)
print("")
print("")

print("Fig 5. Training time and training dataset size 200-1000")
print("set_size \t popsize \t training time")
for set_size in range(100, 1100, 50):
    average_time = 0.0

    # building a balanced data set
    data = []
    for c in classes:
        class_data = [d for d in original_data if d[0] == c]
        shuffle(class_data)
        data = data + class_data[:int(float(set_size) / float(len(classes)))]

    data = normalize(data)
    data = stratify(data, 10)

    for st in range(10):
        test_set = data[st % len(data)] + data[(st + 1) % len(data)] + data[(st + 2) % len(data)]
        training_set = []
        for tsp in range(len(data) - 1):
            training_set = training_set + data[(st + 3 + tsp) % len(data)]
        t1 = datetime.datetime.now()
        antibodies = generate_population(training_set, classes, 1000, parameters)
        t2 = datetime.datetime.now()
        average_time = float(average_time) + float(datetime.timedelta.total_seconds(t2 - t1))
    print(set_size, " \t 1000 \t ", average_time / 10.0)
print("")

import csv
from algorithm.base import *
from random import shuffle
import pandas as pd
import copy


def get_default_parameters():
    parameters = {}
    parameters["step_size"] = 0.1
    parameters['pop_min_size'] = 100
    parameters['pop_size_increment'] = 100
    parameters['pop_max_size'] = 300
    parameters['iteration_count'] = 3
    return parameters

def normalize(data):
    # cycling through each feature, but not the class label
    for i in range(len(data[0][1])):
        lowest = 100000000000000000
        highest = -10000000000000000
        for j in data:
            if float(j[1][i]) < lowest:
                lowest = j[1][i]
            if float(j[1][i]) > highest:
                highest = j[1][i]
        # now that we have the highest and lowest values, we can calculate the normalized value
        for j in data:
            if highest == lowest:
                j[1][i] = 0.5
            else:
                j[1][i] = (j[1][i] - lowest) / (highest - lowest)
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


# this function is for importing data
def getdata(file_name):
    with open(file_name, newline='') as f:
        rowdata = []
        reader = csv.reader(f)
        for row in reader:
            for i in range(1, len(row)):
                row[i] = float(row[i])
            rowdata.append([row[0], row[1:]])
    return rowdata


def prepare_data(original_data, classes):
    data = []
    for c in classes:
        class_data = [d for d in original_data if d[0] == c]
        shuffle(class_data)
        data = data + class_data
    data = normalize(data)
    data = stratify(data, 10)
    return data


def get_accuracy_by_class(antibodies, test_data, training_set):
    classes_data_dict = {}
    error_count = 0
    correct_count = 0
    for a in test_data:
        if a[0] not in classes_data_dict:
            classes_data_dict[a[0]] = []
            classes_data_dict[a[0]].append(a)
        else:
            classes_data_dict[a[0]].append(a)

    classes_train_data_dict = {}
    for a in training_set:
        if a[0] not in classes_train_data_dict:
            classes_train_data_dict[a[0]] = []
            classes_train_data_dict[a[0]].append(a)
        else:
            classes_train_data_dict[a[0]].append(a)

    classes_antibody_dict = get_class_antibody_dict(antibodies)
    class_result_dict = {}

    for cls in classes_data_dict:
        class_result_dict[cls] = {}
        data = classes_data_dict[cls]
        cls_correct_count = 0
        for x in data:
            yhat = predict(antibodies, x)
            if x[0] != yhat:
                error_count = error_count + 1
            else:
                correct_count = correct_count + 1
                cls_correct_count = cls_correct_count + 1
        train_count = 0
        anti_count = 0
        if cls in classes_train_data_dict:
            train_count = len(classes_train_data_dict[cls])
        if cls in classes_antibody_dict:
            anti_count = len(classes_antibody_dict[cls])
        class_result_dict[cls]['accuracy'] = float(cls_correct_count) / float(len(data))
        class_result_dict[cls]['test-count'] = len(data)
        class_result_dict[cls]['train-count'] = train_count
        class_result_dict[cls]['anti-count'] = anti_count
    class_result_dict['summary'] = {}
    class_result_dict['summary']['accuracy'] = float(correct_count) / float(len(test_data))
    class_result_dict['summary']['test-count'] = len(test_data)
    class_result_dict['summary']['train-count'] = len(training_set)
    class_result_dict['summary']['anti-count'] = len(antibodies)
    return class_result_dict


def output_vs_csv(before_func, after_func, data_name, csv_name, parameters):
    # new structure of antibody: [ class, [x1, x2, x3,... ], radius]
    original_data = getdata('../data/' + data_name)
    classes = get_class_labels(original_data)
    pop_min_size = parameters['pop_min_size']
    pop_size_increment = parameters['pop_size_increment']
    pop_max_size = parameters['pop_max_size']
    iteration_count = parameters['iteration_count']

    classes_with_summary = classes.copy()
    classes_with_summary.append('summary')
    pop_class_dict = {}
    for cls in classes_with_summary:
        pop_class_dict[cls] = {}

    headers = ['class', 'train-count', 'test-count']
    for pop_size in range(pop_min_size, pop_max_size + pop_size_increment, pop_size_increment):
        # building a balanced data set

        data = prepare_data(original_data, classes)
        class_dict = {}
        key_before = '{}-before-accuracy'.format(pop_size)
        key_after = '{}-after-accuracy'.format(pop_size)
        key_test_count = 'test-count'
        key_train_count = 'train-count'
        key_before_anti_count = '{}-before-anti-count'.format(pop_size)
        key_after_anti_count = '{}-after-anti-count'.format(pop_size)

        headers += [key_before, key_after, key_before_anti_count, key_after_anti_count]

        dict = {}
        dict[key_before] = 0.0
        dict[key_after] = 0.0
        dict[key_test_count] = 0
        dict[key_train_count] = 0
        dict[key_before_anti_count] = 0
        dict[key_after_anti_count] = 0
        for cls in classes_with_summary:
            class_dict[cls] = copy.deepcopy(dict)
        for st in range(iteration_count):
            test_set = data[st % len(data)]
            validation_set = data[(st + 1) % len(data)]
            training_set = []
            for tsp in range(len(data) - 2):
                training_set = training_set + data[(st + 2 + tsp) % len(data)]

            old_antibodies = before_func(training_set, classes, pop_size, parameters)
            result_before = get_accuracy_by_class(old_antibodies, test_set, training_set)

            new_antibodies = after_func(training_set, classes, pop_size, parameters)
            result_after = get_accuracy_by_class(new_antibodies, test_set, training_set)

            for cls in classes_with_summary:
                if cls in result_before:
                    class_dict[cls][key_before] += result_before[cls]['accuracy'] / float(iteration_count)
                    class_dict[cls][key_test_count] += result_before[cls]['test-count'] / float(iteration_count)
                    class_dict[cls][key_train_count] += result_before[cls]['train-count'] / float(iteration_count)
                    class_dict[cls][key_before_anti_count] += result_before[cls]['anti-count'] / float(iteration_count)
                if cls in result_after:
                    class_dict[cls][key_after] += result_after[cls]['accuracy'] / float(iteration_count)
                    class_dict[cls][key_after_anti_count] += result_after[cls]['anti-count'] / float(iteration_count)

        for cls in classes_with_summary:
            class_dict[cls][key_test_count] = int(class_dict[cls][key_test_count])
            class_dict[cls][key_train_count] = int(class_dict[cls][key_train_count])
            class_dict[cls][key_before_anti_count] = int(class_dict[cls][key_before_anti_count])
            class_dict[cls][key_after_anti_count] = int(class_dict[cls][key_after_anti_count])

        for cls in classes_with_summary:
            pop_class_dict[cls]['class'] = cls
            pop_class_dict[cls] = {**pop_class_dict[cls], **class_dict[cls]}

    df = pd.DataFrame.from_dict(pop_class_dict, orient='index', columns=headers)
    df.sort_values(by='test-count', ascending=False, inplace=True)
    df.to_csv('../result/' + csv_name, index=False, header=True)
    print('../result/' + csv_name + " finished")

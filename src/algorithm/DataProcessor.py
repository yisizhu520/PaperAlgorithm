import csv
from algorithm.base import *
from random import shuffle
import pandas as pd
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import algorithm.OtherAlgorithm as OtherAlgorithm
import algorithm.NegativeSelection as NegativeSelection
import os
import time


def get_default_parameters():
    parameters = {}
    parameters["step_size"] = 0.1
    parameters['pop_min_size'] = 100
    parameters['pop_size_increment'] = 100
    parameters['pop_max_size'] = 200
    parameters['iteration_count'] = 2
    return parameters


def format_fpr_tpr(fprs, tprs):
    dict = {}
    for i in range(len(fprs)):
        dict[fprs[i]] = tprs[i]
    fprs = sorted(fprs)
    new_tprs = []
    for i in range(len(tprs)):
        new_tprs.append(dict[fprs[i]])
    fprs.insert(0, 0)
    fprs.append(1)
    new_tprs.insert(0, 0)
    new_tprs.append(1)
    return fprs, new_tprs


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


def prepare_data(original_data, classes, fold_count):
    data = []
    for c in classes:
        class_data = [d for d in original_data if d[0] == c]
        shuffle(class_data)
        data = data + class_data
    data = normalize(data)
    data = stratify(data, fold_count)
    return data


def clean_test_data(test_data, training_data):
    result = []
    training_classes = get_class_labels(training_data)
    for data in test_data:
        if data[0] in training_classes:
            result.append(data)
    return result


def get_evaluation_data_by_class(antibodies, test_data, training_set):
    test_data = clean_test_data(test_data, training_set)
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
    fmeasure_sum = 0.0
    gmean_sum = 0.0
    TPR_sum = 0.0
    FPR_sum = 0.0
    predict_time_sum = 0
    for cls in classes_data_dict:
        class_result_dict[cls] = {}
        data = classes_data_dict[cls]
        cls_correct_count = 0
        predict_time = 0
        for x in data:
            start_time = time.time()
            yhat = predict(antibodies, x)
            predict_time += time.time() - start_time
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
        class_result_dict[cls]['predict_time'] = predict_time
        eva_indicator = get_evaluation_indicator(antibodies, test_data, cls)
        class_result_dict[cls]['fmeasure'] = eva_indicator['fmeasure']
        class_result_dict[cls]['gmean'] = eva_indicator['gmean']
        class_result_dict[cls]['TPR'] = eva_indicator['TPR']
        class_result_dict[cls]['FPR'] = eva_indicator['FPR']
        fmeasure_sum += eva_indicator['fmeasure']
        gmean_sum += eva_indicator['gmean']
        TPR_sum += eva_indicator['TPR']
        FPR_sum += eva_indicator['FPR']
        predict_time_sum += predict_time
    class_result_dict['summary'] = {}
    class_result_dict['summary']['accuracy'] = float(correct_count) / float(len(test_data))
    class_result_dict['summary']['test-count'] = len(test_data)
    class_result_dict['summary']['train-count'] = len(training_set)
    class_result_dict['summary']['anti-count'] = len(antibodies)
    class_result_dict['summary']['fmeasure'] = fmeasure_sum / len(classes_data_dict)
    class_result_dict['summary']['gmean'] = gmean_sum / len(classes_data_dict)
    class_result_dict['summary']['TPR'] = TPR_sum / len(classes_data_dict)
    class_result_dict['summary']['FPR'] = FPR_sum / len(classes_data_dict)
    class_result_dict['summary']['predict_time'] = predict_time_sum
    return class_result_dict


def compare_with_other_algorithm(data_name, file_name, parameters):
    iteration_count = parameters['iteration_count']
    pop_size = parameters['pop_size']
    original_data = getdata('data/' + data_name)
    classes = get_class_labels(original_data)
    data = prepare_data(original_data, classes, iteration_count)
    other_algorithms = ['svm', 'naive_bayes', 'decision_tree', 'neural_network', 'knn']
    chart_dict = {'nega': {'TPR': [], 'FPR': [], 'predict_time': 0}}
    for algo in other_algorithms:
        chart_dict[algo] = {'TPR': [], 'FPR': [], 'predict_time': 0}

    for st in range(iteration_count):
        test_set = data[st % len(data)]
        training_set = []
        for tsp in range(len(data) - 1):
            training_set = training_set + data[(st + 2 + tsp) % len(data)]

        old_antibodies = NegativeSelection.generate_population(training_set, classes, pop_size, parameters)
        result_nega = get_evaluation_data_by_class(old_antibodies, test_set, training_set)
        # collect roc chart data
        chart_dict['nega']['TPR'].append(result_nega['summary']['TPR'])
        chart_dict['nega']['FPR'].append(result_nega['summary']['FPR'])
        chart_dict['nega']['fmeasure'] = result_nega['summary']['fmeasure']
        chart_dict['nega']['gmean'] = result_nega['summary']['gmean']
        chart_dict['nega']['predict_time'] += result_nega['summary']['predict_time'] / iteration_count
        for algo in other_algorithms:
            result_other = OtherAlgorithm.get_evaluation_indicator(algo, training_set, test_set)
            chart_dict[algo]['TPR'].append(result_other['TPR'])
            chart_dict[algo]['FPR'].append(result_other['FPR'])
            chart_dict[algo]['fmeasure'] = result_other['fmeasure']
            chart_dict[algo]['gmean'] = result_other['gmean']
            chart_dict[algo]['predict_time'] += result_other['predict_time'] / iteration_count

    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    other_algorithms.insert(0, 'nega')
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'deeppink', 'navy', 'aqua']
    line_styles = ['solid', 'dashed', 'dashdot', 'dotted', 'solid', 'dashed']
    line_widths = [1, 1, 1, 1, 2, 2]
    for i in range(len(other_algorithms)):
        algo = other_algorithms[i]
        chart_dict[algo]['algorithm'] = algo
        FPRs, TPRs = format_fpr_tpr(chart_dict[algo]['FPR'], chart_dict[algo]['TPR'])
        roc_auc = auc(FPRs, TPRs)
        plt.plot(FPRs, TPRs, color=colors[i], linestyle=line_styles[i],
                 lw=lw, label='%s (area = %0.2f)' % (algo, roc_auc), linewidth=line_widths[i])
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(file_name)
    plt.legend(loc="lower right")
    file = 'result/{}.png'.format(file_name)
    if os.path.exists(file):
        os.remove(file)
    plt.savefig(file)
    # plt.show()

    headers = ['algorithm', 'fmeasure', 'gmean', 'predict_time']
    df = pd.DataFrame.from_dict(chart_dict, orient='index', columns=headers)
    df.sort_values(by='fmeasure', ascending=False, inplace=True)
    file = 'result/' + file_name + '.csv'
    if os.path.exists(file):
        os.remove(file)
    df.to_csv(file, index=False, header=True)
    print('result/' + file_name + " finished")


def output_vs_csv_and_chart(before_func, after_func, data_name, file_name, parameters):
    # new structure of antibody: [ class, [x1, x2, x3,... ], radius]
    original_data = getdata('data/' + data_name)
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
    pop_chart_dict = {}
    for pop_size in range(pop_min_size, pop_max_size + pop_size_increment, pop_size_increment):
        # building a balanced data set

        data = prepare_data(original_data, classes, iteration_count)
        class_dict = {}
        key_before = '{}-before-accuracy'.format(pop_size)
        key_after = '{}-after-accuracy'.format(pop_size)
        key_test_count = 'test-count'
        key_train_count = 'train-count'
        key_before_anti_count = '{}-before-anti-count'.format(pop_size)
        key_after_anti_count = '{}-after-anti-count'.format(pop_size)
        key_before_fmeasure = '{}-before-fmeasure'.format(pop_size)
        key_after_fmeasure = '{}-after-fmeasure'.format(pop_size)
        key_before_gmean = '{}-before-gmean'.format(pop_size)
        key_after_gmean = '{}-after-gmean'.format(pop_size)

        headers += [key_before, key_after, key_before_anti_count, key_after_anti_count,
                    key_before_fmeasure, key_after_fmeasure, key_before_gmean, key_after_gmean]

        dict = {}
        dict[key_before] = 0.0
        dict[key_after] = 0.0
        dict[key_test_count] = 0
        dict[key_train_count] = 0
        dict[key_before_anti_count] = 0
        dict[key_after_anti_count] = 0
        dict[key_before_fmeasure] = 0
        dict[key_after_fmeasure] = 0
        dict[key_before_gmean] = 0
        dict[key_after_gmean] = 0
        for cls in classes_with_summary:
            class_dict[cls] = copy.deepcopy(dict)

        # {class:{before:{tpr:[],fpr:[]},after:{tpr:[],fpr:[]}}
        chart_dict = {'before': {'TPR': [], 'FPR': []}, 'after': {'TPR': [], 'FPR': []}}

        for st in range(iteration_count):
            test_set = data[st % len(data)]
            training_set = []
            for tsp in range(len(data) - 1):
                training_set = training_set + data[(st + 2 + tsp) % len(data)]

            old_antibodies = before_func(training_set, classes, pop_size, parameters)
            result_before = get_evaluation_data_by_class(old_antibodies, test_set, training_set)
            print('result_before-' + str(st) + '-popsize-' + str(pop_size))

            new_antibodies = after_func(training_set, classes, pop_size, parameters)
            result_after = get_evaluation_data_by_class(new_antibodies, test_set, training_set)
            print('result_after-' + str(st) + '-popsize-' + str(pop_size))

            for cls in classes_with_summary:
                if cls in result_before:
                    class_dict[cls][key_before] += result_before[cls]['accuracy'] / float(iteration_count)
                    class_dict[cls][key_test_count] += result_before[cls]['test-count'] / float(iteration_count)
                    class_dict[cls][key_train_count] += result_before[cls]['train-count'] / float(iteration_count)
                    class_dict[cls][key_before_anti_count] += result_before[cls]['anti-count'] / float(iteration_count)
                    class_dict[cls][key_before_fmeasure] += result_before[cls]['fmeasure'] / float(iteration_count)
                    class_dict[cls][key_before_gmean] += result_before[cls]['gmean'] / float(iteration_count)
                if cls in result_after:
                    class_dict[cls][key_after] += result_after[cls]['accuracy'] / float(iteration_count)
                    class_dict[cls][key_after_anti_count] += result_after[cls]['anti-count'] / float(iteration_count)
                    class_dict[cls][key_after_fmeasure] += result_after[cls]['fmeasure'] / float(iteration_count)
                    class_dict[cls][key_after_gmean] += result_after[cls]['gmean'] / float(iteration_count)

            # collect roc chart data
            chart_dict['before']['TPR'].append(result_before['summary']['TPR'])
            chart_dict['before']['FPR'].append(result_before['summary']['FPR'])
            chart_dict['after']['TPR'].append(result_after['summary']['TPR'])
            chart_dict['after']['FPR'].append(result_after['summary']['FPR'])

        pop_chart_dict[pop_size] = chart_dict

        for cls in classes_with_summary:
            class_dict[cls][key_test_count] = int(class_dict[cls][key_test_count])
            class_dict[cls][key_train_count] = int(class_dict[cls][key_train_count])
            class_dict[cls][key_before_anti_count] = int(class_dict[cls][key_before_anti_count])
            class_dict[cls][key_after_anti_count] = int(class_dict[cls][key_after_anti_count])

        for cls in classes_with_summary:
            pop_class_dict[cls]['class'] = cls
            pop_class_dict[cls] = {**pop_class_dict[cls], **class_dict[cls]}

    # generate roc chart
    generate_roc_chart(pop_chart_dict, file_name)

    df = pd.DataFrame.from_dict(pop_class_dict, orient='index', columns=headers)
    df.sort_values(by='test-count', ascending=False, inplace=True)
    file = 'result/' + file_name + '.csv'
    if os.path.exists(file):
        os.remove(file)
    df.to_csv(file, index=False, header=True)
    print('result/' + file_name + " finished")


def generate_roc_chart(pop_chart_dict, file_name):
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    colors = ['aqua', 'darkorange']
    line_styles = ['solid', 'dashed']
    count = 1
    x = []
    y = []
    for size in pop_chart_dict:
        chart_dict = pop_chart_dict[size]
        # 计算auc的值
        before_FPRs, before_TPRs, = format_fpr_tpr(chart_dict['before']['FPR'], chart_dict['before']['TPR'])
        after_FPRs, after_TPRs, = format_fpr_tpr(chart_dict['after']['FPR'], chart_dict['after']['TPR'])
        x += before_FPRs + after_FPRs
        y += before_TPRs + after_TPRs
        roc_auc_before = auc(before_FPRs, before_TPRs)
        roc_auc_after = auc(after_FPRs, after_TPRs)
        # 假正率为横坐标，真正率为纵坐标做曲线
        plt.plot(before_FPRs, before_TPRs, color='darkorange', linewidth=count,
                 lw=lw, label='%d before (AUC area = %0.2f)' % (size, roc_auc_before), linestyle='solid')
        plt.plot(after_FPRs, after_TPRs, color='green', linewidth=count,
                 lw=lw, label='%d after (AUC area = %0.2f)' % (size, roc_auc_before), linestyle='dashed')
        count += 1
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='dotted')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(file_name)
    plt.legend(loc="lower right")
    file = 'result/{}.png'.format(file_name)
    if os.path.exists(file):
        os.remove(file)
    plt.savefig(file)
    # plt.show()

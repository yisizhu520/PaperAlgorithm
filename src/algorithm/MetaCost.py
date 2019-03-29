import copy
import algorithm.NegativeSelection as NegativeSelection
from algorithm.base import *
import random

def generate_population(training_set, classes, size, parameters):
    init_antibodies = NegativeSelection.generate_population(training_set, classes, size, parameters)
    return generate_population_with_antibodies(init_antibodies, training_set, classes, size, parameters)


def generate_population_with_antibodies(init_antibodies, training_set, classes, size, parameters):
    training_set = copy.deepcopy(training_set)

    for data in training_set:
        # Step3：使用检测器检测所有样本，得到样本基于每一类的先验概率
        class_prob_dict = get_class_bayes_prob(data, init_antibodies, training_set)
        class_cost_dict = get_class_cost(data, training_set)
        cost_dict = {}
        # Step4：根据代价矩阵C，计算使代价最大的类别，即为当前样本的“真实类”，根据真实类，再次生成检测器
        for c in class_prob_dict:
            if c not in class_cost_dict:
                cost_dict[c] = 0
            else:
                cost_dict[c] = class_prob_dict[c] * class_cost_dict[c]
        cost_dict = sorted(cost_dict.items(), key=lambda d: d[1], reverse=True)
        real_class = cost_dict[0][0]
        if cost_dict[0][1] > 0 and real_class != data[0]:
            print('origin class is ' + data[0] + ', real class is ' + real_class)
            data[0] = real_class

    return NegativeSelection.generate_population(training_set, classes, size, parameters)


def get_class_prob(data, antibodies, training_set):
    result = {}
    class_hit_dict = {}
    classes = get_class_labels(antibodies)
    total_hit_count = 0
    for c in classes:
        class_hit_dict[c] = 0

    for a in antibodies:
        d = distance(data[1], a[1])
        if d <= a[2]:
            class_hit_dict[a[0]] = class_hit_dict[a[0]] + 1
            total_hit_count = total_hit_count + 1

    for hit in class_hit_dict:
        if total_hit_count == 0:
            result[hit] = 0
        else:
            result[hit] = class_hit_dict[hit] / float(total_hit_count)
    return result

import operator

def get_class_bayes_prob(data, antibodies, training_set):
    class_data_dict = get_class_data_dict(training_set)
    classes = get_class_labels(antibodies)
    class_bayes_prob = {}
    for c in classes:
        class_data = class_data_dict[c]
        same_count = cal_same_count(data, class_data)
        # 类C中的样本跟data一毛一样的比例
        pxi = float(same_count) / (len(class_data))
        # 类C数据占总体的比例
        pi = len(class_data) / (len(training_set))
        # 总体中跟data一毛一样的比例
        px = cal_same_count(data, training_set) / (len(training_set))
        pix = pxi * pi / px
        class_bayes_prob[c] = pix
    return class_bayes_prob

def cal_same_count(data, class_data_set):
    count = 0
    for x in class_data_set:
        if operator.eq(data[1], x[1]):
            count += 1
    return count



def get_class_ratio(data, training_set):
    class_count_dict = {}
    classes = get_class_labels(training_set)
    for c in classes:
        class_count_dict[c] = 0

    for t in training_set:
        class_count_dict[t[0]] = class_count_dict[t[0]] + 1
    return class_count_dict[data[0]] / float(len(training_set))


def get_class_cost(data, training_set):
    result = {}
    class_count_dict = {}
    classes = get_class_labels(training_set)
    for c in classes:
        class_count_dict[c] = 0

    for t in training_set:
        class_count_dict[t[0]] = class_count_dict[t[0]] + 1

    for c in class_count_dict:
        result[c] = ( class_count_dict[data[0]] / float(class_count_dict[c])) * (2 + random.random())
    return result

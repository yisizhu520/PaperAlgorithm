import copy

from algorithm.NegativeSelection import *


def generate_population_by_metacost(training_set, classes, size, parameters):
    training_set = copy.deepcopy(training_set)
    init_antibodies = generate_population(training_set, classes, size, parameters)

    for data in training_set:
        # Step3：使用检测器检测所有样本，得到样本基于每一类的先验概率
        class_prob_dict = get_class_prob(data, init_antibodies)
        class_cost_dict = get_class_cost(data, training_set)
        cost_dict = {}
        # Step4：根据代价矩阵C，计算使代价最小的类别，即为当前样本的“真实类”，根据真实类，再次生成检测器
        for c in class_prob_dict:
            cost_dict[c] = class_prob_dict[c] * class_cost_dict[c]
        cost_dict = sorted(cost_dict.items(), key=lambda d: d[1], reverse=True)
        real_class = cost_dict[0][0]
        if real_class != data[0]:
            # print('origin class is ' + data[0] + ', real class is ' + real_class)
            data[0] = real_class

    return generate_population(training_set, classes, size, parameters)


def get_class_prob(data, antibodies):
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
            result[hit] = 0.0
        else:
            result[hit] = class_hit_dict[hit] / float(total_hit_count)
    return result


def get_class_cost(data, training_set):
    result = {}
    class_count_dict = {}
    classes = get_class_labels(training_set)
    for c in classes:
        class_count_dict[c] = 0

    for t in training_set:
        class_count_dict[t[0]] = class_count_dict[t[0]] + 1

    for c in class_count_dict:
        result[c] = float(class_count_dict[c]) / class_count_dict[data[0]]
    return result

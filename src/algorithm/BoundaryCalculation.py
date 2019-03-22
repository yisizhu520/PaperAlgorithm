import algorithm.NegativeSelection as NegativeSelection
from algorithm.KDTree import KDTree
from algorithm.base import *


def generate_population(training_set, classes, size, parameters):
    init_antis = NegativeSelection.generate_population(training_set, classes, size, parameters)
    return generate_population_with_antibodies(init_antis, training_set, parameters)



def generate_population_with_antibodies(init_antibodies, training_set, parameters):
    # not_hit_set = get_not_hit_data(init_antibodies, training_set)
    # the count which is more than 10% of total data is major class
    judge_ratio = 0.1
    k = 10
    classes = get_class_labels(training_set)
    major_class_set = get_major_class_set(training_set, judge_ratio, classes)
    minority_class_set = list(set(classes).difference(set(major_class_set)))
    noise_data_set = []
    danger_data_set = []
    # Step10：对少类样本中的每一个样本点，计算其在样本总体中的k个最近邻，其中为非我样本的数量是m；
    tree = KDTree.construct_from_data(training_set)
    # Step11：如果m等于k，即该样本为孤立点，则该样本可能为噪音并移除，如果m小于k / 2，则该样本为安全，也无需特别处理；如果m大于或等于k / 2，则标记此样本为“危险”；
    for data in training_set:
        if data[0] in minority_class_set:
            nearest = tree.query(data, t=k)
            major_class_count = 0
            for near in nearest:
                if near[0] in major_class_set:
                    major_class_count = major_class_count + 1
            if major_class_count == k:
                noise_data_set.append(data)
            elif major_class_count >= k / 2:
                min_dis = get_min_distance(data, nearest, major_class_set)
                danger_data_set.append([data, min_dis])

    # Step12：以每一个“危险”样本为中心，以其与非我样本中最近点的距离减去step_size得到半径，生成检测器，并与中的检测器进行判重处理，如果不重复则加入
    for data_dis in danger_data_set:
        new_anti = [data_dis[0][0], data_dis[0][1], data_dis[1] - parameters['step_size']]
        if new_anti not in init_antibodies:
            init_antibodies.append(new_anti)

    # for i in noise_data_set:
    #     training_set.remove(i)

    return init_antibodies


def get_min_distance(data, nearest, major_class_set):
    for near in nearest:
        if near[0] in major_class_set:
            return distance(data[1], near[1])


def get_not_hit_data(init_antibodies, data_set):
    classes_antibody_dict = get_class_antibody_dict(init_antibodies)
    not_hit_set = []
    correct_count = 0
    for d in data_set:
        class_antibodies = classes_antibody_dict[d[0]]
        is_hit = False
        for a in class_antibodies:
            dis = distance(d[1], a[1])
            if dis <= a[2]:
                correct_count = correct_count + 1
                is_hit = True
                break
        if not is_hit:
            not_hit_set.append(d)
    return not_hit_set


def get_major_class_set(data_set, ratio, classes):
    major_class_set = []
    class_count_dict = {}
    for c in classes:
        class_count_dict[c] = 0

    for t in data_set:
        class_count_dict[t[0]] = class_count_dict[t[0]] + 1

    for c in class_count_dict:
        if class_count_dict[c] / float(len(data_set)) > ratio:
            major_class_set.append(c)
    return major_class_set

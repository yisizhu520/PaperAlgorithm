from random import choice

from algorithm.KDTree import KDTree
from algorithm.base import *


def generate_population(training_set, classes, size, parameters):
    antibodies = []

    for c in classes:

        class_data = [i for i in training_set if i[0] == c]
        if len(class_data) == 0:
            continue
        non_class_data = [i for i in training_set if i[0] != c]
        # Step1: 根据每一类样本所占总体的比例，分配每一类检测器数量
        num_of_antibodies = math.ceil(size * float(len(class_data)) / len(training_set))
        # Step2: 在第i类样本中分别选取个点作为检测器中心点，计算该中心点与非我样本点的距离，
        # 以最短距离减去step-size的结果为半径，生成检测器，并形成一个检测器集合Ab，并加入解空间
        tree = KDTree.construct_from_data(non_class_data)
        distinct_indexes = [i for i in range(len(class_data))]
        for i in range(num_of_antibodies):
            # replace random choice with CSA here
            if len(distinct_indexes) == 0:
                chosen_index = 0
            else:
                chosen_index = choice(distinct_indexes)
                distinct_indexes.remove(chosen_index)
            # print('chosen_index %s : %d', c, chosen_index)
            proposed_center = class_data[chosen_index]
            nearest = tree.query(proposed_center, t=1)[0]
            dist = distance(nearest[1], proposed_center[1])

            if dist <= parameters["step_size"]:
                radius = 0.0
            else:
                radius = dist - (dist % parameters["step_size"])

            proposed_antibody = [proposed_center[0], proposed_center[1], radius]
            antibodies.append(proposed_antibody)
    return antibodies




import csv as csv
import math as math
import time as time
from operator import itemgetter
from os import listdir
from random import choice
from random import shuffle
import random
from math import sqrt
from sklearn import svm
import copy
import numpy as np
import pandas


def separate(data):
    labels = []
    dataa = []
    for i in data:
        labels.append(i[0])
        dataa.append(i[1:])
    return [labels, dataa]


def square_distance(pointA, pointB):
    # squared euclidean distance
    distance = 0
    dimensions = len(pointA)  # assumes both points have the same dimensions
    for dimension in range(dimensions):
        distance += (pointA[dimension] - pointB[dimension]) ** 2
    distance = math.sqrt(distance)
    return distance


class KDTreeNode:
    def __init__(self, point, left, right):
        self.point = point
        self.left = left
        self.right = right

    def is_leaf(self):
        return self.left is None and self.right is None


class KDTreeneighbors:
    def __init__(self, query_point, t):
        self.query_point = query_point
        self.t = t  # neighbors wanted
        self.largest_distance = 0  # squared
        self.current_best = []

    def calculate_largest(self):
        if self.t >= len(self.current_best):
            self.largest_distance = self.current_best[-1][1]
        else:
            self.largest_distance = self.current_best[self.t - 1][1]

    def add(self, point):
        sd = square_distance(point[1], self.query_point[1])
        # run through current_best, try to find appropriate place
        for i, e in enumerate(self.current_best):
            if i == self.t:
                return  # enough neighbors, this one is farther, let's forget it
            if e[1] > sd:
                self.current_best.insert(i, [point, sd])
                self.calculate_largest()
                return
        # append it to the end otherwise
        self.current_best.append([point, sd])
        self.calculate_largest()

    def get_best(self):
        return [element[0] for element in self.current_best[:self.t]]


class KDTree:
    def __init__(self, data):
        def build_kdtree(point_list, depth):
            # code based on wikipedia article: http://en.wikipedia.org/wiki/Kd - tree
            if not point_list:
                return None
            # select axis based on depth so that axis cycles through all valid values
            axis = depth % len(point_list[0][1])  # assumes all points have the same dimension

            # sort point list and choose median as pivot point,
            # TODO: better selection method, linear-time selection, distribution
            point_list.sort(key=lambda x: x[1][axis])
            median = int(len(point_list) / 2)  # choose median

            # create node and recursively construct subtrees
            node = KDTreeNode(point=point_list[median], left=build_kdtree(point_list[0:median], depth + 1),
                              right=build_kdtree(point_list[median + 1:], depth + 1))
            return node

        self.root_node = build_kdtree(data, depth=0)

    @staticmethod
    def construct_from_data(data):
        tree = KDTree(data)
        return tree

    def query(self, query_point, t=1):
        statistics = {'nodes_visited': 0, 'far_search': 0, 'leafs_reached': 0}

        def nn_search(node, query_point, t, depth, best_neighbors):
            if node is None:
                return
            # statistics['nodes_visited'] += 1
            # if we have reached a leaf, let's add to current best neighbors,
            # (if it's better than the worst one or if there is not enough neighbors)

            if node.is_leaf():
                # statistics['leafs_reached'] += 1
                best_neighbors.add(node.point)
                return

            # this node is no leaf

            # select dimension for comparison (based on current depth)
            axis = depth % len(query_point[1])

            # figure out which subtree to search
            near_subtree = None  # near subtree
            far_subtree = None  # far subtree (perhaps we'll have to traverse it as well)

            # compare query_point and point of current node in selected dimension and figure out which subtree is
            # farther than the other
            if query_point[1][axis] < node.point[1][axis]:
                near_subtree = node.left
                far_subtree = node.right
            else:
                near_subtree = node.right
                far_subtree = node.left

            # recursively search through the tree until a leaf is found
            nn_search(near_subtree, query_point, t, depth + 1, best_neighbors)

            # while unwinding the recursion, check if the current node
            # is closer to query point than the current best,
            # also, until t points have been found, search radius is infinity
            best_neighbors.add(node.point)

            # check whether there could be any points on the other side of the
            # splitting plane that are closer to the query point than the current best

            if (node.point[1][axis] - query_point[1][axis]) ** 2 < best_neighbors.largest_distance:
                # statistics['far_search'] += 1
                nn_search(far_subtree, query_point, t, depth + 1, best_neighbors)
            return

        # if there's no tree, there's no neighbors
        if self.root_node is not None:
            neighbors = KDTreeneighbors(query_point, t)
            nn_search(self.root_node, query_point, t, depth=0, best_neighbors=neighbors)
            result = neighbors.get_best()
        else:
            result = []
        # print (statistics
        return result


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


def distance(x1, x2, parameters):
    if len(x1) != len(x2):
        return False
    else:
        distance = 0
        for i in range(len(x1)):
            distance = distance + ((float(x1[i]) - float(x2[i])) ** 2)
        # distance = math.sqrt(distance)
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

    # print (TP, TN, FP, FN)
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

    return "TP: " + str(TP) + " TN: " + str(TN) + " FP: " + str(FP) + " FN: " + str(FN)
    # return [precision, recall, fmeasure]


# testing the prediction performance
def test_accuracy(antibodies, test_data, parameters):
    error_count = 0
    correct_count = 0
    for x in test_data:
        yhat = predict(antibodies, x, parameters)
        # yhat = predictByVote(antibodies, x, parameters)
        if x[0] != yhat:
            error_count = error_count + 1
            # print ("predicted: ", yhat, " actual: ", x[0], "\t\t#")
        else:
            correct_count = correct_count + 1
            # print ("predicted: ", yhat, " actual: ", x[0])
    return float(correct_count) / float(len(test_data))


def test_accuracy_by_class(antibodies, test_data, parameters, training_set, pop_size, sequence):
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
    table = []
    table.append(['class', 'accuracy', 'test_count', 'train_count', 'anti_count'])

    for cls in classes_data_dict:
        data = classes_data_dict[cls]
        cls_correct_count = 0
        for x in data:
            yhat = predict(antibodies, x, parameters)
            # yhat = predictByVote(antibodies, x, parameters)
            if x[0] != yhat:
                error_count = error_count + 1
            else:
                correct_count = correct_count + 1
                cls_correct_count = cls_correct_count + 1
        # print('%s accuracy: %f' % (cls, float(cls_correct_count) / float(len(data))))
        train_count = 0
        anti_count = 0
        if cls in classes_train_data_dict:
            train_count = len(classes_train_data_dict[cls])
        if cls in classes_antibody_dict:
            anti_count = len(classes_antibody_dict[cls])
        table.append([cls, float(cls_correct_count) / float(len(data)), len(data), train_count, anti_count])

    table.append(['summary', float(correct_count) / float(len(test_data)), len(test_data), len(training_set), len(antibodies)])
    printTable(table)
    df = pandas.DataFrame(table)
    df.to_csv('class_accuray_'+str(pop_size)+'_'+str(sequence)+'.csv', index=False, header=False)
    return float(correct_count) / float(len(test_data))


def maxStr(Str):
    maxLen = 0
    for i in range(len(Str)):
        if len(str(Str[i])) > maxLen:
           maxLen = len(str(Str[i]))
        else:
           pass
    return maxLen


def printTable(tableData):
    for row in range(len(tableData)):
        for column in range(len(tableData[0])):
            print(str(tableData[row][column]), end=', ')
        print()


def test_accuracy_by_vote(antibodies, test_data, parameters):
    error_count = 0
    correct_count = 0
    for x in test_data:
        # yhat = predict(antibodies, x, parameters)
        yhat = predictByVote(antibodies, x, parameters)
        if x[0] != yhat:
            error_count = error_count + 1
            # print ("predicted: ", yhat, " actual: ", x[0], "\t\t#")
        else:
            correct_count = correct_count + 1
            # print ("predicted: ", yhat, " actual: ", x[0])
    return float(correct_count) / float(len(test_data))


# vote for the best classification
def predict(antibodies, x, parameters):
    distances = []
    for a in antibodies:
        d = distance(x[1], a[1], parameters)
        if d <= a[2]:
            return a[0]
        else:
            distances.append([a[0], d])
    distances.sort(key=itemgetter(1))
    return distances[0][0]


def predictByVote(antibodies, x, parameters):
    class_hit_count_dict = {}
    distances = []
    for a in antibodies:
        d = distance(x[1], a[1], parameters)
        if d <= a[2]:
            if a[0] not in class_hit_count_dict:
                class_hit_count_dict[a[0]] = 1
            else:
                class_hit_count_dict[a[0]] = class_hit_count_dict[a[0]] + 1
        else:
            distances.append([a[0], d])
    if not class_hit_count_dict:
        distances.sort(key=itemgetter(1))
        return distances[0][0]
    class_hit_count_dict = sorted(class_hit_count_dict.items(), key=lambda d: d[1], reverse=True)
    # print(class_hit_count_dict[0][0])
    return class_hit_count_dict[0][0]


def sort_by_value(d):
    items = d.items()
    backitems = [[v[1], v[0]] for v in items]
    backitems.sort()
    return [backitems[i][1] for i in range(0, len(backitems))]


# def optimized_predict(antibodies, x, self_class):
#     # select the antibodies that could contain the point
#     # for every dimension in the antibody center:
#     for i in range(len(antibodies[0][1])):
#         antibodies = [a for a in antibodies if x[1][i] > (a[1][i] - a[2]) and x[1][i] < (a[1][i] + a[2])]

#     # further filter the set of antibodies
#     for a in antibodies:
#         if distance(a[1], x[1][:], {}) < a[2]:
#             return 'nothing'
#     return self_class


def distance2(x1, x2):
    distance = sqrt(sum((x1 - x2) ** 2 for x1, x2 in zip(x1, x2)))
    return distance


def generate_population(training_set, classes, size, parameters):
    antibodies = []
    num_of_antibodies = int(float(size) / float(len(classes)))

    for c in classes:
        class_data = [i for i in training_set if i[0] == c]
        if len(class_data) == 0:
            continue
        non_class_data = [i for i in training_set if i[0] != c]
        tree = KDTree.construct_from_data(non_class_data)
        distinct_indexes = [i for i in range(len(class_data))]
        # print('old class %s - count %d', c, num_of_antibodies)
        for i in range(num_of_antibodies):
            # replace random choice with CSA here
            if len(distinct_indexes) == 0:
                chosen_index = 0
            else:
                chosen_index = choice(distinct_indexes)
                distinct_indexes.remove(chosen_index)
            proposed_center = class_data[chosen_index]
            nearest = tree.query(proposed_center, t=1)[0]
            dist = distance(nearest[1], proposed_center[1], parameters)

            if dist <= parameters["step_size"]:
                radius = 0.0
            else:
                radius = dist - (dist % parameters["step_size"])

            proposed_antibody = [proposed_center[0], proposed_center[1], radius]
            antibodies.append(proposed_antibody)
    return antibodies


def generate_population_by_number_ratio(training_set, classes, size, parameters):
    antibodies = []
    num_of_antibodies = int(float(size) / float(len(classes)))

    for c in classes:
        class_data = [i for i in training_set if i[0] == c]
        if len(class_data) == 0:
            continue
        non_class_data = [i for i in training_set if i[0] != c]
        # generate antibodies according to data number of class
        num_of_antibodies = math.ceil(size * float(len(class_data)) / len(training_set))
        # num_of_antibodies = len(class_data)
        # print('class %s - data count %d', c, num_of_antibodies)
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
            dist = distance(nearest[1], proposed_center[1], parameters)

            if dist <= parameters["step_size"]:
                radius = 0.0
            else:
                radius = dist - (dist % parameters["step_size"])

            proposed_antibody = [proposed_center[0], proposed_center[1], radius]
            antibodies.append(proposed_antibody)
        # print('class %s - antibodies count %d', c, len(antibodies))
    return antibodies

def error_count(antibody, training_set, parameters):
    error_count = 0
    class_data = [i for i in training_set if i[0] != antibody[0]]
    for t in class_data:
        if distance(t[1], antibody[1], parameters) <= antibody[2]:
            error_count = error_count + 1
    return error_count
 

def test_svm_accuracy(clf, test_set):
    error_count = 0
    correct_count = 0
    test_set_2d = [d[1] for d in test_set]
    for i in range(len(test_set_2d)):
        test_result = clf.predict([test_set_2d[i]])
        if test_set[i][0] != test_result[0]:
            error_count = error_count + 1
        else:
            correct_count = correct_count + 1
    # test_result = clf.predict(test_set_2d)
    # for i in range(len(test_set)):
    #     if test_set[i][0] != test_result[i]:
    #         error_count = error_count + 1
    #     else:
    #         correct_count = correct_count + 1
    return float(correct_count) / float(len(test_set))


# ------------ CSA start --------------------

def get_best_population_with_csa(training_set, classes, size, parameters):
    iteration_time = 1000
    save_count = 5
    accuracy = 0.000001
    init_chromosome_size = 10
    last_results = []
    results = get_random_population_results(training_set, classes, size, parameters, init_chromosome_size)
    print('old')
    print(results[0]['fitness'])
    print('new')
    for i in range(iteration_time):
        results = csa(results, training_set, classes, size, parameters)
        results.sort(key=lambda result: result['fitness'], reverse=True)
        last_results.append(results[0])
        if len(last_results) == save_count:
            flag = True
            for j in range(save_count - 1):
                if abs(last_results[save_count - 1]['fitness'] - last_results[j]['fitness']) > accuracy:
                    flag = False
                    break
            if flag:
                # print(last_results[save_count - 1]['antibodies'])
                print(results[0]['fitness'])
                return last_results[save_count - 1]['antibodies']
            else:
                last_results.remove(last_results[0])


def get_random_population_results(training_set, classes, size, parameters, init_chromosome_size):
    result_objs = []
    for i in range(init_chromosome_size):
        antibodies = generate_population(training_set, classes, size, parameters)
        fitness = cal_fitness(antibodies, training_set)
        obj = {'antibodies': antibodies, 'fitness': fitness}
        result_objs.append(obj)
    return result_objs


def csa(population_results, training_set, classes, size, parameters):
    choose_top_size = 5
    clone_size = 5
    random_reproduced_antibody_percent = 0.25
    result_objs = population_results
    result_objs.sort(key=lambda result: result['fitness'], reverse=True)
    top_result = result_objs[:choose_top_size]
    last_result = result_objs[choose_top_size:len(result_objs)]

    clone_pop_results = clone_population(top_result, clone_size)
    for pop_result in clone_pop_results:
        antis = pop_result['antibodies']
        antis = reproduce_antibodies(antis, random_reproduced_antibody_percent, training_set, classes, parameters)
        pop_result['fitness'] = cal_fitness(antis, training_set)

    for pop_result in last_result:
        clone_pop_results.append(pop_result)
    clone_pop_results.sort(key=lambda result: result['fitness'], reverse=True)
    last_result = clone_pop_results[:len(last_result)]

    combine_result = top_result + last_result
    # print(combine_result)
    return combine_result


def get_class_antibody_dict(antibodies):
    classes_antibody_dict = {}
    for a in antibodies:
        if a[0] not in classes_antibody_dict:
            classes_antibody_dict[a[0]] = []
            classes_antibody_dict[a[0]].append(a)
        else:
            classes_antibody_dict[a[0]].append(a)
    return classes_antibody_dict


def cal_fitness(antibodies, data_set):
    classes_antibody_dict = get_class_antibody_dict(antibodies)
    correct_count = 0
    not_hit_set = []
    for d in data_set:
        class_antibodies = classes_antibody_dict[d[0]]
        is_hit = False
        for a in class_antibodies:
            dis = distance(d[1], a[1], parameters)
            if dis <= a[2]:
                correct_count = correct_count + 1
                is_hit = True
                break
        if not is_hit:
            # print('not hit')
            # print(d)
            # print('nearest class dist--d%', cal_nearest_distance(d, data_set, True))
            # print('nearest non-class dist--d%', cal_nearest_distance(d, data_set, False))
            if cal_nearest_distance(d, data_set, True) > cal_nearest_distance(d, data_set, False):
                not_hit_set.append(d)
    for i in not_hit_set:
        data_set.remove(i)
    antibodies = generate_population(data_set, classes, 1000, parameters)
    print('adjust accuracy-- %d', test_accuracy(antibodies, test_set, parameters))
    # print('adjust fitness-- %d', float(correct_count) / float(len(data_set)))
    return float(correct_count) / float(len(data_set))


def cal_nearest_distance(data, data_set, is_same_class):
    class_data = []
    if is_same_class:
        class_data = [i for i in data_set if (i[0] == data[0] and i != data)]
    else:
        class_data = [i for i in data_set if i[0] != data[0]]
    if len(class_data) == 0:
        return -1
    tree = KDTree.construct_from_data(class_data)
    nearest = tree.query(data, t=1)[0]
    dist = distance(nearest[1], data[1], None)
    return dist


def clone_population(original_pops, clone_size):
    results = []
    for pop in original_pops:
        for i in range(clone_size):
            results.append(copy.deepcopy(pop))
    return results


def reproduce_antibodies(antibodies, percent, training_set, classes, parameters):
    classes_antibody_dict = get_class_antibody_dict(antibodies)
    # remove percent of antibodies randomly
    for cls in classes_antibody_dict:
        antis = classes_antibody_dict[cls]
        remove_size = int(len(antis) * percent)
        for i in range(remove_size):
            antis.remove(choice(antis))
    # reproduce percent of antibodies
    reproduced_pops = generate_population(training_set, classes, int(len(antibodies) * percent), parameters)
    new_antis = []
    for cls in classes_antibody_dict:
        for a in classes_antibody_dict[cls]:
            new_antis.append(a)
    for pop in reproduced_pops:
        new_antis.append(pop)
    return new_antis


# ------------ CSA end ----------------------

# ------------ GA start --------------------

def get_best_population_with_ga(training_set, classes, size, parameters):
    iteration_time = 1000
    save_count = 5
    accuracy = 0.000001
    init_chromosome_size = 10  # 染色体数量
    cp_ratio = 0.2  # 染色体复制的比例(每代中保留适应度较高的染色体直接成为下一代)

    last_results = []
    results = get_random_population_results(training_set, classes, size, parameters, init_chromosome_size)
    print('old')
    print(results[0]['fitness'])
    print('new')
    for i in range(iteration_time):
        results = ga(results, training_set, classes, size, parameters, init_chromosome_size, cp_ratio)
        results.sort(key=lambda result: result['fitness'], reverse=True)
        last_results.append(results[0])
        if len(last_results) == save_count:
            flag = True
            for j in range(save_count - 1):
                if abs(last_results[save_count - 1]['fitness'] - last_results[j]['fitness']) > accuracy:
                    flag = False
                    break
            if flag:
                # print(last_results[save_count - 1]['antibodies'])
                print(results[0]['fitness'])
                return last_results[save_count - 1]['antibodies']
            else:
                last_results.remove(last_results[0])


def ga(result_objs, training_set, classes, size, parameters, init_chromosome_size, cp_ratio):
    crossover_mutation_num = int(init_chromosome_size * (1 - cp_ratio))  # 参与交叉变异的染色体数量

    # 计算适应度总和
    fitness_sum = 0.0
    for i in range(len(result_objs)):
        fitness_sum += result_objs[i]['fitness']

    # 计算自然选择概率
    for i in range(len(result_objs)):
        result_objs[i]['prob'] = result_objs[i]['fitness'] / fitness_sum

    # XXOO 交叉生成{crossover_mutation_num}条染色体
    new_result_objs = []

    # 按class排好antibody，便于后面的交叉
    sort_antibodies_by_class(result_objs, classes)

    for i in range(crossover_mutation_num):
        # 采用轮盘赌选择父母染色体
        father = result_objs[roulette_wheel_select(result_objs)]
        mother = result_objs[roulette_wheel_select(result_objs)]
        # 交叉
        cross_index = random.randint(0, len(classes))

        new_antibody_result = {}
        new_antibody_result['list'] = father['list'][:cross_index] + mother['list'][cross_index:]
        new_result_objs.append(new_antibody_result)

    # 变异
    # 随机找一个染色体
    chromosome_index = random.randint(0, crossover_mutation_num - 1)
    # 随机找一个类
    class_index = random.randint(0, len(classes) - 1)
    # 替换成重新随机生成的抗体
    random_antibodies = generate_population(training_set, classes, size, parameters)  # 可以优化，不用生成全部类的antibodies
    class_antibodies = [i for i in random_antibodies if i[0] == classes[class_index]]
    new_result_objs[chromosome_index][class_index] = class_antibodies

    # 新生的抗体，重新计算适应度
    for antibody_result in new_result_objs:
        restore_antibodies_by_list(antibody_result)
        antibody_result['fitness'] = cal_fitness(antibody_result['antibodies'], training_set)

    # 复制适应度最高的 init_chromosome_size * cp_ratio条染色
    result_objs.sort(key=lambda result: result['fitness'], reverse=True)
    copy_size = int(init_chromosome_size * cp_ratio)
    for i in range(copy_size):
        new_result_objs.append(result_objs[i])

    return new_result_objs


def sort_antibodies_by_class(result_objs, classes):
    for result_obj in result_objs:
        class_antibody_dict = get_class_antibody_dict(result_obj['antibodies'])
        antibody_list = []
        for j in classes:
            if j in class_antibody_dict:
                antibody_list.append(class_antibody_dict[j])
            else:
                antibody_list.append('empty_stub')
        result_obj['list'] = antibody_list


def restore_antibodies_by_list(result_obj):
    antibodies = []
    for antibody_list in result_obj['list']:
        if antibody_list == 'empty_stub':
            continue
        for antibody in antibody_list:
            antibodies.append(antibody)
    result_obj['antibodies'] = antibodies


def roulette_wheel_select(result_objs):
    rand = random.random()
    prob_sum = 0.0
    for i in range(len(result_objs)):
        prob_sum += result_objs[i]['prob']
        if prob_sum >= rand:
            return i
    return 0


# ------------ GA end ----------------------


# new structure of antibody: [ class, [x1, x2, x3,... ], radius]
original_data = getdata("data/kddcup_30w.csv")
classes = get_class_labels(original_data)
# proportions = proportion_per_class(original_data)

parameters = {}
parameters["step_size"] = 0.01

# varying the training set size
# learning time, seconds of time over training set size,
# print("set_size \t time_with_kd \t time_without_kd")
# for set_size in range(200, 1050, 50):
#     average_time = 0.0
#     # building a balanced data set
#     data = []
#     for c in classes:
#         class_data = [d for d in original_data if d[0] == c]
#         shuffle(class_data)
#         data = data + class_data[:int(float(set_size) / float(len(classes)))]
#     data = normalize(data)
#     data = stratify(data, 10)
#
#     time_with_kd = 0.0
#     time_without_kd = 0.0
#     SVM_time = 0.0
#     average_accuracy = 0.0
#     for st in range(10):
#         test_set = data[st % len(data)]
#         validation_set = data[(st + 1) % len(data)]
#         training_set = []
#         for tsp in range(len(data) - 2):
#             training_set = training_set + data[(st + 2 + tsp) % len(data)]
#
#         start = time.time()
#         antibodies = generate_population(training_set, classes, 1000, parameters)
#         end = time.time()
#
#         time_with_kd = time_with_kd + (end - start)
#
#         # start = time.time()
#         # antibodies = original_generate_population(training_set, classes, 1000, parameters)
#         # end = time.time()
#         accuracy = test_accuracy(antibodies, test_set, parameters)
#         average_accuracy = average_accuracy + accuracy
#     print("1000 \t ", " \t ", average_accuracy / 10.0)
#
#     #     time_without_kd = time_without_kd + (end - start)
#     #
#     #     # preparing data for SVM
#     #     datta = separate(training_set)
#     #     training_set_labels = datta[0]
#     #     training_set_data = datta[1]
#     #     training_set_data_2d = [d[0] for d in training_set_data]
#     #
#     #     start = time.time()
#     #     lin_clf = svm.LinearSVC()
#     #     lin_clf.fit(training_set_data_2d, training_set_labels)
#     #     end = time.time()
#     #     SVM_time = SVM_time + (end - start)
#     # print(set_size, " \t ", time_with_kd / 10.0, " \t ", time_without_kd / 10.0, " \t ", SVM_time / 10.0)
# print("")


# varying the antibody population time
print(
    "pop_size \t time_with_kd_CSA, accuracy \t time_with_kd_GA, accuracy \t time_with_kd, accuracy \t time_without_kd, accuracy \t svm_time, accuracy")
for pop_size in range(1000, 2000, 1000):
    # building a balanced data set
    data = []
    for c in classes:
        class_data = [d for d in original_data if d[0] == c]
        shuffle(class_data)
        data = data + class_data

    data = normalize(data)
    data = stratify(data, 10)
    time_with_kd_CSA = 0.0
    time_with_kd_GA = 0.0
    time_with_kd = 0.0
    time_without_kd = 0.0
    SVM_time = 0.0
    average_accuracy_with_kd = 0.0
    average_accuracy_without_kd = 0.0
    average_accuracy_svm = 0.0
    average_accuracy_with_kd_CSA = 0.0
    average_accuracy_with_kd_GA = 0.0
    iteration_count = 3
    for st in range(iteration_count):
        test_set = data[st % len(data)]
        validation_set = data[(st + 1) % len(data)]
        training_set = []
        for tsp in range(len(data) - 2):
            training_set = training_set + data[(st + 2 + tsp) % len(data)]
        # test with CSA
        # csa_antibodies = get_best_population_with_csa(training_set, classes, pop_size, parameters)
        # start = time.time()
        # accuracy = test_accuracy(csa_antibodies, test_set, parameters)
        # end = time.time()
        # time_with_kd_CSA = time_with_kd_CSA + (float(end) - float(start))
        # accuracy = test_accuracy(csa_antibodies, test_set, parameters)
        # average_accuracy_with_kd_CSA = average_accuracy_with_kd_CSA + accuracy

        # test with GA
        ga_antibodies = generate_population_by_number_ratio(training_set, classes, pop_size, parameters)
        start = time.time()
        accuracy = test_accuracy_by_class(ga_antibodies, test_set, parameters, training_set, pop_size, st)
        end = time.time()
        time_with_kd_GA = time_with_kd_GA + (float(end) - float(start))
        average_accuracy_with_kd_GA = average_accuracy_with_kd_GA + accuracy
        # print('old accuray-- %d', accuracy)
        # print('ratio fitness-- %d', cal_fitness(ga_antibodies, training_set))

        # test without CSA
        # antibodies = generate_population(training_set, classes, pop_size, parameters)
        # # print('fitness-- %d', cal_fitness(antibodies, training_set))
        # start = time.time()
        # accuracy = test_accuracy(antibodies, test_set, parameters)
        # end = time.time()
        # time_with_kd = time_with_kd + (float(end) - float(start))
        # average_accuracy_with_kd = average_accuracy_with_kd + accuracy
        # print('old fitness-- %d', cal_fitness(antibodies, training_set))

        # antibodies = original_generate_population(training_set, classes, pop_size, parameters)
        # start = time.time()
        # accuracy = test_accuracy(antibodies, test_set, parameters)
        # end = time.time()

        # time_without_kd = time_without_kd + (float(end) - float(start))

        # accuracy = test_accuracy(antibodies, test_set, parameters)
        # average_accuracy_without_kd = average_accuracy_without_kd + accuracy

        # preparing data for SVM
        # datta = separate(training_set)
        # training_set_labels = datta[0]
        # training_set_data = datta[1]
        # training_set_data_2d = [d[0] for d in training_set_data]
        # lin_clf = svm.LinearSVC()
        # lin_clf.fit(training_set_data_2d, training_set_labels)
        # start = time.time()
        # accuracy = test_svm_accuracy(lin_clf, test_set)
        # end = time.time()
        #
        # SVM_time = SVM_time + (end - start)
        # accuracy = test_svm_accuracy(lin_clf, test_set)
        # average_accuracy_svm = average_accuracy_svm + accuracy

    print(pop_size,
          " \t ", time_with_kd_CSA / iteration_count, ", ", average_accuracy_with_kd_CSA / iteration_count,
          " \t ", time_with_kd_GA / iteration_count, ", ", average_accuracy_with_kd_GA / iteration_count,
          " \t ", time_with_kd / iteration_count, ", ", average_accuracy_with_kd / iteration_count,
          " \t ", SVM_time / iteration_count, ", ", average_accuracy_svm / iteration_count
          )
print("")

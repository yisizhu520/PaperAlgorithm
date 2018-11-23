import csv as csv
import math as math
import time as time
from operator import itemgetter
from os import listdir
from random import choice
from random import shuffle

from sklearn import svm


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


def generate_population(training_set, classes, size, parameters):
    antibodies = []
    num_of_antibodies = int(float(size) / float(len(classes)))

    for c in classes:
        class_data = [i for i in training_set if i[0] == c]
        non_class_data = [i for i in training_set if i[0] != c]

        tree = KDTree.construct_from_data(non_class_data)

        for i in range(num_of_antibodies):
            proposed_center = choice(class_data)
            nearest = tree.query(proposed_center, t=1)[0]
            dist = distance(nearest[1], proposed_center[1], parameters)

            if dist <= parameters["step_size"]:
                radius = 0.0
            else:
                radius = dist - (dist % parameters["step_size"])

            proposed_antibody = [proposed_center[0], proposed_center[1], radius]
            antibodies.append(proposed_antibody)
    return antibodies


def error_count(antibody, training_set, parameters):
    error_count = 0
    class_data = [i for i in training_set if i[0] != antibody[0]]
    for t in class_data:
        if distance(t[1], antibody[1], parameters) <= antibody[2]:
            error_count = error_count + 1
    return error_count


def original_generate_population(training_set, classes, size, parameters):
    antibodies = []
    # select random antibodies from the self class, and add with a radius of 0
    for c in classes:
        class_data = [i for i in training_set if i[0] == c]

        num_of_antibodies = int(float(size) / float(len(classes)))
        for i in range(num_of_antibodies):
            proposed_center = choice(class_data)
            proposed_antibody = [proposed_center[0], proposed_center[1], 0.0]
            antibodies.append(proposed_antibody)

            # expand the antibodies by a step size until it misclassify a non-self point
    for a in antibodies:
        changed = True
        while changed:
            if error_count(a, training_set, parameters) > 0:
                a[2] = a[2] - parameters["step_size"]
                changed = False
            else:
                a[2] = a[2] + parameters["step_size"]
                changed = True
    return antibodies


# new structure of antibody: [ class, [x1, x2, x3,... ], radius]
files = [f for f in listdir("C:/Users/Brian/Documents/IPython Notebooks/network_data/")]
original_data = []
for f in files:
    original_data = original_data + getdata(
        "C:/Users/Brian/Documents/IPython Notebooks/network_data/" + f)
classes = get_class_labels(original_data)
proportions = proportion_per_class(original_data)

parameters = {}
parameters["step_size"] = 0.01

# varying the training set size
# learning time, seconds of time over training set size,
print("set_size \t time_with_kd \t time_without_kd")
for set_size in range(200, 1050, 50):
    average_time = 0.0
    # building a balanced data set
    data = []
    for c in classes:
        class_data = [d for d in original_data if d[0] == c]
        shuffle(class_data)
        data = data + class_data[:int(float(set_size) / float(len(classes)))]
    data = normalize(data)
    data = stratify(data, 10)

    time_with_kd = 0.0
    time_without_kd = 0.0
    SVM_time = 0.0

    for st in range(10):
        test_set = data[st % len(data)]
        validation_set = data[(st + 1) % len(data)]
        training_set = []
        for tsp in range(len(data) - 2):
            training_set = training_set + data[(st + 2 + tsp) % len(data)]

        start = time.time()
        antibodies = generate_population(training_set, classes, 1000, parameters)
        end = time.time()

        time_with_kd = time_with_kd + (end - start)

        start = time.time()
        antibodies = original_generate_population(training_set, classes, 1000, parameters)
        end = time.time()

        time_without_kd = time_without_kd + (end - start)

        # preparing data for SVM
        datta = separate(training_set)
        training_set_labels = datta[0]
        training_set_data = datta[1]
        start = time.time()
        lin_clf = svm.LinearSVC()
        lin_clf.fit(training_set_data, training_set_labels)
        end = time.time()
        SVM_time = SVM_time + (end - start)
    print(set_size, " \t ", time_with_kd / 10.0, " \t ", time_without_kd / 10.0, " \t ", SVM_time / 10.0)
print("")


# varying the antibody population time
print("pop_size \t time_with_kd \t time_without_kd")
for pop_size in range(200, 1050, 50):
    # building a balanced data set
    data = []
    for c in classes:
        class_data = [d for d in original_data if d[0] == c]
        shuffle(class_data)
        data = data + class_data[:int(float(1000) / float(len(classes)))]

    data = normalize(data)
    data = stratify(data, 10)

    time_with_kd = 0.0
    time_without_kd = 0.0
    SVM_time = 0.0

    for st in range(10):
        test_set = data[st % len(data)]
        validation_set = data[(st + 1) % len(data)]
        training_set = []
        for tsp in range(len(data) - 2):
            training_set = training_set + data[(st + 2 + tsp) % len(data)]

        start = time.clock()
        antibodies = generate_population(training_set, classes, pop_size, parameters)
        end = time.clock()

        time_with_kd = time_with_kd + (float(end) - float(start))

        start = time.clock()
        antibodies = original_generate_population(training_set, classes,pop_size, parameters)
        end = time.clock()

        time_without_kd = time_without_kd + (float(end) - float(start))

        # preparing data for SVM
        datta = separate(training_set)
        training_set_labels = datta[0]
        training_set_data = datta[1]

        start = time.time()
        lin_clf = svm.LinearSVC()
        lin_clf.fit(training_set_data, training_set_labels)
        end = time.time()

        SVM_time = SVM_time + (end - start)
    print(pop_size, " \t ", time_with_kd / 10.0, " \t ", time_without_kd / 10.0, " \t ", SVM_time / 10.0)
print("")

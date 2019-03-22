from data_processor import *
from algorithm.base import *
from random import shuffle
import algorithm.BoundaryCalculation as BoundaryCalculation
import algorithm.NegativeSelection as NegativeSelection
import pandas

# new structure of antibody: [ class, [x1, x2, x3,... ], radius]
original_data = getdata("data/nsl-kddcup_1k.csv")
classes = get_class_labels(original_data)
parameters = {}
parameters["step_size"] = 0.1
pop_min_size = 100
pop_size_increment = 100
pop_max_size = 300
iteration_count = 3


# varying the antibody population time

def prepare_data(original_data, classes):
    data = []
    for c in classes:
        class_data = [d for d in original_data if d[0] == c]
        shuffle(class_data)
        data = data + class_data
    data = normalize(data)
    data = stratify(data, 10)
    return data


def test_accuracy_by_class(antibodies, test_data, training_set, pop_size, sequence):
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
        table.append([cls, float(cls_correct_count) / float(len(data)), len(data), train_count, anti_count])

    table.append(
        ['summary', float(correct_count) / float(len(test_data)), len(test_data), len(training_set), len(antibodies)])
    return float(correct_count) / float(len(test_data))


for pop_size in range(pop_min_size, pop_max_size + pop_size_increment, pop_size_increment):
    # building a balanced data set

    data = prepare_data(original_data, classes)
    table = []
    table.append(['class', 'accuracy', 'test_count', 'train_count', 'anti_count'])
    for st in range(iteration_count):
        test_set = data[st % len(data)]
        validation_set = data[(st + 1) % len(data)]
        training_set = []
        for tsp in range(len(data) - 2):
            training_set = training_set + data[(st + 2 + tsp) % len(data)]

        bound_antibodies = BoundaryCalculation.generate_population(training_set, classes, pop_size, parameters)

        nega_antibodies = NegativeSelection.generate_population(training_set, classes, pop_size, parameters)




        start = time.time()
        accuracy = test_accuracy(csa_antibodies, test_set, parameters)
        end = time.time()
        time_with_kd_CSA = time_with_kd_CSA + (float(end) - float(start))
        average_accuracy_with_kd_CSA = average_accuracy_with_kd_CSA + accuracy

        # antibodies = generate_population_by_bound_calculation(best_antibodies_with_csa, training_set, parameters)
        start = time.time()

        accuracy = test_accuracy(antibodies, test_set, parameters)
        end = time.time()
        time_with_kd_GA = time_with_kd_GA + (float(end) - float(start))
        average_accuracy_with_kd_GA = average_accuracy_with_kd_GA + accuracy

    print(pop_size,
          " \t ", time_with_kd_CSA / iteration_count, ", ", average_accuracy_with_kd_CSA / iteration_count,
          " \t ", time_with_kd_GA / iteration_count, ", ", average_accuracy_with_kd_GA / iteration_count,
          " \t ", time_with_kd / iteration_count, ", ", average_accuracy_with_kd / iteration_count,
          " \t ", SVM_time / iteration_count, ", ", average_accuracy_svm / iteration_count
          )
print("")

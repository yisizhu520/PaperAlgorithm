import time as time
from random import shuffle
from algorithm.NegativeSelection import *
from algorithm import CSA

# 迭代次数G，初始种群数量M，选择率S，克隆比例C，检测器总数量为N，类别数量C
G = 3
M = 5
S = 0.3
C = 5
# TODO 检测器数量为N，CSA迭代次数也为N？
N = 10
















def start():
    # new structure of antibody: [ class, [x1, x2, x3,... ], radius]
    original_data = getdata("data/nsl-kddcup_1k.csv")
    classes = get_class_labels(original_data)
    parameters = {}
    parameters["step_size"] = 0.1
    # varying the antibody population time
    print(
        "pop_size \t time_with_kd_CSA, accuracy \t time_with_kd_GA, accuracy \t time_with_kd, accuracy \t time_without_kd, accuracy \t svm_time, accuracy")

    for pop_size in range(100, 750, 50):
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
        iteration_count = 1
        for st in range(iteration_count):
            test_set = data[st % len(data)]
            validation_set = data[(st + 1) % len(data)]
            training_set = []
            for tsp in range(len(data) - 2):
                training_set = training_set + data[(st + 2 + tsp) % len(data)]
            # test with CSA
            csa_antibodies = generate_population(training_set, classes, pop_size, parameters)
            start = time.time()
            accuracy = test_accuracy(csa_antibodies, test_set, parameters)
            end = time.time()
            time_with_kd_CSA = time_with_kd_CSA + (float(end) - float(start))
            average_accuracy_with_kd_CSA = average_accuracy_with_kd_CSA + accuracy

            init_antibodies_set = []
            for i in range(M):
                # Step5：重复执行2 - 4的过程M次，得到克隆选择过程的初始解空间
                init_antibodies_set.append(generate_population(training_set, classes, pop_size, parameters))
            antibodies = CSA.get_best_population_with_csa(init_antibodies_set, training_set, classes,
                                                          pop_size, parameters, S, C)
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




start()

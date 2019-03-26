from algorithm.base import *
import time
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn import naive_bayes
from sklearn import tree
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn import neighbors


def get_evaluation_indicator(algorithm, training_set, test_set):
    datta = separate(training_set)
    training_set_labels = datta[0]
    training_set_data = datta[1]
    training_set_data_2d = [d[0] for d in training_set_data]
    model = None
    if algorithm == 'naive_bayes':
        model = naive_bayes.GaussianNB()  # 高斯贝叶斯
    elif algorithm == 'decision_tree':
        model = tree.DecisionTreeClassifier(max_depth=3)
        # model = tree.DecisionTreeClassifier(criterion=’gini’, max_depth = None,
        # min_samples_split = 2, min_samples_leaf = 1, min_weight_fraction_leaf = 0.0,
        # max_features = None, random_state = None, max_leaf_nodes = None,
        # min_impurity_decrease = 0.0, min_impurity_split = None,
        # class_weight = None, presort = False)
    elif algorithm == 'neural_network':
        # 定义多层感知机分类算法
        model = MLPClassifier(solver='lbfgs', activation='identity', max_iter=2, alpha=0.01,
                              hidden_layer_sizes=(10, 10),
                              random_state=1, verbose=True)
    elif algorithm == 'svm':
        model = svm.LinearSVC()
    elif algorithm == 'knn':
        model = neighbors.KNeighborsClassifier(n_neighbors=5, n_jobs=1)

    model.fit(training_set_data_2d, training_set_labels)
    start = time.time()
    accuracy = test_accuracy(model, test_set)
    end = time.time()
    classes = get_class_labels(training_set)
    fmeasure = 0.0
    gmean = 0.0
    TPR = 0.0
    FPR = 0.0
    for cls in classes:
        dict = get_fmeasure(model, test_set, cls)
        fmeasure += dict['fmeasure'] / len(classes)
        gmean += dict['gmean'] / len(classes)
        TPR += dict['TPR'] / len(classes)
        FPR += dict['FPR'] / len(classes)
    return {
        'fmeasure': fmeasure,
        'gmean': gmean,
        'TPR': TPR,
        'FPR': FPR,
        'accuracy': accuracy,
        'predict_time': end - start
    }


def naive_bayes_test(training_set, test_set):
    datta = separate(training_set)
    training_set_labels = datta[0]
    training_set_data = datta[1]
    training_set_data_2d = [d[0] for d in training_set_data]
    model = naive_bayes.GaussianNB()  # 高斯贝叶斯
    model.fit(training_set_data_2d, training_set_labels)
    start = time.time()
    accuracy = test_accuracy(model, test_set)
    print(accuracy)
    end = time.time()


def svm_test(training_set, test_set):
    datta = separate(training_set)
    training_set_labels = datta[0]
    training_set_data = datta[1]
    training_set_data_2d = [d[0] for d in training_set_data]
    model = svm.LinearSVC()
    model.fit(training_set_data_2d, training_set_labels)
    start = time.time()
    accuracy = test_accuracy(model, test_set)
    end = time.time()


def decision_tree_test(training_set, test_set):
    datta = separate(training_set)
    training_set_labels = datta[0]
    training_set_data = datta[1]
    training_set_data_2d = [d[0] for d in training_set_data]

    model = tree.DecisionTreeClassifier()
    model.fit(training_set_data_2d, training_set_labels)
    start = time.time()
    accuracy = test_accuracy(model, test_set)
    print(accuracy)
    end = time.time()


def neural_network_test(training_set, test_set):
    datta = separate(training_set)
    training_set_labels = datta[0]
    training_set_data = datta[1]
    training_set_data_2d = [d[0] for d in training_set_data]
    from sklearn.neural_network import MLPClassifier
    # 定义多层感知机分类算法
    model = MLPClassifier(solver='lbfgs', activation='identity', max_iter=2, alpha=0.01, hidden_layer_sizes=(50, 50),
                          random_state=1, verbose=True)
    model.fit(training_set_data_2d, training_set_labels)
    start = time.time()
    accuracy = test_accuracy(model, test_set)
    print(accuracy)
    end = time.time()


def get_fmeasure(model, test_data, class_label):
    TP, TN, FP, FN = 0, 0, 0, 0
    SE, SP, TPR, FPR = 0, 0, 0, 0
    for x in test_data:
        if x[0] == class_label:
            yhat = model.predict([x[1]])
            if x[0] == yhat:
                TP += 1
            else:
                FN += 1
        else:
            yhat = model.predict([x[1]])
            if yhat == class_label:
                FP += 1
            else:
                TN += 1
    if float(TP + FP) != 0:
        precision = float(TP) / float(TP + FP)
    else:
        precision = 0.0
    if float(TP + FN) != 0:
        recall = float(TP) / float(TP + FN)
    else:
        recall = 0.0
    if float(FP + TN) != 0:
        FPR = float(FP) / float(FP + TN)
        SP = float(TN) / float(FP + TN)
    else:
        FPR = 0.0
        SP = 0.0
    if (precision + recall) != 0:
        fmeasure = 2 * ((precision * recall) / (precision + recall))
    else:
        fmeasure = 0.0

    SE = TPR = recall
    gmean = math.sqrt(SE * SP)
    return {
        'fmeasure': fmeasure,
        'gmean': gmean,
        'TPR': TPR,
        'FPR': FPR,
    }


def test_accuracy(model, test_set):
    error_count = 0
    correct_count = 0
    test_set_2d = [d[1] for d in test_set]
    for i in range(len(test_set_2d)):
        test_result = model.predict([test_set_2d[i]])
        if test_set[i][0] != test_result[0]:
            error_count = error_count + 1
        else:
            correct_count = correct_count + 1
    return float(correct_count) / float(len(test_set))


def svm_demo(training_set, test_set, classes):
    datta = separate(training_set)
    training_set_labels = datta[0]
    training_set_data = datta[1]
    training_set_data_2d = [d[0] for d in training_set_data]

    # Binarize the output
    y = label_binarize(training_set_labels, classes=classes)
    n_classes = y.shape[1]

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(np.array(training_set_data_2d), y, test_size=.1,
                                                        random_state=0)

    # Learn to predict each class against the other
    model = svm.LinearSVC()
    from sklearn.neural_network import MLPClassifier
    # 定义多层感知机分类算法
    # clf = MLPClassifier(solver='sgd', activation='identity', max_iter=5, alpha=1e-5, hidden_layer_sizes=(100, 50),
    #                     random_state=1, verbose=True)
    from sklearn import naive_bayes
    model = naive_bayes.GaussianNB()  # 高斯贝叶斯
    classifier = OneVsRestClassifier(model)
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

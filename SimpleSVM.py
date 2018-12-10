from __future__ import division
import numpy as np
import csv as csv
import time as time
"""
This dataset is part of MNIST dataset,but there is only 3 classes,
classes = {0:'0',1:'1',2:'2'},and images are compressed to 14*14 
pixels and stored in a matrix with the corresponding label, at the 
end the shape of the data matrix is 
num_of_images x 14*14(pixels)+1(lable)
"""


def load_data(split_ratio):
    tmp = np.load("data216x197.npy")
    data = tmp[:, :-1]
    label = tmp[:, -1]
    mean_data = np.mean(data, axis=0)
    train_data = data[int(split_ratio * data.shape[0]):] - mean_data
    train_label = label[int(split_ratio * data.shape[0]):]
    test_data = data[:int(split_ratio * data.shape[0])] - mean_data
    test_label = label[:int(split_ratio * data.shape[0])]
    return train_data, train_label, test_data, test_label

def getdata(file_name):
    with open(file_name, newline='') as f:
        rowdata = []
        reader = csv.reader(f)
        for row in reader:
            for i in range(1, len(row)):
                row[i] = float(row[i])
            rowdata.append(row)
    return rowdata

def load_csv(split_ratio):
    tmp = getdata("data/kddcup_1k_fac.csv")
    tmp = np.array(tmp)
    # tmp = tmp.astype(float)
    # data = tmp[0:, 1:]
    # label = tmp[:, 0]
    data = tmp[:, :-1]
    label = tmp[:, -1]
    data = data.astype(float)
    mean_data = np.mean(data, axis=0)
    train_data = data[int(split_ratio * data.shape[0]):] - mean_data
    train_label = label[int(split_ratio * data.shape[0]):]
    test_data = data[:int(split_ratio * data.shape[0])] - mean_data
    test_label = label[:int(split_ratio * data.shape[0])]
    return train_data, train_label, test_data, test_label


"""compute the hingle loss without using vector operation,
While dealing with a huge dataset,this will have low efficiency
X's shape [n,14*14+1],Y's shape [n,],W's shape [num_class,14*14+1]"""


def lossAndGradNaive(X, Y, W, reg):
    dW = np.zeros(W.shape)
    loss = 0.0
    num_class = W.shape[0]
    num_X = X.shape[0]
    for i in range(num_X):
        scores = np.dot(W, X[i])
        cur_scores = scores[int(float(Y[i]))]
        for j in range(num_class):
            if j == Y[i]:
                continue
            margin = scores[j] - cur_scores + 1
            if margin > 0:
                loss += margin
                dW[j, :] += X[i]
                dW[int(float(Y[i])), :] -= X[i]
    loss /= num_X
    dW /= num_X
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    return loss, dW


def lossAndGradVector(X, Y, W, reg):
    dW = np.zeros(W.shape)
    N = X.shape[0]
    Y_ = X.dot(W.T)
    margin = Y_ - Y_[range(N), Y.astype(int)].reshape([-1, 1]) + 1.0
    margin[range(N), Y.astype(int)] = 0.0
    margin = (margin > 0) * margin
    loss = 0.0
    loss += np.sum(margin) / N
    loss += reg * np.sum(W * W)
    """For one data,the X[Y[i]] has to be substracted several times"""
    countsX = (margin > 0).astype(int)
    countsX[range(N), Y.astype(int)] = -np.sum(countsX, axis=1)
    dW += np.dot(countsX.T, X) / N + 2 * reg * W
    return loss, dW


def predict(X, W):
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    Y_ = np.dot(X, W.T)
    Y_pre = np.argmax(Y_, axis=1)
    return Y_pre


def accuracy(X, Y, W):
    start = time.time()
    Y_pre = predict(X, W)
    Y = Y.astype(float).astype(int)
    Y_float = Y_pre.astype(float).astype(int)
    acc = (Y_float == Y).mean()
    end = time.time()
    print(end - start)
    return acc


def model(X, Y, alpha, steps, reg):
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    W = np.random.randn(34, X.shape[1]) * 0.0001
    for step in range(steps):
        loss, grad = lossAndGradNaive(X, Y, W, reg)
        W -= alpha * grad
        print("The {} step, loss={}, accuracy={}".format(step, loss, accuracy(X[:, :-1], Y, W)))
    return W


# train_data, train_label, test_data, test_label = load_data(0.2)
train_data, train_label, test_data, test_label = load_csv(0.4)
W = model(train_data, train_label, 0.0001, 25, 0.5)
print("Test accuracy of the model is {}".format(accuracy(test_data, test_label, W)))



import numpy as np
import pandas as pd
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt
import math


# todo: complete the following functions, you may add auxiliary functions or define class to help you

def softsvm(l, trainX: np.array, trainy: np.array):
    """

    :param l: the parameter lambda of the soft SVM algorithm
    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: linear predictor w, a numpy array of size (d, 1)
    """
    m = len(trainy)
    d = len(trainX[0])
    l_identity = 2*l * np.identity(d)

    HRight = np.zeros((d + m, m))
    zeroLeftCorner = np.zeros((m, d))
    HLeft = np.concatenate((l_identity, zeroLeftCorner), axis=0)
    H = matrix(np.concatenate((HLeft, HRight), axis=1))

    u_left = np.zeros(d)
    u_right = (1 / m) * np.ones(m)
    u = matrix(np.concatenate((u_left, u_right)))


    v_up = np.zeros((m, 1))
    v_down = np.ones((m, 1))

    v = matrix(np.concatenate((v_up, v_down)))
    A_topLeft = np.zeros((m, d))
    A_topRight = np.identity(m)
    A_downRight = np.identity(m)
    A_right = np.concatenate((A_topRight, A_downRight), axis=0)
    A_downLeft = np.array([np.array([trainy[i] * trainX[i][j] for j in range(d)]) for i in range(m)])
    A_left = np.concatenate((A_topLeft, A_downLeft), axis=0)
    A = np.concatenate((A_left, A_right), axis=1)
    A = matrix(A)
    sol = solvers.qp(H, u, -A, -v)
    return np.array(sol["x"][:d])


# w(1)*y(1)*x(1) ---
# w(1)- y(1)x1(1),y(1)x1(2),
# w(2)- - y(2)x2(1),y(1)x2(2),

# w
# np.identity(n)

# s = (2,2)
# np.zeros(s)
# array([[ 0.,  0.],
#        [ 0.,  0.]])

def simple_test():
    # load question 2 data
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = 100
    d = trainX.shape[1]

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvm algorithm
    min_ = 1
    for l in {10, 10 ** 3, 10 ** 5, 10 ** 8}:
        w = softsvm(l, _trainX, _trainy)

        # tests to make sure the output is of the intended class and shape
        assert isinstance(w, np.ndarray), "The output of the function softsvm should be a numpy array"
        assert w.shape[0] == d and w.shape[1] == 1, f"The shape of the output should be ({d}, 1)"

        # get a random example from the test set, and classify it

        predicty = np.array([1.0 if np.sign(testX[i] @ w)[0] == 1.0 else -1.0 for i in range(testX.shape[0])])
        tempMin = np.mean(predicty != testy)
        print(tempMin)
        min_ = min(tempMin, min_)
    print(f"The absolute minimum {min_}")


def task2(m: int, lamdaArray: [int], numberOfExperiments: int) -> ():  # (min[],max[],average[])
    min_array_per_lambda = []
    max_array_per_lambda = []
    average_array_per_lambda = []
    average_train_error = []
    for l in lamdaArray:
        absolute_min = 1
        absolute_max = -1
        sum_of_testError = 0
        sum_of_trainError = 0
        for _ in range(numberOfExperiments):
            testX, testy, w, trainX, trainY = doSoftSVM(l, m)
            predictyTest = np.array([1.0 if np.sign(testX[i] @ w)[0] == 1.0 else -1.0 for i in range(testX.shape[0])])
            testError = np.mean(predictyTest != testy)
            absolute_min = min(absolute_min, testError)
            absolute_max = max(absolute_max, testError)
            sum_of_testError += testError
            predictTrain = np.array([1.0 if np.sign(trainX[i] @ w)[0] == 1.0 else -1.0 for i in range(trainX.shape[0])])
            trainError = np.mean(predictTrain != trainY)
            sum_of_trainError += trainError
        absolute_Average = sum_of_testError / numberOfExperiments
        absolute_trainAverageError = sum_of_trainError / numberOfExperiments
        min_array_per_lambda.append(absolute_min)
        max_array_per_lambda.append(absolute_max)
        average_array_per_lambda.append(absolute_Average)
        average_train_error.append(absolute_trainAverageError)
    return min_array_per_lambda, max_array_per_lambda, average_array_per_lambda, average_train_error


def doSoftSVM(l, m):
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']
    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]
    w = softsvm(l, _trainX, _trainy)
    return testX, testy, w, _trainX, _trainy


def runtask2():
    lambdas = [10 ** i for i in range(10)]
    min_array_per_lambda, max_array_per_lambda, average_array_per_lambda, average_train_error = task2(100, lambdas, 10)

    # yticks = [i for i in np.arange(0, 0.325, 0.025)]
    df = pd.DataFrame({'Average test Error': average_array_per_lambda,
                       'Max Error test Error': max_array_per_lambda,
                       'Lowest Error test Error': min_array_per_lambda,
                       'average train Error': average_train_error}, index=[f'10^{i + 1}' for i in range(10)])
    ax = df.plot.bar(rot=0)
    # ax.set_yticks(yticks)
    ax.set_xlabel("lambda values")
    ax.set_ylabel("Error")
    ax.set_title("Error of the soft-SVM on different lambda's values")
    # ax.tick_params(axis='y', labelsize=5)
    fig = ax.get_figure()
    fig.savefig("2a.png")

    lambda_for_B = [1, 3, 5, 8]
    lambdas = [10 ** i for i in lambda_for_B]
    min_array_per_lambda, max_array_per_lambda, average_array_per_lambda, average_train_error = task2(1000, lambdas, 1)

    # yticks = [i for i in np.arange(0, 0.325, 0.025)]
    df = pd.DataFrame({
        'Average test Error': average_array_per_lambda,
        'Average train Error': average_train_error}, index=[f'10^{i}' for i in lambda_for_B])
    # 'Max Error': max_array_per_lambda,
    # 'Lowest Error': min_array_per_lambda}

    ax = df.plot.bar(rot=0)
    # ax.set_yticks(yticks)
    ax.set_xlabel("lambda values")
    ax.set_ylabel("Error")
    ax.set_title("Error of the soft-SVM on different lambda's values")
    # ax.tick_params(axis='y', labelsize=5)
    fig = ax.get_figure()
    fig.savefig("2b.png")




if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    # simple_test()
    runtask2()
    # here you may add any code that uses the above functions to solve question 2

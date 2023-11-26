import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    sum_e = np.sum(np.exp(z))
    return np.exp(z) / sum_e

def logloss(y, y_hat):
    ans = 0
    for (i, j) in zip(y, y_hat):
      ans += np.sum((i * np.log(j) + (1 - i) * np.log(1 - j)))
    return -ans

def cross_entropy(y, y_hat):
    ans = 0
    for (i, j) in zip(y, y_hat):
      ans += np.sum(i * np.log(j))
    return -ans

def solution():
    n, k = map(int, input().split())
    y = np.array([list(map(int, input().split())) for _ in range(n)])
    z = np.array([list(map(float, input().split())) for _ in range(n)])

    y_hat_log = sigmoid(z)
    y_hat_ent = [softmax(x) for x in z]

    logloss_value = logloss(y, y_hat_log)
    crossentropy_value = cross_entropy(y, y_hat_ent)

    logloss_value = str(np.round(logloss_value, 3))
    crossentropy_value = str(np.round(crossentropy_value, 3))
    print(logloss_value + ' ' + crossentropy_value)


solution()
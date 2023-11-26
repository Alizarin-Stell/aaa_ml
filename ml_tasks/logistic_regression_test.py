import numpy as np


class LogisticRegression:

    def __init__(self, max_iter=1e3, lr=0.03, tol=0.001, l1_coef = 0.1):
        '''
        max_iter – максимальное количеств
        '''

        self.max_iter = max_iter
        self.lr = lr
        self.tol = tol
        self.l1_coef = l1_coef

        self.weights = None
        self.bias = None

    def fit(self, X_train, y_train):
        '''
        Обучение модели.

        X_train – матрица объектов для обучения
        y_train – ответы на объектах для обучения

        '''

        n, m = X_train.shape

        self.weights = np.zeros((m, 1))
        self.bias = np.mean(y_train)

        n_iter = 0
        gradient_norm = np.inf

        while n_iter < self.max_iter and gradient_norm > self.tol:
            dJdw, dJdb = self.grads(X_train, y_train)
            gradient_norm = np.linalg.norm(np.hstack([dJdw.flatten(), [dJdb]]))

            self.weights = self.weights - self.lr * dJdw
            self.bias = self.bias - self.lr * dJdb

            n_iter += 1

        return self

    def predict(self, X):
        '''
        Метод возвращает предсказанную метку класса на объектах X
        '''

        return (self.predict_proba(X) > 0.5).astype(int)

    def predict_proba(self, X):
        '''
        Метод возвращает вероятность класса 1 на объектах X
        '''
        z = X.dot(self.weights) + self.bias
        return self.sigmoid(z)

    def grads(self, x, y):
        '''
        Рассчёт градиентов
        '''
        y_hat = self.predict_proba(x)
        error = y_hat - y
        #print(np.sign(self.weights).shape)
        #print((error * x).mean(axis=0, keepdims=True).T.shape)
        dJdw = np.mean(x * error, axis=0, keepdims=True).T
        dJdw += self.l1_coef * (np.sign(self.weights).reshape(-1, 1))
        dJdb = error.mean()

        return dJdw, dJdb

    @staticmethod
    def sigmoid(x):
        '''
        Сигмоида от x
        '''
        return 1 / (1 + np.exp(-x))


# def read_input():
#     n, m, k = map(int, input().split())
#
#     x_train = np.array([input().split() for _ in range(n)]).astype(float)
#     y_train = np.array([input().split() for _ in range(n)]).astype(float)
#     x_test = np.array([input().split() for _ in range(k)]).astype(float)
#     return x_train, y_train, x_test
#
#
# def solution():
#     x_train, y_train, x_test = read_input()
#
#     model = LogisticRegression()
#     model.fit(x_train, y_train)
#
#     predictions = model.predict(x_test)
#
#     result = ' '.join(map(lambda x: str(int(x)), predictions))
#     print(result)
#     #print(model.predict_proba(x_train))


def read_input():
    n, m = map(int, input().split())
    x_train = np.array([input().split() for _ in range(n)]).astype(float)
    y_train = np.array([input().split() for _ in range(n)]).astype(float)
    return x_train, y_train


def solution():
    x_train, y_train = read_input()

    model = LogisticRegression(max_iter=5e3, lr=0.04, l1_coef=0.1)
    model.fit(x_train, y_train)

    all_weights = [model.bias] + list(model.weights.flatten())
    result = ' '.join(map(lambda x: str(float(x)), all_weights))
    print(result)


solution()

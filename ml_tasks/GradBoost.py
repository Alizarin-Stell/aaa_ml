import numpy as np


class MyDecisionTreeRegressor:

    def __init__(self, max_depth=None, max_features=None, min_leaf_samples=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_leaf_samples = min_leaf_samples
        self._node = {
                        'left': None,
                        'right': None,
                        'feature': None,
                        'threshold': None,
                        'depth': 0,
                        'value': 0
                    }
        self.tree = None  # словарь в котором будет храниться построенное дерево

    def fit(self, X, y):
        self.tree = {'root': self._node.copy()}  # создаём первую узел в дереве
        self._build_tree(self.tree['root'], X, y)  # запускаем рекурсивную функцию для построения дерева
        return self

    def predict(self, X):
        preds = [self._get_predict(self.tree['root'], x) for x in X]
        return np.array(preds)

    def mse(self, y):
        return np.mean((y - np.mean(y))**2)

    def calc_Q(self, y, y_left, y_right):
        mse_total = self.mse(y)
        mse_left = self.mse(y_left)
        mse_right = self.mse(y_right)
        return mse_total - (len(y_left) / len(y)) * mse_left \
            - (len(y_right) / len(y)) * mse_right

    def get_best_split(self, X, y):

        features = range(X.shape[1])

        if not self.max_features is None:
            features = np.random.choice(X.shape[1], self.max_features, replace=False)

        best_j = 0
        best_t = 0
        best_q = 0
        best_left_ids = []
        best_right_ids = []

        for feature in features:
          cur_feature = X[:, feature]

          unique_values = np.sort(np.unique(cur_feature))

          for i in range(len(unique_values) - 1):
            value = (unique_values[i] + unique_values[i + 1]) / 2

            left_ids = cur_feature <= value
            right_ids = cur_feature > value

            if np.sum(left_ids) == 0 or np.sum(right_ids) == 0:
                continue

            left_labels = y[left_ids]
            right_labels = y[right_ids]

            cur_q = self.calc_Q(y, left_labels, right_labels)

            if cur_q > best_q:
                best_q = cur_q
                best_t = value
                best_j = feature
                best_left_ids = left_ids
                best_right_ids = right_ids

        return best_j, best_t, best_left_ids, best_right_ids

    def _build_tree(self, curr_node, X, y):

        if curr_node['depth'] == self.max_depth or len(np.unique(y)) <= 1:  # выход из рекурсии если построили до максимальной глубины
            curr_node['value'] = np.mean(y)  # сохраняем предсказания листьев дерева перед выходом из рекурсии
            return

        j, t, left_ids, right_ids = self.get_best_split(X, y)  # нахождение лучшего разбиения

        curr_node['feature'] = j  # признак по которому производится разбиение в текущем узле
        curr_node['threshold'] = t  # порог по которому производится разбиение в текущем узле

        left = self._node.copy()  # создаём узел для левого поддерева
        right = self._node.copy()  # создаём узел для правого поддерева

        left['depth'] = curr_node['depth'] + 1  # увеличиваем значение глубины в узлах поддеревьев
        right['depth'] = curr_node['depth'] + 1

        curr_node['left'] = left
        curr_node['right'] = right

        self._build_tree(left, X[left_ids], y[left_ids])  # продолжаем построение дерева
        self._build_tree(right, X[right_ids], y[right_ids])

    def _get_predict(self, node, x):
        if node['threshold'] is None:  # если в узле нет порога, значит это лист, выходим из рекурсии
            return node['value']

        if x[node['feature']] <= node['threshold']:  # уходим в правое или левое поддерево в зависимости от порога и признака
            return self._get_predict(node['left'], x)
        else:
            return self._get_predict(node['right'], x)


class MyGradientBoostingRegressor:

    def __init__(self, learning_rate, max_depth, max_features, n_estimators):
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.trees = []
        self.initial_model = None

    def fit(self, X, y):
        self.initial_model = np.mean(y)

        residuals = y - self.initial_model

        for _ in range(self.n_estimators):
            tree = MyDecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)

            predictions = tree.predict(X).reshape(-1, 1)
            predictions = np.nan_to_num(predictions)
            residuals -= self.learning_rate * predictions

    def predict(self, X):
        y_pred = np.full(X.shape[0], self.initial_model)

        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)

        return y_pred.reshape(-1, 1)


def read_matrix(n, dtype=float):
    matrix = np.array([list(map(dtype, input().split())) for _ in range(n)])
    return matrix

def read_input_matriсes(n, m, k):
    X_train, y_train, X_test = read_matrix(n), read_matrix(n), read_matrix(k)
    return X_train, y_train, X_test

def print_matrix(matrix):
    for row in matrix:
        print(' '.join(map(str, row)))

def solution():
    n, m, k = map(int, input().split())
    X_train, y_train, X_test = read_input_matriсes(n, m, k)

    gb = MyGradientBoostingRegressor(learning_rate=0.1, max_depth=4,
                                     max_features=m, n_estimators=45)
    gb.fit(X_train, y_train)

    predictions = gb.predict(X_test)
    print_matrix(predictions)

#solution()


import pandas as pd


def mape(y_true, y_pred):
    non_zero_mask = y_true != 0
    print(sum(non_zero_mask))
    y_true, y_pred = y_true[non_zero_mask], y_pred[non_zero_mask]
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def solution1():
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    val_const = data.shape[0]

    x_train = data[:val_const]
    y_train = target[:val_const].reshape(-1, 1)
    x_test = data[val_const:]
    y_test = target[val_const:].reshape(-1, 1)

    gb = MyGradientBoostingRegressor(learning_rate=0.1, max_depth=3,
                                     max_features=data.shape[1], n_estimators=55)

    gb.fit(x_train, y_train)

    predictions = gb.predict(x_train)

    print(mape(y_train, predictions))


solution1()

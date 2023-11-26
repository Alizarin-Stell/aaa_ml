import numpy as np


class RNN:

    def __init__(self, in_features, hidden_size, n_classes, activation='tanh'):
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.activation = activation
        self.Wax = self.init_weight_matrix((self.hidden_size, self.in_features))
        self.Waa = self.init_weight_matrix((self.hidden_size, self.hidden_size))
        self.Wya = self.init_weight_matrix((self.n_classes, self.hidden_size))
        self.ba = self.init_weight_matrix((self.hidden_size, 1))
        self.by = self.init_weight_matrix((self.n_classes, 1))


    def init_weight_matrix(self, size):
        np.random.seed(1)
        W = np.random.uniform(size=size)
        return W

    def tanh(self, x):
        ex = np.exp(-2*x)
        return (1 - ex) / (1 + ex)

    def soft_max(self, x):
        e_x = np.exp(x)
        return e_x / e_x.sum()

    def forward(self, x):
        x = x.T

        mem = np.zeros((self.hidden_size, 1))
        it_ans = np.zeros((self.n_classes, len(x)))
        for i, el in enumerate(x):
            el = el.reshape(-1, 1)
            ans = self.tanh(self.Waa @ mem + self.Wax @ el + self.ba)
            mem = ans
            y = self.Wya @ ans + self.by
            it_ans[:,i] = self.soft_max(y).squeeze()
        return it_ans


def read_matrix(n_rows, dtype=float):
    return np.array([list(map(dtype, input().split())) for _ in range(n_rows)])


def print_matrix(matrix):
    for row in matrix:
        print(' '.join(map(str, row)))


def solution():
    in_features, hidden_size, n_classes = map(int, input().split())
    input_vectors = read_matrix(in_features)

    rnn = RNN(in_features, hidden_size, n_classes)
    output = rnn.forward(input_vectors).round(3)
    print_matrix(output)

solution()
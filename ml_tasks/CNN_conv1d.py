import numpy as np


class Conv1d:

    def __init__(self, in_channels, out_channels, kernel_size, padding='same', activation='relu'):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation

        self.W, self.biases = self.init_weight_matrix()

    def init_weight_matrix(self,):
        np.random.seed(1)
        W = np.random.uniform(size=(self.in_channels, self.kernel_size, self.out_channels))
        biases = np.random.uniform(size=(1, self.out_channels))
        return W, biases

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        pad_width = (self.kernel_size - 1) // 2
        x = np.pad(x, pad_width=((0, 0), (pad_width, pad_width)),
                   mode='constant', constant_values=0)

        output_length = x.shape[1] - self.kernel_size + 1
        ans = np.zeros((self.out_channels, output_length))

        for i in range(self.out_channels):
            for j in range(output_length):
                temp = x[:, j:j+self.kernel_size]
                ans[i, j] = np.sum(temp * self.W[:, :, i]) + self.biases[0, i]

        ans = self.relu(ans)
        return ans


def read_matrix(n_rows, dtype=float):
    return np.array([list(map(dtype, input().split())) for _ in range(n_rows)])


def print_matrix(matrix):
    for row in matrix:
        print(' '.join(map(str, row)))


def solution():
    in_channels, out_channels, kernel_size = map(int, input().split())
    input_vectors = read_matrix(in_channels)

    conv = Conv1d(in_channels, out_channels, kernel_size)
    output = conv.forward(input_vectors).round(3)
    print_matrix(output)

solution()
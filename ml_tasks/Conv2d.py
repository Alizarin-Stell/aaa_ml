import numpy as np


class Conv2d:

    def __init__(
        self, in_channels, out_channels, kernel_size_h, kernel_size_w, padding=0, stride=1
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.padding = padding
        self.stride = stride

        self.W, self.biases = self.init_weight_matrix()

    def init_weight_matrix(self,):
        np.random.seed(1)
        W = np.random.uniform(size=(
            self.in_channels, self.kernel_size_h,
            self.kernel_size_w, self.out_channels
        ))
        biases = np.random.uniform(size=(1, self.out_channels))
        return W, biases

    def forward(self, x):
        x = np.pad(x, ((0, 0), (self.padding, self.padding),
                       (self.padding, self.padding)),
                   mode='constant', constant_values=0)
        coords = x.shape
        out_h = (coords[1] - self.kernel_size_h) // self.stride + 1
        out_w = (coords[2] - self.kernel_size_w) // self.stride + 1

        ans = np.zeros((self.out_channels, out_h, out_w))

        for cur_chan in range(self.out_channels):
            out_channel = np.zeros((out_h, out_w))
            st_i = 0

            for i in range(0, coords[1] - self.kernel_size_h + 1, self.stride):
                st_j = 0
                for j in range(0, coords[2] - self.kernel_size_w + 1, self.stride):
                    val = x[:, i:i+self.kernel_size_h,
                          j:j+self.kernel_size_w] *\
                          self.W[:, :, :, cur_chan]
                    out_channel[st_i, st_j] = np.sum(val) +\
                                              self.biases[0, cur_chan]
                    st_j += 1
                st_i += 1
            ans[cur_chan] = out_channel

        return ans



def read_matrix(in_channels, h, w, dtype=float):
    return np.array([list(map(dtype, input().split()))
                     for _ in range(in_channels * h)]).reshape(in_channels, h, w)

def print_matrix(matrix):
    for channel in matrix:
        for row in channel:
            print(' '.join(map(str, row)))

def solution():
    in_channels, out_channels, kernel_size_h, kernel_size_w, h, w, padding, stride = map(int, input().split())
    input_image = read_matrix(in_channels, h, w)

    conv = Conv2d(in_channels, out_channels, kernel_size_h, kernel_size_w, padding, stride)
    output = conv.forward(input_image).round(3)
    print_matrix(output)

solution()
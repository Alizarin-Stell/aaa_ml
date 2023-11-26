import numpy as np


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class CBOW:

    def __init__(self, vocab_size: int, embedding_dim: int,
                 random_state: int = 1):
        np.random.seed(random_state)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embeddings = self.init_weight_matrix()
        self.contexts = self.init_weight_matrix().T

    def init_weight_matrix(self, ):
        W = np.random.uniform(size=(self.vocab_size, self.embedding_dim))
        return W

    def forward(self, x):
        sum_embedding = np.sum(self.embeddings[x], axis=0)
        output = np.dot(sum_embedding, self.contexts)
        return softmax(output)

def read_vector(dtype=int):
    return np.array(list(map(dtype, input().split())))

def solution():
    vocab_size, embedding_dim = read_vector()
    input_vector = read_vector()

    cbow = CBOW(vocab_size, embedding_dim)
    output = cbow.forward(input_vector).round(3)
    print(' '.join(map(str, output)))


solution()

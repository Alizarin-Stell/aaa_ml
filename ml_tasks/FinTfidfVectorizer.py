import string
from collections import Counter, defaultdict

import numpy as np


class TfidfVectorizer:

    def __init__(self):
        self.sorted_vocab = defaultdict(int)
        self.idf = []

    def tf_transform(self, count_matrix: np.array) -> list:
        word_sum = np.sum(count_matrix, axis=1, keepdims=True)
        word_sum = np.where(word_sum == 0, 1, word_sum)
        tf_matrix = count_matrix / word_sum
        return tf_matrix

    def idf_transform(self, count_matrix: np.array) -> list:
        total_docs = count_matrix.shape[0]
        word_presence = np.where(count_matrix > 0, 1, 0)
        docs_per_word = np.sum(word_presence, axis=0)
        idf_values = np.log(total_docs / docs_per_word)
        return idf_values

    def fit(self, X):
        self.idf = Counter(
            ' '.join([' '.join(set(doc.split())) for doc in X]).split())
        for key in self.idf:
            self.idf[key] = np.log(len(X) / self.idf[key])
        self.sorted_vocab = dict(sorted(self.idf.items()))

        return self

    def transform(self, X):
        vectors = []
        for doc in X:
            vector = []
            word_counts = Counter(doc.split())
            sum_doc = sum(word_counts.values())
            for word in self.sorted_vocab.keys():
                if word in word_counts:
                    vector.append((word_counts[word] / sum_doc)
                                  * self.sorted_vocab[word])
                else:
                    vector.append(0.0)
            vectors.append(vector)

        return vectors


def read_input():
    n1, n2 = map(int, input().split())

    train_texts = [input().strip() for _ in range(n1)]
    test_texts = [input().strip() for _ in range(n2)]

    return train_texts, test_texts

def solution():
    train_texts, test_texts = read_input()
    vectorizer = TfidfVectorizer()
    vectorizer.fit(train_texts)
    transformed = vectorizer.transform(test_texts)

    for row in transformed:
        row_str = ' '.join(map(str, np.round(row, 3)))
        print(row_str)

solution()
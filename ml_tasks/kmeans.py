import numpy as np

# следует реализовать данный вид инициализации
# без нее тесты скорее всего не пройдут
def k_plus_plus(X: np.ndarray, k: int, random_state: int = 27) -> np.ndarray:
    """Инициализация центроидов алгоритмом k-means++.

    :param X: исходная выборка
    :param k: количество кластеров
    :return: набор центроидов в одном np.array
    """
    np.random.seed(random_state)

    n = X.shape[0]
    start_ind = np.random.randint(0, n)

    centroids = [X[start_ind]]

    for _ in range(1, k):
        distances = np.array(
            [min([np.linalg.norm(x - c) ** 2 for c in centroids]) for x in X])

        probabilities = distances / distances.sum()
        cumulative_probabilities = np.cumsum(probabilities)
        r = np.random.rand()

        for i, p in enumerate(cumulative_probabilities):
            if r < p:
                centroids.append(X[i])
                break

    return np.array(centroids)


class KMeans:
    def __init__(self, n_clusters=8, tol=0.0001, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.tol = tol
        self.max_iter = max_iter
        self.n_iter_ = 0
        self.random_state = random_state

    def fit(self, X):
        np.random.seed(self.random_state)

        n_samples = X.shape[0]
        n_features = X.shape[1]

        # инициализируем центры кластеров
        # centers.shape = (n_clusters, n_features)
        centers = k_plus_plus(X=X, k=self.n_clusters)
        labels = np.zeros(n_samples)
        distances = np.zeros(n_samples)

        for n_iter in range(self.max_iter):
            # считаем расстояние от точек из X до центроидов
            distances = np.array(
                [min([np.linalg.norm(x - c) ** 2 for c in centers]) for x in X])
            # определяем метки как индекс ближайшего для каждой точки центроида
            labels = np.array(
                [np.argmin([np.linalg.norm(x - c) for c in centers]) for x in X])


            old_centers = centers.copy()
            for c in range(self.n_clusters):
                # пересчитываем центроид
                # новый центроид есть среднее точек X с меткой рассматриваемого центроида

                cur_values = X[labels == c]
                if len(cur_values) > 0:
                    centers[c, :] = np.mean(cur_values, axis=0)

            self.n_iter_ = n_iter

            # записываем условие сходимости
            # норма Фробениуса разности центров кластеров двух последовательных итераций < tol
            if np.linalg.norm(centers - old_centers) < self.tol or \
                    n_iter == self.max_iter - 1:
                break

        # cчитаем инерцию
        # сумма квадратов расстояний от точек до их ближайших центров кластеров
        inertia_values = np.array(
                [min([np.linalg.norm(x - c) ** 2 for c in centers]) for x in X])

        self.cluster_centers_ = centers
        self.labels_ = labels
        self.distances_ = distances

        self.inertia_ = np.sum(inertia_values)

        return self


    def predict(self, X):
        # определяем метку для каждого элемента X на основании обученных центров кластеров

        distances = np.array([min([np.linalg.norm(x - c) ** 2
                      for c in self.cluster_centers_]) for x in X])
        labels = np.array([np.argmin([np.linalg.norm(x - c) ** 2
                            for c in self.cluster_centers_]) for x in X])
        return labels

    def fit_predict(self, X):
        return self.fit(X).labels_


def read_input():
    n1, n2, k = map(int, input().split())

    read_line = lambda x: list(map(float, x.split()))
    X_train = np.array([read_line(input()) for _ in range(n1)])
    X_test = np.array([read_line(input()) for _ in range(n2)])

    return X_train, X_test, k

def solution():
    X_train, X_test, k = read_input()
    kmeans = KMeans(n_clusters=k, tol=1e-8, random_state=27)
    kmeans.fit(X_train)
    train_labels = kmeans.labels_
    test_labels = kmeans.predict(X_test)

    print(' '.join(map(str, train_labels)))
    print(' '.join(map(str, test_labels)))

solution()
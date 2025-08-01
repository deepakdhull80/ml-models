import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k=2, num_iter=10):
        self.k = k
        self.clusters = None
        self.num_iter = num_iter
    
    def fit(self, X):
        n_samples, n_features = X.shape
        idx = np.random.choice(np.arange(n_samples), size=self.k, replace=False)
        self.clusters = X[idx]
        
        for _ in range(self.num_iter):
            # update labels
            labels = self.predict(X)
            # update centroid
            for i in range(self.k):
                self.clusters[i] = X[labels == i].mean(axis=0)
    
    def _distance(self, x1, x2):
        return np.sqrt(np.mean((x1-x2) ** 2, axis=1))
    
    def predict(self, X):
        labels = []
        for point in X:
            dist = self._distance(self.clusters, point)
            labels.append(np.argmin(dist))
        return np.array(labels)



if __name__ == "__main__":
    type_1 = np.random.normal(0, 1, (100, 2))
    type_2 = np.random.normal(1, 2, (100, 2))
    type_3 = np.random.normal(-1, 1, (100, 2))
    
    X = np.concatenate([type_1, type_2, type_3], axis=0)
    labels = np.arange(3).reshape(-1, 1).repeat(100, axis=1).reshape(-1)
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    
    
    model = KMeans(k=3, num_iter=10)
    model.fit(X)
    pred_labels = model.predict(X)
    
    
    for point in model.clusters:
        plt.plot(point[0], point[1], 'x')
    
    plt.show()
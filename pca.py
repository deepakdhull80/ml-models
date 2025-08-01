import numpy as np
import matplotlib.pyplot as plt

class PCA:
    def __init__(self, k=2):
        self.k = k
    
    def fit(self, X):
        n_sample, n_features = X.shape
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        X = (X - self.mean) / (self.std + 1e-7)
        
        cov_matrix = np.cov(X.T)
        e_values, e_vectors = np.linalg.eigh(cov_matrix)
        e_values = e_values/e_values.sum()
        sorted_idx = np.argsort(e_values)[::-1]
        # column vectors are eigen vectors
        self.pc = e_vectors[:, sorted_idx][:, :self.k]
        self.feature_importance = e_values[sorted_idx]
        print(np.cumsum(self.feature_importance[: self.k]))
    
    def transform(self, X):
        X = (X - self.mean) / (self.std + 1e-7)
        return np.dot(X, self.pc)


if __name__ == "__main__":
    X1 = np.random.normal(-2, 2, (1000, 32))
    label1 = np.ones((1000,1)) * 1
    X2 = np.random.normal(0, 1, (1000, 32))
    label2 = np.ones((1000,1)) * 2
    X3 = np.random.normal(2, 5, (1000, 32))
    label3 = np.ones((1000,1)) * 3
    X = np.vstack([X1, X2, X3])
    labels = np.vstack([label1, label2, label3])
    model = PCA()
    
    model.fit(X)
    X_transform = model.transform(X)
    plt.scatter(X_transform[:, 0], X_transform[:, 1], c=labels)
    plt.show()
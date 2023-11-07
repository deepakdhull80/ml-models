import numpy as np

class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components_ = None

    def fit(self, x):
        u,s,v = np.linalg.svd(x)
        norm_s = s/np.linalg.norm(s + 1e-6)
        idx = np.argsort(-norm_s)
        v = v[idx]
        self.components_ = v[:, :self.n_components]
        return self

    def predict(self, x):
        return np.dot(x, self.components_)

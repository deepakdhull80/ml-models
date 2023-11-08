import numpy as np

"""
### weakness:
1. not able to handle the cov singularity issue while updating
2. its work fine when n_feature <= 2, when we increase it to more than 2 cov determinant become zero.
"""

class GaussianMixture:
    def __init__(self, n_clusters, iterations = 30, debug=False):
        self.n_clusters = n_clusters
        self.iterations = iterations
        self.epsilon = 1e-6
        self.debug = debug
        # init params
        

    def init_weigh(self, x):
        self.alpha = np.ones((self.n_clusters, 1)) * 1/self.n_clusters
        self.mean = x[np.random.randint(0, high=x.shape[0], size=(self.n_clusters))]
        tmp = np.random.rand(self.n_clusters, x.shape[1], x.shape[1])
        self.cov = np.matmul(tmp, np.transpose(tmp, axes=(0,2,1)))

    def _debugger(self):
        print(self.alpha)
        print(self.mean)
        print(self.cov)
    def _pdf_batch(X, mean, covariance):
        # need to verify
        n = X.shape[1]
        det_cov = np.abs(np.linalg.det(covariance))
        inv_cov = np.linalg.inv(covariance)
        constant = 1 / ((2 * np.pi) ** (n / 2) * np.sqrt(det_cov))
        X_mean = X - mean
        exponent = -0.5 * np.diagonal(X_mean @ inv_cov @ X_mean.T)
        pdf = constant * np.exp(exponent)
        return pdf
        
    def pdf_batch(self, X):
        return np.array([GaussianMixture._pdf_batch(X, self.mean[i, :], self.cov[i, :]) for i in range(self.mean.shape[0])])

    def expectation(self, x: np.ndarray):
        # Correct
        x = self.alpha * self.pdf_batch(x) # n_cluster x N
        scale = np.sum(x, axis=0) + self.epsilon
        return x / scale

    def update(self, q, x):
        for i in range(self.n_clusters):
            self.alpha[i, :] = np.sum(q[i, :])/x.shape[0] + self.epsilon
            scale = np.sum(q[i, :]) + self.epsilon
            self.mean[i, :] = np.sum(q[i, :, None] * x, axis=0) / scale  + self.epsilon
            self.cov[i, :] = (q[i, :, None] * (x-self.mean[i, :])).T @ (x-self.mean[i, :]) / scale + self.epsilon

    def log_likelihood(self, q):
        log_likelihood = np.sum(q * np.log(self.alpha * pdf_batch(x, self.mean, self.cov) + self.epsilon))
        return -log_likelihood
        
    def fit(self, x, _ver = 5):
        self.init_weigh(x)
        for i in range(self.iterations):
            q = self.expectation(x)
            self.update(q, x)
            if i % _ver == 0:
                print(self.log_likelihood(q))
                # self._debugger()
    def predict(self, x):
        return np.argmax(self.expectation(x), axis=0)

if __name__ == "__main__":
  dim = 2
  x1 = np.random.normal(loc=10, scale=2.0, size=(100,dim))
  x2 = np.random.normal(loc=0, scale=2.0, size=(100,dim))
  x3 = np.random.normal(loc=-10, scale=2.0, size=(100,dim))
  x = np.concatenate([x1, x2, x3])
  np.random.shuffle(x)


  m = GaussianMixture(3, iterations=30)
  m.fit(x)
  print("mean:",m.mean , "cov", m.cov)

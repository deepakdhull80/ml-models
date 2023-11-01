import numpy as np
import matplotlib.pyplot as plt

class SVMDualForm:
    def __init__(self, C=1.0, kernel='rbf', degree=1, sigma=0.1):
        self.degree = degree
        self.kernel = kernel
        self.C = C
        self.sigma = sigma
        
        self.kernel_function = self.rbf_kernel_function if self.kernel == 'rbf' else self.poly_kernel_function
    
    def rbf_kernel_function(self, X1, X2):
        """ radical basis kernel function

        Args:
            X1 (np.ndarray): input matrix
            X2 (np.ndarray): input matrix

        Returns:
            np.ndarray: pairwise matrix 
        """
        # pairwise difference (x1 and x2) -> shape-(n,m,d)
        pairwise_diff = X1[:, np.newaxis] - X2[np.newaxis, :]
        # norm ord=l2
        return np.exp(-(1 / self.sigma ** 2) * np.linalg.norm(pairwise_diff, axis=2) ** 2)
    
    def poly_kernel_function(self, X1, X2):
        """ polynomial kernel function

        Args:
            X1 (np.ndarray): input matrix
            X2 (np.ndarray): input matrix

        Returns:
            np.ndarray: pairwise matrix 
        """
        return (self.C + X1.dot(X2.T)) ** self.degree
    
    def fit(self, X, y, lr=1e-3, epochs=500):
    
        self.X = X
        self.y = y
    
        # (n,)
        self.alpha = np.random.random(X.shape[0])
        self.b = 0
        # (n,)
        self.ones = np.ones(X.shape[0])
        #(n,n) =      (n,n) *        (n,n)
        y_iy_jk_ij = np.outer(y, y) * self.kernel_function(X, X)
        
        losses = []
        for _ in range(epochs):
            #gradient has been computed, update α as per gradient ascent rule
            # (n,)  =    (n,)      (n,n).(n,)=(n,)
            gradient = self.ones - y_iy_jk_ij.dot(self.alpha)
            self.alpha = self.alpha + lr * gradient
            #Clip the α values accordingly.
            self.alpha[self.alpha > self.C] = self.C
            self.alpha[self.alpha < 0] = 0
            #                                        (500,500)                            (500,500)
            loss = np.sum(self.alpha) - 0.5 * np.sum(np.outer(self.alpha, self.alpha) * y_iy_jk_ij)
            losses.append(loss)
            
        index = np.where((self.alpha) > 0 & (self.alpha < self.C))[0]
        #(m,)= (m,)       (n,).(n,m)= (m,)
        b_i = y[index] - (self.alpha * y).dot(self.kernel_function(X, X[index]))
        self.b = np.mean(b_i)
        plt.plot(losses)
        plt.title("loss per epochs")
        plt.show()
    
    def _decision_function(self, X):
        return (self.alpha * self.y).dot(self.kernel_function(self.X, X)) + self.b
    
    def predict(self, X):
        return np.sign(self._decision_function(X))
 
    def score(self, X, y):
        y_hat = self.predict(X)
        return np.mean(y == y_hat)

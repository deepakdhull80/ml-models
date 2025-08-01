import numpy as np
import matplotlib.pyplot as plt


def rbf_kernel(xi, xj, std=0.5):
    return np.exp(-(xi-xj)**2/(2*std))

class SVM:
    def __init__(self, n_iter=10, lr=0.01, lambda_param=0.5):
        self.n_iter = n_iter
        self.lr = lr
        self.lambda_param = lambda_param
    
    def fit(self , X, y):
        # initialize the hyperplane
        y = np.where(y==0, -1, 1)
        n_samples, n_features = X.shape
        self.w = np.random.randn(n_features)
        self.b = np.random.randn(1)
        
        # optimize the hyperplane
        '''
        loss = max(0, 1 - y(wTx + b)) + lambda * W**2
        '''
        
        for _ in range(self.n_iter):
            for idx in range(n_samples):
                logit = np.dot(self.w, X[idx]) + self.b
                cond = logit * y[idx] >= 1
                if cond:
                    self.w -= 2*self.lr * self.lambda_param * self.w
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - X[idx] * y[idx] )
                    self.b -= self.lr * (-1 * y[idx])
        
    
    def predict(self, X):
        return np.where(np.dot(X, self.w) + self.b >=0, 1, 0)

def accuracy(y, y_pred):
    return np.sum(y == y_pred)/y.shape[0]

if __name__ == "__main__":
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    x, y = make_classification(1000, 10, n_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, stratify=y)
    
    model = SVM(n_iter=100)
    model.fit(X_train, y_train)
    print("Train Acc:", accuracy(y_train, model.predict(X_train)))
    print("Test Acc:", accuracy(y_test, model.predict(X_test)))
import numpy as np

"""
This is called the Perceptron criterion, defined over all misclassified points:
L(w,b)= i∈misclassified∑​ −y (i)(w ⊤x (i)+b)
"""

class Perceptron:
    def __init__(self, num_iter=100, lr=0.1):
        self.num_iter = num_iter
        self.lr = lr
    
    def activation_func(self, x):
        return np.where(x >= 0, 1, -1)
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.random.normal(0, 1, size=(n_features))
        self.bias = np.random.normal(0, 1, size=(1))
        
        y_ = np.where(y == 0, -1, 1)
        
        for _ in range(self.num_iter):
            for idx in range(n_samples):
                
                logit = np.dot(X[idx], self.w) + self.bias
                y_pred = self.activation_func(logit)
                if y_[idx] != y_pred:
                    self.w += self.lr * np.dot(X[idx], y_[idx])
                    self.bias += self.lr * y_[idx]
    
    def predict(self, X):
        return self.activation_func(np.dot(X, self.w) + self.bias)


def accuracy(y, y_pred):
    return np.sum(y == y_pred)/y.shape[0]

if __name__ == "__main__":
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    x, y = make_classification(1000, 10, n_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, stratify=y)
    
    model = Perceptron(num_iter=1000, lr=0.01)
    model.fit(X_train, y_train)
    print("Train Acc:", accuracy(y_train, model.predict(X_train)))
    print("Test Acc:", accuracy(y_test, model.predict(X_test)))
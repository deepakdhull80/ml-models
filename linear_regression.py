import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, num_iter=100, lr=0.1):
        self.num_iter = num_iter
        self.lr = lr
        self.alpha = 0.1
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros((n_features))
        self.b = 0
        y_pred = self.predict(X)
        print(f"init:", mse(y, y_pred))
        
        for epoch in range(self.num_iter):
            for idx in range(n_samples):
                y_pred = np.dot(self.w, X[idx, :]) + self.b
                
                dw = (y_pred - y[idx]) * X[idx, :]
                db = (y_pred - y[idx])
                self.w -= self.lr * dw + self.alpha*self.w**2
                self.b -= self.lr * db

            y_pred = self.predict(X)
            print(f"epoch {epoch}:", mse(y, y_pred))
    
    def predict(self, X):
        return np.dot(X, self.w) + self.b


def mse(y, y_pred):
    return ((y - y_pred) ** 2).mean()

if __name__ == "__main__":
    # prepare data
    x = np.random.normal(0, 1, size=(1000, 1))
    w = np.random.normal(x.shape[1])
    b = np.random.normal(1)
    noise = np.random.normal(0,1, (1000, 1)) * 0.05
    y = x * w + b + noise
    
    train_size = 0.8
    train_samples = int(train_size * x.shape[0])
    idx = np.arange(0, x.shape[0])
    np.random.shuffle(idx)
    train_idx, test_idx = idx[:train_samples], idx[train_samples:]
    train_x, train_y = x[train_idx], y[train_idx]
    test_x, test_y = x[test_idx], y[test_idx]
    
    model = LinearRegression()
    model.fit(train_x, train_y)
    
    y_train_pred = model.predict(train_x)
    y_pred = model.predict(test_x)
    print("Train MSE:", mse(train_y, y_train_pred))
    print("Test MSE:", mse(test_y, y_pred))
    
    plt.scatter(x, y)
    
    plane = np.linspace(x.min(), x.max(), 1000).reshape(-1, 1)
    pred = model.predict(plane)
    print(pred.shape)
    plt.scatter(plane, pred, alpha=0.2)
    plt.show()
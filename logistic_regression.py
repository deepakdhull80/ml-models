import numpy as np

class LogisticRegression:
    def __init__(self, num_iter=100, lr=0.1):
        self.num_iter = num_iter
        self.lr = lr
        self.weight = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1/(1+ np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weight = np.zeros(shape=(n_features))
        self.bias = 0
        
        
        prob = self.predict_prob(X)
        print(f"Init:", self.bce(y, prob))
        
        for epoch in range(self.num_iter):
            for idx in range(n_samples):
                x_i, y_i = X[idx, :].reshape(-1, n_features), y[idx]
                pred = np.dot(x_i, self.weight) + self.bias
                pred_prob = self.sigmoid(pred)
                dw = np.dot((pred_prob - y_i), x_i)
                db = pred_prob - y_i
                
                self.weight -= self.lr * dw
                self.bias -= self.lr * db.item()

            prob = self.predict_prob(X)
            print(f"Epoch ({epoch}):", self.bce(y, prob))
            
    def bce(self, y, y_pred):
        loss = -1 * np.mean(y * np.log(y_pred) + (1-y) * np.log(1-y_pred))
        return loss
    
    def predict_prob(self, X):
        return self.sigmoid(np.dot(X, self.weight) + self.bias)
    
    def predict(self, X):
        return (self.predict_prob(X) > 0.5).astype('int')

def accuracy(y, y_pred):
    return np.sum(y == y_pred)/y.shape[0]

if __name__ == "__main__":
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    x, y = make_classification(1000, 10, n_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, stratify=y)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    print("Train Acc:", accuracy(y_train, model.predict(X_train)))
    print("Test Acc:", accuracy(y_test, model.predict(X_test)))
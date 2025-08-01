import numpy as np

"""
* k nearest neighbor *



"""

class KNN:
    def __init__(self, k):
        self.k = k
        self.X = None
        self.y = None
    
    def _norm(self, x):
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        x = x/(norm + 1e-7)
        return x
    
    def fit(self, X, y):
        self.X = self._norm(X)
        
        self.y = y
    
    def _distance(self, x1, x2):
        return np.mean((x1 - x2) ** 2)
    
    def _predict_sample(self, x):
        res = []
        for i in range(self.X.shape[0]):
            res.append(self._distance(x, self.X[i, :]))
        
        res = np.array(res)
        top_idx = np.argsort(res)[:self.k]
        return self.y[top_idx].reshape(1, -1)
    
    def predict(self, X):
        res = []
        X = self._norm(X)
        for i in range(X.shape[0]):
            res.append(self._predict_sample(X[i, :]))
        pred = np.concatenate(res, axis=0)
        return pred


def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / y_pred.shape[0]

if __name__ == "__main__":
    # prepare dataset
    class_1_data = np.random.normal(0, 1, (100, 5))
    label_1 = np.zeros(100, )
    class_2_data = np.random.normal(50, 1, (100, 5))
    label_2 = np.ones(100, )
    X = np.concatenate([class_1_data, class_2_data], axis=0)
    y = np.concatenate([label_1, label_2], axis=0)
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    X, y = make_classification(1000, n_classes=2, n_features=20)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y)
    print(X_train.shape, X_test.shape)
    model = KNN(k=1)
    model.fit(X_train, y_train)
    
    y_predict = model.predict(X_test)
    print(y_predict.shape, y_test.shape)
    print(accuracy(y_test, y_predict[:, 0]))
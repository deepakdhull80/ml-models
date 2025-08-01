import numpy as np
from collections import Counter
from decision_tree import DecisionTree


class RandomForest:
    def __init__(self, n_estimator=2, min_samples=5, max_depth=5, max_n_features=None):
        self.n_estimator = n_estimator
        self.trees = [
            DecisionTree(
                max_depth=max_depth, 
                minimum_sample_split=min_samples, 
                max_n_features=max_n_features
                ) for _ in range(self.n_estimator)
            ]
        
    
    def fit(self, X, y):
        
        for i in range(self.n_estimator):
            sample_idx = np.random.choice(np.arange(X.shape[0]), size=(X.shape[0]), replace=True)
            sample_X, sample_y = X[sample_idx], y[sample_idx]
            self.trees[i].fit(sample_X, sample_y)
    
    
    def predict(self, X):
        res = []
        for idx in range(X.shape[0]):
            count = Counter([tree.predict(X[idx].reshape(1, -1)).item() for tree in self.trees])
            res.append(count.most_common(1)[0][0])
        
        return np.array(res)


def accuracy(y, y_pred):
    return (y == y_pred).sum() / y.shape[0]

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2)
    model = RandomForest(n_estimator=5, max_depth=10, min_samples=2, max_n_features=4)
    model.fit(X, y)
    y_pred = model.predict(X)
    print("Accuracy:", accuracy_score(y, y_pred))
            
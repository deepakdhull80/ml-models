import numpy as np

class Node:
    def __init__(self, left=None, right=None, feature=None, threshold=None, label=None, is_leaf=False, information_gain=0):        
        self.left = left
        self.right = right
        self.feature = feature
        self.threshold = threshold
        self.label = label
        self.is_leaf = is_leaf
        self.information_gain = information_gain
    

class DecisionTree:
    def __init__(self, max_depth=5, minimum_sample_split=5, max_n_features=None):
        self.max_depth = max_depth
        self.minimum_sample_split = minimum_sample_split
        self.max_n_features = max_n_features
        self.root = None
    
    def fit(self, X, y):
        if self.max_n_features is None:
            self.max_n_features = X.shape[1]
        self.root = self._make_tree(X, y, 0)
    
    def _impurity(self, y, type='gini'):
        if len(y) == 0:
            return 0
        y = y.astype(int)
        prob = np.bincount(y) / y.shape[0]
        # gini impurity
        if type == 'gini':
            return 1 - np.sum(prob ** 2)

        elif type == 'entropy':
            prob = prob[prob > 0]
            return -1 * np.sum(prob * np.log(prob))
        
        raise ValueError("Selected Unknown impurity fn")
    

    
    def _make_split(self, X, y):
        n_samples, n_features = X.shape
        parent_gini = self._impurity(y)
        max_value = -1
        feature_idx = None
        threshold = None
        feature_indices = np.random.choice(range(n_features), size=(min(n_features, self.max_n_features)), replace=False)
        for i in feature_indices:
            sorted_vals = np.unique(X[:, i])
            thresholds = (sorted_vals[1:] + sorted_vals[:-1]) / 2
            for thres in thresholds:
                left = X[:, i] <= thres
                right = ~left
                if left.sum() == 0 or right.sum() == 0:
                    continue
                children = (left.sum()/n_samples) * self._impurity(y[left]) + (right.sum()/n_samples) * self._impurity(y[right])
                value = parent_gini - children
                if value > max_value:
                    max_value = value
                    feature_idx = i
                    threshold = thres
        
        return feature_idx, threshold, max_value
    
    def _make_tree(self, X, y, depth):
        # make split
        feature, threshold, information_gain = self._make_split(X, y)
        # check base condition for leaf
        if feature is None or threshold is None or X.shape[0] <= self.minimum_sample_split or depth >= self.max_depth:
            label = np.argmax(np.bincount(y))
            return Node(label=label, is_leaf=True)
        
        left_idx = X[:, feature] <= threshold
        right_idx = ~left_idx
        left = self._make_tree(X[left_idx], y[left_idx], depth+1)
        right = self._make_tree(X[right_idx], y[right_idx], depth+1)
        
        return Node(left=left, right=right, feature=feature, threshold=threshold, information_gain=information_gain)

    def _predict_sample(self, x, node: Node):
        if node.is_leaf:
            return node.label

        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        return self._predict_sample(x, node.right)
    
    def predict(self, X):
        res = [self._predict_sample(X[i, :], self.root) for i in range(X.shape[0])]
        return np.array(res)

def accuracy(y, y_pred):
    return (y == y_pred).sum() / y.shape[0]

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score
    X, y = make_classification(n_samples=30, n_features=5, n_classes=2)
    model = DecisionTree(max_depth=10, minimum_sample_split=2)
    model.fit(X, y)
    y_pred = model.predict(X)
    print("Accuracy:", accuracy_score(y, y_pred))
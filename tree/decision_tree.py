import numpy as np

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        unique_classes = np.unique(y)

        # If only one class in the data or max depth reached, create a leaf node
        if len(unique_classes) == 1 or (self.max_depth is not None and depth == self.max_depth):
            return {'class': unique_classes[0]}

        # If no features left, create a leaf node with the majority class
        if num_features == 0:
            majority_class = np.argmax(np.bincount(y))
            return {'class': majority_class}

        # Find the best split
        best_feature, best_threshold = self._find_best_split(X, y)

        # Split the data
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = ~left_indices

        # Recursively build the left and right subtrees
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        # Return the current node
        return {
            'feature_index': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }

    def _find_best_split(self, X, y):
        num_samples, num_features = X.shape
        best_gini = float('inf')
        best_feature = None
        best_threshold = None

        for feature_index in range(num_features):
            unique_values = np.unique(X[:, feature_index])

            for threshold in unique_values:
                left_indices = X[:, feature_index] <= threshold
                right_indices = ~left_indices

                gini = self._calculate_gini(y[left_indices]) * len(y[left_indices]) / num_samples \
                    + self._calculate_gini(y[right_indices]) * len(y[right_indices]) / num_samples

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def _calculate_gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities**2)
        return gini

    def predict(self, X):
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x, node):
        if 'class' in node:
            return node['class']

        if x[node['feature_index']] <= node['threshold']:
            return self._predict_tree(x, node['left'])
        else:
            return self._predict_tree(x, node['right'])

if __name__ == "__main__":
  from sklearn.datasets import load_iris
  iris = load_iris()
  X, y = iris.data, iris.target
  
  
  # Initialize and train the decision tree classifier
  dt_classifier = DecisionTreeClassifier(max_depth=3)
  dt_classifier.fit(X, y)
  

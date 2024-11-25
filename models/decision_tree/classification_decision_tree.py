import numpy as np

class ClassificationDecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
    
    def fit(self, x, y):
        self.tree = self._build_tree(x, y)
    
    def _build_tree(self, x, y, depth=0):
        n_samples, n_features = x.shape
        unique_classes = np.unique(y)

        # Stopping conditions
        # First: All data points belong to the same class (pure leaf)
        if len(unique_classes) == 1:
            return {"label": unique_classes[0]}
        
        # Second: Reached max depth
        if self.max_depth is not None and depth >= self.max_depth:
            return {"label": self._most_common_class(y)}
        
        # Third: Not enough samples to split
        if n_samples < self.min_samples_split:
            return {"label": self._most_common_class(y)}
        
        # Find the best split
        best_split = self._best_split(x, y, n_features)
        left_tree = self._build_tree(*best_split['left'], depth+1)
        right_tree = self._build_tree(*best_split['right'], depth+1)

        return {
            "feature_index": best_split['feature_index'],
            "threshold": best_split['threshold'],
            'left': left_tree,
            'right': right_tree
        }
    
    def _best_split(self, x, y, n_features):
        best_gini = float('inf')
        best_split = None

        for feature_index in range(n_features):
            thresholds = np.unique(x[:, feature_index])

            for threshold in thresholds:
                left_mask = x[:, feature_index] <= threshold
                right_mask = ~left_mask

                left_y = y[left_mask]
                right_y = y[right_mask]

                gini = self._gini_impurity(left_y, right_y)

                if gini < best_gini:
                    best_gini = gini
                    best_split = {
                        "threshold": threshold,
                        'left': (x[left_mask], left_y),
                        'right': (x[right_mask], right_y),
                        'feature_index': feature_index
                    }
        return best_split

    def _gini_impurity(self, left_y, right_y):
        left_size = len(left_y)
        right_size = len(right_y)
        total_size = left_size + right_size

        def gini(y):
            _, counts = np.unique(y, return_counts=True)
            prob = counts / len(y)
            return 1 - sum(prob ** 2)
        
        left_gini = gini(left_y)
        right_gini = gini(right_y)

        return (left_size / total_size) * left_gini + (right_size / total_size) * right_gini 

    def _most_common_class(self, y):
        return np.bincount(y).argmax()
    
    def predict(self, X):
        return [self._predict_sample(x, self.tree) for x in X]
    
    def _predict_sample(self, x, tree):
        if 'label' in tree:
            return tree['label']
        
        feature_index = tree['feature_index']
        threshold = tree['threshold']

        if x[feature_index] <= threshold:
            return self._predict_sample(x, tree['left'])
        else:
            return self._predict_sample(x, tree['right'])

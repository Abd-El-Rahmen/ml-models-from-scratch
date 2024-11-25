import numpy as np
import sys
import os
sys.path.append(os.path.abspath('../'))
from decision_tree.classification_decision_tree import ClassificationDecisionTree
class ClassificationRandomForest :
    def __init__(self, n_estimators = 100 , max_depth = None , min_samples_split = 2 , max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
    
    def fit(self, X , y):
        self.trees = []
        n_samples , n_features = X.shape

        for _ in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(n_samples , size = n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            if self.max_features == 'sqrt':
                max_features = int(np.sqrt(n_features))
            elif isinstance(self.max_features, int):
                max_features = self.max_features
            else:
                max_features = n_features
            tree = ClassificationDecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_bootstrap,y_bootstrap)
            self.trees.append(tree)
    def predict(self,X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees] )   
        return np.apply_along_axis(lambda x : np.bincount(x).argmax() , axis=0 , arr=tree_preds)



            


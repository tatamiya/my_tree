import numpy as np
from .tree_base import MyTree
from .utils import select_majority


class MyForest():
    
    def __init__(self, n_estimators=10, max_features=None,
                 threshold_gini=0.05, min_node_size=5, max_depth=3,
                 verbose=False):
        
        self.n_estimators = n_estimators
        
        self.threshold_gini, self.min_node_size, self.max_depth = threshold_gini, min_node_size, max_depth
        self.verbose = verbose

        self.max_features = max_features # for RF
        
        self.n_classes = None
        self.list_trees = None
        
    
    def fit(self, X, y):
        
        self.n_classes = len(np.unique(y))
        self.list_trees = []
        length_data = len(y)
        
        for i in range(0, self.n_estimators):
            if self.verbose == True:
                print('=== {}th tree ==='.format(i))
    
            index_resample = np.random.choice(length_data, length_data, replace=True)
            X_resample, y_resample = X[index_resample], y[index_resample]

            a_tree = MyTree(splitter='random', max_features=self.max_features,
                            threshold_gini=self.threshold_gini, min_node_size=self.min_node_size, max_depth=self.max_depth,
                            verbose=self.verbose)
            
            a_tree.fit(X_resample, y_resample)
            self.list_trees.append(a_tree)

    
    def _pole(self, X, n_classes, list_trees):
        
        list_pred = []
        for a_tree in list_trees:
            list_pred.append(a_tree.predict(X))
        
        array_pred = np.array(list_pred)
        
        return np.apply_along_axis(func1d=np.bincount, axis=0, 
                                   arr=array_pred, minlength=n_classes)
    
    
    def predict(self, X):

        pole_result = self._pole(X, self.n_classes, self.list_trees)
        
        return pole_result.argmax(axis=0)
    
    
    def predict_proba(self, X):
        
        pole_result = self._pole(X, self.n_classes, self.list_trees)
        
        return (pole_result / self.n_estimators).T
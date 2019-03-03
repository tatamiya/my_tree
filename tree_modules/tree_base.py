import numpy as np


def gini(y):
    _, counts = np.unique(y, return_counts=True)

    prob = counts / len(y)
    
    return 1 - (prob * prob).sum()


class node_basis():
    
    def __init__(self, i_node, depth):
        
        self.i_node = i_node
        self.depth = depth

class node_internal(node_basis):
    
    def __init__(self, i_node, depth, i_feature, threshold):
        
        super().__init__(i_node, depth)
        self.i_feature = i_feature
        self.threshold = threshold
        
        self.node_child = {0:None, 1:None}
        

    def set_node_child(self, lr, i):
        self.node_child[lr] = i


class node_leaf(node_basis):
    
    def __init__(self, i_node, depth, k):
        
        super().__init__(i_node, depth)
        self.k_decided = k


class MyTree():
    
    
    def __init__(self, threshold_gini=0.05, min_node_size=5, max_depth=3,
                 splitter='best', max_features=None,
                 verbose=False):
        
        self.threshold_gini, self.min_node_size, self.max_depth = threshold_gini, min_node_size, max_depth
        self.i_node = None
        self.dict_nodes = None
        
        self.splitter = splitter # for RF
        self.max_features = max_features # for RF
        
        self.verbose = verbose
    
    
    def _find_optimal_division(self, x, y):
        list_gini = []
        x_unique = np.unique(x)

        for threshold in x_unique:

            mask_divide = x > threshold
            y_right = y[mask_divide]
            y_left = y[~mask_divide]

            gini_divide = (gini(y_right) * len(y_right) + gini(y_left) * len(y_left)) / len(y)

            list_gini.append(gini_divide)

        array_gini = np.array(list_gini)
        i_div_opt = np.argmin(array_gini)

        return x_unique[i_div_opt], array_gini[i_div_opt]


    def _divide(self, X, y):

        results = np.apply_along_axis(self._find_optimal_division, 0, X, y)

        arg_div = np.argmin(results[1])
        x_div = results[0, arg_div]

        return arg_div, x_div


    def _go_on_dividing(self, X, y, depth=0):

        depth += 1
        
        if self.splitter == 'best':
            X_chosen = X
            index_feat_chosen = np.arange(X.shape[1])
        
        elif self.splitter == 'random':
            
            n_features = X.shape[1]
            if self.max_features is None:
                num_feat_chosen = int(np.sqrt(n_features))
            elif isinstance(self.max_features, int) and self.max_features>0 and self.max_features <= n_features:
                num_feat_chosen = self.max_features
            elif isinstance(self.max_features, float) and self.max_features>0 and self.max_features<=1.0:
                num_feat_chosen = int(n_features * self.max_features)
            else:
                raise ValueError
                
            index_feat_chosen = np.random.choice(n_features, num_feat_chosen, replace=False)
            X_chosen = X[:, index_feat_chosen]
            
        else:
            raise ValueError

        arg_div_tmp, x_div = self._divide(X_chosen, y)
        arg_div = index_feat_chosen[arg_div_tmp] # inevitable in case of RF
        
        node_current = node_internal(self.i_node, depth, arg_div, x_div)
        self.dict_nodes[self.i_node] = node_current
        
        if self.verbose == True:
            print("=== node {} (depth {}): arg_div -> {}, x_div -> {} ===".format(self.i_node, depth, arg_div, x_div))

        mask = X[:, arg_div] > x_div
        X_right, X_left = X[mask], X[~mask]
        y_right, y_left = y[mask], y[~mask]

        gini_left = gini(y_left)
        gini_right = gini(y_right)

        list_divided = [(X_left, y_left, gini_left), (X_right, y_right, gini_right)]

        for lr, divided in enumerate(list_divided):
            self.i_node +=1

            X_i, y_i, gini_i = divided
            if gini_i > self.threshold_gini and len(y_i)>self.min_node_size and depth+1 <= self.max_depth:
                
                node_current.set_node_child(lr, self.i_node)
                self._go_on_dividing(X_i, y_i, depth=depth)
            else:
                node_current.set_node_child(lr, self.i_node)                
                feature_majority = np.bincount(y_i).argmax()
                
                node_terminal = node_leaf(self.i_node, depth, feature_majority)
                self.dict_nodes[self.i_node] = node_terminal
                

    def fit(self, X, y):
        
        self.i_node = 0
        self.dict_nodes = {}
        
        self._go_on_dividing(X, y)


    def _pred_each_vector(self, x):
        
        node_current = self.dict_nodes[0]
        while True:
            lr = int(x[node_current.i_feature] > node_current.threshold)
            node_next = self.dict_nodes[node_current.node_child[lr]]
            
            if node_next.__class__.__name__ == 'node_leaf':
                return node_next.k_decided
            else:
                node_current = node_next
    
    
    def predict(self, X):
        
        return np.apply_along_axis(self._pred_each_vector, 1, X)
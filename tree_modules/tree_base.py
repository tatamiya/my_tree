import numpy as np
from .utils import gini, select_majority


CRITERION = {'gini':gini, 'mse':np.var}
FUNC_NODE_VALUE = {'gini':select_majority, 'mse':np.mean}


def criterion_lr(y_left, y_right, func_criterion):
    
    return (func_criterion(y_left) * len(y_left) + func_criterion(y_right) * len(y_right) ) / (len(y_left) + len(y_right))


class Node():
    
    def __init__(self, i_node, depth, criterion, value):

        self.i_node, self.depth = i_node, depth
        self.criterion, self.value = criterion, value
        
        self.is_leaf = False
        
        self.division = {}
        self.child = {0:None, 1:None}
        
    def set_division(self, i_feature, threshold, criterion_diff):
        
        self.division['i_feature'] = i_feature
        self.division['threshold'] = threshold
        self.division['criterion_diff'] = criterion_diff


class MyTree():
    
    def __init__(self, minimum_criterion_diff=0.01, min_node_size=5, max_depth=5,
                 splitter='best', max_features=None,
                 criterion='gini',
                 verbose=False):
        
        self.minimum_criterion_diff, self.min_node_size, self.max_depth = minimum_criterion_diff, min_node_size, max_depth
        
        self.func_criterion = CRITERION[criterion]
        self.calc_node_value = FUNC_NODE_VALUE[criterion]
                
        self.splitter = splitter # for RF
        self.max_features = max_features # for RF
        
        self.verbose = verbose
        
        self.i_node = 0
        self.node_tree = None
    
    def _make_new_node(self, i_node, depth, y):

        criterion = self.func_criterion(y)
        node_value = self.calc_node_value(y)

        node_new = Node(i_node, depth, criterion, node_value)

        return node_new
    
    def _find_optimal_division(self, x, y):
        list_criterion = []
        x_unique = np.unique(x)

        if len(x_unique) == 1:
            return x_unique[0], self.func_criterion(y)

        for threshold in x_unique[:-1]:

            mask_divide = x > threshold
            y_left = y[mask_divide]
            y_right = y[~mask_divide]

            criterion_divide = criterion_lr(y_left, y_right, self.func_criterion)
            list_criterion.append(criterion_divide)

        array_criterion = np.array(list_criterion)
        i_div_opt = np.argmin(array_criterion)

        return x_unique[i_div_opt], array_criterion[i_div_opt]
    
    def _divide(self, X, y):

        results = np.apply_along_axis(self._find_optimal_division, 0, X, y)

        arg_div = np.argmin(results[1])
        x_div = results[0, arg_div]
        criterion_opt = results[1, arg_div]

        return arg_div, x_div, criterion_opt

    def _check_node_size(self, mask):

        sum_true = mask.sum()
        node_size_smaller = min(sum_true, len(mask) - sum_true)

        return node_size_smaller < self.min_node_size

    def _go_on_dividing(self, X, y, node):
        
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
        
        criterion_initial = node.criterion
        arg_div_tmp, x_div, criterion_optimized = self._divide(X_chosen, y)
        arg_div = index_feat_chosen[arg_div_tmp] # inevitable in case of RF

        mask = X[:, arg_div] > x_div
        X_right, X_left = X[mask], X[~mask]
        y_right, y_left = y[mask], y[~mask]

        criterion_diff = criterion_initial - criterion_optimized

        if criterion_diff < self.minimum_criterion_diff or self._check_node_size(mask):
            node.is_leaf = True
            
            if self.verbose == True:
                print("=== node {} (depth {}): LEAF, value -> {}, criterion -> {} ===".format(self.i_node, node.depth, node.value, criterion_initial))

        else:
            if self.verbose == True:
                print("=== node {} (depth {}): INTERNAL, arg_div -> {}, x_div -> {}, criterion_diff -> {} ===".format(self.i_node, node.depth, arg_div, x_div, criterion_diff))
            
            node.set_division(arg_div, x_div, criterion_diff)

            depth_next = node.depth + 1
            list_divided = [(X_left, y_left), (X_right, y_right)]
            for lr, divided in enumerate(list_divided):
                self.i_node += 1

                X_i, y_i = divided

                node_next = self._make_new_node(self.i_node, depth_next, y_i)
                node.child[lr] = node_next

                if depth_next == self.max_depth:
                    node_next.is_leaf = True
                    if self.verbose == True:
                        print("=== node {} (depth {}): LEAF, value -> {}, criterion -> {} ===".format(self.i_node, node.depth, node.value, criterion_initial))
                elif depth_next < self.max_depth:
                    self._go_on_dividing(X_i, y_i, node_next)

    def fit(self, X, y):
        
        self.i_node = 0
        self.node_tree = self._make_new_node(self.i_node, 0, y)
        
        self._go_on_dividing(X, y, self.node_tree)
                
    def _pred_each_vector(self, x):

        node_current = self.node_tree

        while node_current.is_leaf == False:
            division = node_current.division
            lr = int(x[division['i_feature']] > division['threshold'])
            node_current = node_current.child[lr]

        return node_current.value
    
    def predict(self, X):
        
        return np.apply_along_axis(self._pred_each_vector, 1, X)

import numpy as np


def gini(y):
    _, counts = np.unique(y, return_counts=True)

    prob = counts / len(y)
    
    return 1 - (prob * prob).sum()


def find_optimal_division(x, y):
    list_gini = []
    x_unique = np.unique(x)

    for threshold in x_unique:

        mask_divide = x > threshold
        y_upper = y[mask_divide]
        y_lower = y[~mask_divide]

        gini_divide = (gini(y_upper) * len(y_upper) + gini(y_lower) * len(y_lower)) / len(y)

        list_gini.append(gini_divide)
        
    array_gini = np.array(list_gini)
    i_div_opt = np.argmin(array_gini)
    
    return x_unique[i_div_opt], array_gini[i_div_opt]


def divide_tree(X, y):

    results = np.apply_along_axis(find_optimal_division, 0, X, y)

    arg_div = np.argmin(results[1])
    x_div = results[0, arg_div]

    return arg_div, x_div


def go_on_dividing(X, y, depth=0, div_set=None,
                   threshold_gini=0.05, min_node_size=5, max_depth=3):
    
    global i
    if div_set is None:
        div_set = []
        
    depth += 1
    
    arg_div, x_div = divide_tree(X, y)
    
    print("=== node {} (depth {}): arg_div -> {}, x_div -> {} ===".format(i, depth, arg_div, x_div))
    
    mask = X[:, arg_div] > x_div
    X_upper, X_lower = X[mask], X[~mask]
    y_upper, y_lower = y[mask], y[~mask]
    
    gini_lower = gini(y_lower)
    gini_upper = gini(y_upper)
    
    list_divided = [(X_lower, y_lower, gini_lower), (X_upper, y_upper, gini_upper)]
    
    for ul, divided in enumerate(list_divided):
        i +=1
        div_set_tmp = div_set.copy()
        div_set_tmp.append((depth, i, arg_div, x_div, ul))
        
        X_i, y_i, gini_i = divided
        if gini_i > threshold_gini and len(y_i)>min_node_size and depth+1 <= max_depth:
            go_on_dividing(X_i, y_i, depth=depth, div_set=div_set_tmp)
        else:
            # なんらかの停止処理
            print(div_set_tmp, np.bincount(np.array(y_i)).argmax())
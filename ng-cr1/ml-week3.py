import numpy as np

x_train = np.array([0., 1, 2, 3, 4, 5],dtype=np.longdouble)
y_train = np.array([0,  0, 0, 1, 1, 1],dtype=np.longdouble)

def sigmoid(x):

    x_exp = np.exp(-x)
    g = 1/(1+x_exp)

    return g


def compute_cost_logistic(x, y, w, b, sigmoid):

    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb = np.dot(x[i],w) + b
        f_wb = sigmoid(f_wb)
        print(f_wb)
        if y[i] == 1:
            cost += -np.log(f_wb)
        else:
            cost += -np.log(1-f_wb)
    cost = cost/m
    return cost

print(compute_cost_logistic(x_train, y_train, 1, -3, sigmoid))
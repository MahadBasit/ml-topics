import numpy as np

x_train = np.array([0., 1, 2, 3, 4, 5],dtype=np.longdouble)
y_train = np.array([0,  0, 0, 1, 1, 1],dtype=np.longdouble)

def sigmoid(x):

    x_exp = np.exp(-x)
    g = 1/(1+x_exp)

    return g


def compute_cost_logistic(x, y, w, b, sigmoid):

    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(x[i],w) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i]*(np.log(f_wb_i)) - (1-y[i])*(np.log(1-f_wb_i))
    cost = cost/m
    return cost

print(compute_cost_logistic(x_train, y_train, 1, -3, sigmoid))
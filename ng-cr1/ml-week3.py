import numpy as np

X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  #(m,n)
y_train = np.array([0, 0, 0, 1, 1, 1])                                           #(m,)


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


def compute_gradient(x, y, w, b):
    m,n = x.shape

    dj_dw = np.zeros(n)
    dj_db = 0.0

    for i in range(m):
        z_i = np.dot(x[i], w) + b
        f_wb_i = sigmoid(z_i)
        err = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err*x[i,j]
        dj_db += err

    dj_dw = dj_dw/m
    dj_db = dj_db/m

    return dj_dw, dj_db

X_tmp = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_tmp = np.array([0, 0, 0, 1, 1, 1])
w_tmp = np.array([2.,3.])
b_tmp = 1.
dj_dw_tmp, dj_db_tmp = compute_gradient(X_tmp, y_tmp, w_tmp, b_tmp)
print(f"dj_db: {dj_db_tmp}" )
print(f"dj_dw: {dj_dw_tmp.tolist()}" )
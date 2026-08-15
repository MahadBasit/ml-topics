import numpy as np
import math

X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

def sigmoid(x):
    x_exp = np.exp(-x)
    g = 1/(1+x_exp)

    return g

def compute_cost_logistics(x, y, w, b):
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(x[i],w) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i]*(np.log(f_wb_i)) - (1-y[i])*(np.log(1-f_wb_i))
    cost = cost/m
    return cost

def compute_gradient_logistics(x, y, w, b):
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

def gradient_descent_logistics(x, y, w_in, b_in, num_iters, alpha):
    
    J_hist = []
    w = w_in
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient_logistics(x,y,w,b)
        w = w - alpha*dj_dw
        b = b - alpha*dj_db

        if i < 10000:
            J_hist.append(compute_cost_logistics(x,y,w,b))

        if i % math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4d}: Cost {J_hist[-1]:8.2f}")

    return w, b, J_hist

w_tmp  = np.zeros_like(X_train[0])
b_tmp  = 0.
alph = 0.1
iters = 10000

w_out, b_out, _ = gradient_descent_logistics(X_train, y_train, w_tmp, b_tmp,iters, alph) 
print(f"\nupdated parameters: w:{w_out}, b:{b_out}")
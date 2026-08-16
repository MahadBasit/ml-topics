import numpy as np
import math

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

def compute_cost(X, y, w, b, lmda): 
   
    m = X.shape[0]
    cost = 0.0
    for i in range(m):                                
        f_wb_i = np.dot(X[i], w) + b           
        cost = cost + (f_wb_i - y[i])**2
    cost = cost + lmda*np.dot(w,w)      
    cost = cost / (2 * m)                      
    return cost


def compute_gradient(X, y, w, b, lmda):

    m,n = X.shape
    dj_dw = np.zeros((n))
    dj_db = 0.

    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        err = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err*X[i, j]
        dj_db += err

    for j in range(n):
        dj_dw[j] += lmda*w[j]

    dj_dw = dj_dw/m
    dj_db = dj_db/m 

    return dj_dw, dj_db


def gradient_descent(X, y, w_in, b_in, num_iters, alpha, lmda):
    J_hist = []

    w = w_in
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b, lmda)
        w = w - alpha*dj_dw
        b = b - alpha*dj_db

        if i < 100000:
            J_hist.append(compute_cost(X, y, w, b, lmda))

        if i % math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4d}: Cost {J_hist[-1]:8.2f}   ")

    return w, b, J_hist


def zscore_normalize_features(X):

    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X = (X - mu)/sigma

    return X, mu, sigma

X_scaled, mu, sig = (zscore_normalize_features(X_train))


# initialize parameters
initial_w = np.zeros_like(w_init)
initial_b = 0.
# some gradient descent settings
iterations = 500
alpha = 0.1
# run gradient descent 
w_final, b_final, J_hist = gradient_descent(X_scaled, y_train, initial_w, initial_b, iterations, alpha, lmda=0)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = X_train.shape
for i in range(m):
    print(f"prediction: {np.dot(X_scaled[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")
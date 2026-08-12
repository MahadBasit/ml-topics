import numpy as np
import math

def compute_model_output(x,w,b):
    m = x.shape[0]
    f_w_b = np.zeros(m)
    for i in range(m):
        f_w_b[i] = w*x[i] + b

    return f_w_b

def cost_function(x,y,w,b):
    m = x.shape[0]

    cost_sum = 0
    for i in range(m):
        cost = ((w*x[i] + b) - y[i])**2
        cost_sum = cost_sum + cost

    total_cost = (1/(2*m))*cost_sum

    return total_cost

def gradient_function(x, y, w, b):
    m = x.shape[0]

    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w*x[i] + b
        dj_dw_i = (f_wb - y[i])*x[i]
        dj_db_i = f_wb - y[i]
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    dj_dw = dj_dw/m
    dj_db = dj_db/m

    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    J_hist = []
    p_hist = []

    w = w_in
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)
        w = w - alpha*dj_dw
        b = b - alpha*dj_db
        if i < 100000:
            J_hist.append(cost_function(x,y,w,b))
            p_hist.append([w,b])

        if i % math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_hist[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")

    return w, b, J_hist, p_hist
import numpy as np

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
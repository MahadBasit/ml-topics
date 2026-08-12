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
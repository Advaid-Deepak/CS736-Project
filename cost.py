import numpy as np

def J_fun(u,y,c,q):
    return np.sum(np.power(u,q)@np.square(y-c.T))
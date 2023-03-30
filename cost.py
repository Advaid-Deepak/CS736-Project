import numpy as np

def J_fun(u,y,c,q):
    return np.sum(np.power(u,q).T@np.square(y-c.T))

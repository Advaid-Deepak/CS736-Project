import numpy as np

def class_means(u,y,q) :
    c = np.power(u,q)@y/np.sum(np.power(u,q),axis = 1)
    return c
import numpy as np

def class_means(memberships,pixels,q) :

    powered_Membership = memberships ** q
    c = powered_Membership.T@pixels
    c = c.T/np.sum(powered_Membership,axis = 0)
    return c.T

"""
Provides some standard sampling functions
"""

import numpy as np
import numpy.linalg as alg

def uniform_sphere(d, gen):
    """Samples uniformly from the (d-1)-sphere
        
        Args:
            d (int): dimension of space containing the sphere
            gen: instance of rnd.Generator to be used for pseudo-random number
                generation

        Returns:
            Z (float): size d np array containing a sample from the 
                uniform distribution on the (d-1)-sphere
    """

    Z = gen.normal(size=d)
    return Z/alg.norm(Z)

def uniform_great_subsphere(p, gen):
    """Samples uniformly from the great subsphere of the (d-1)-sphere w.r.t.
        the pole p
    
        Args:
            p (array): pole, should be 1d numpy array of norm 1
            gen: instance of rnd.Generator to be used for pseudo-random number
                generation

        Returns:
            y (array): 1d numpy array of same size as p, is a sample from 
                the great subsphere w.r.t. p
    """

    Z = gen.normal(size=p.shape[0])
    Z = Z - np.inner(Z, p) * p
    return Z/alg.norm(Z)

def uniform_ball(d, b, gen):
    """Samples uniformly from d-dimensional zero-centered Euclidean balls
    
        Args:
            d (int): dimension of space containing the ball
            b (float): radius of the ball
            gen: instance of rnd.Generator to be used for pseudo-random number
                generation    """

    thetas = uniform_sphere(d)
    rs = b * gen.uniform()**(1/d)
    return rs * thetas


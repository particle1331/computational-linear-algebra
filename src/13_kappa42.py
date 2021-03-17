import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from utils import plot_svd_scree
np.random.seed(0)

'''
Code challenge: Generate random matrix with condition no. 42
'''


def generate_cond_42(n):
    # generate random matrix; get svd
    A = np.random.randn(n, n)
    U, s, VT = np.linalg.svd(A)

    # rescale: f(σ) = aσ + b, such that f(σ_0) = 42, and f(σ_n) = 1 
    a = -41.0 / (s[-1] - s[0])
    b = 42 - a * s[0]
    s = a * s + b

    # new random matrix with cond #. k = 42
    return U @ np.diag(s) @ VT


if __name__ == '__main__':
    A = generate_cond_42(20)
    
    # plot svd decomposition
    plot_svd_scree(A, save=True, filename='13_kappa=42.png')


    

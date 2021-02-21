import numpy as np 
import matplotlib.pyplot as plt


def cgs(A):
    '''Compute the classical Gram-Schmidt of the matrix A.'''
    
    U = np.zeros_like(A)
    U[:, 0] = A[:, 0] / np.linalg.norm(A[:, 0])
    for k in range(1, A.shape[1]):
        u = A[:, k] - U[:, :k] @ U[:, :k].T @ A[:, k]
        U[:, k] = u / np.linalg.norm(u)
    return U

def mgs(A):
    '''Compute the modified Gram-Schmidt of the matrix A.'''
    
    U = np.copy(A)
    for k in range(A.shape[1]):
        U[:, k] /= np.linalg.norm(U[:, k])
        for j in range(k+1, A.shape[1]):
            U[:, j] -= U[:, k] * ( U[:, k].T @ U[:, j] ) 
    return U

np.random.seed(4)
eps = lambda : 1e-8*np.random.randn()
A = np.array([
        [1,     1,      1    ],
        [eps(), 0,      0    ],
        [0,     eps(),  0    ],
        [0,     0,      eps()]
    ])
n = A.shape[1]

U = cgs(A)
print('L1 error (classical GS) =', np.abs((U.T @ U) - np.eye(n)).sum() )

U = mgs(A)
print('L1 error (modified GS) =', np.abs((U.T @ U) - np.eye(n)).sum() )

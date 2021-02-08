import numpy as np
import matplotlib.pyplot as plt


n = 1000           # sparse matrix
A = np.zeros([n, n]) 

for i in range(n):
    j = np.random.randint(n)
    A[i, j] += np.random.randn()

A += 0.01 * np.eye(n)

print('A sparsity:\t', np.count_nonzero(A) / (n*n) )
print('A_inv sparsity:\t', np.count_nonzero(np.linalg.inv(A)) / (n*n) )
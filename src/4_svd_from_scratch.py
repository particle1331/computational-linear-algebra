import numpy as np 

# calculate matrix V from eigenvectors of A^T A
np.random.seed(0)
A = np.random.randn(5, 4)
B = A.T @ A # shape 4 x 4

# compute eigenvalues / vectors, sorted decreasing
eigvals, eigvecs = np.linalg.eig(B)
eigvecs = eigvecs[:, np.argsort(-eigvals)]
eigvals = -np.sort(-eigvals)

# compact SVD -- too lazy to code Gram-Schmidt completion
Sigma = np.zeros_like(A)
U = np.zeros(shape=(A.shape[0], A.shape[0]))
V = eigvecs
for i in range(len(eigvals)):
    Sigma[i, i] = np.sqrt(eigvals[i])
    U[:, i] = A @ eigvecs[:, i] / np.sqrt(eigvals[i])

print('U @ Sigma @ V.T =') 
print( U @ Sigma @ V.T )
print('\nA=')
print(A)
print('\nL1 error =', np.abs(U @ Sigma @ V.T - A).sum() )
print('\nU.T @ U =')
print( U.T @ U )

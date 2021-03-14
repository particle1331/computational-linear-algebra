import numpy as np 

# calculate matrix V from eigenvectors of A^T A
np.random.seed(0)
A = np.random.randn(5, 4)
B = A.T @ A # shape 4 x 4

# compute eigenvalues / vectors, sorted decreasing
eigvals, eigvecs = np.linalg.eig(B)
eigvecs = eigvecs[:, np.argsort(-eigvals)]
eigvals = -np.sort(-eigvals)

# compact SVD -- too lazy to code Gram-Schmidt
# UPDATE: see 10_stability_gram-schmidt.py
Sigma = np.zeros(shape=(A.shape[1], A.shape[1]))
U = np.zeros(shape=(A.shape[0], A.shape[1]))
V = eigvecs[: A.shape[1]]
for i in range(len(eigvals)):
    Sigma[i, i] = np.sqrt(eigvals[i])
    U[:, i] = A @ eigvecs[:, i] / np.sqrt(eigvals[i])

# print the results
print('\nA=')
print(np.round(A, 4))
print('\nU @ Sigma @ V.T =') 
print(np.round(A, 4))
print('\nFrobenius norms:')
print('|| A - U @ Sigma @ V.T || =', np.linalg.norm(U @ Sigma @ V.T - A))
print('|| V.T @ V - I || =', np.linalg.norm(V.T @ V - np.eye(4)))
print('|| U.T @ U - I || =', np.linalg.norm(U.T @ U - np.eye(4)))
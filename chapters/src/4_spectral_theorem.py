import numpy as np 
np.random.seed(0)

# sample a random symmetric matrix
A = np.random.randn(3, 3)
A = A.T @ A  

# calculate an eigenvalue-eigenevctor pair using numpy
eigvals, eigvecs = np.linalg.eig(A)
lam = eigvals[0]
v = eigvecs[:, 0]       # already normalized; defines a 1-d subspace

# now we want to find a basis Y for the subspace orthogonal to v, i.e. (x, y, z) @ v = 0. 
x1 = -(1*v[1] + 1*v[2]) / v[0] 
y1 = np.array([x1, 1, 1], dtype=np.float) # y = z = 1

x2 = -(1*v[1] + 0*v[2]) / v[0] 
y2 = np.array([x2, 1, 0], dtype=np.float) # y = 1, z = 0

# make y1 and y2 orthonormal (Gram-Schmidt)
y1 = y1 / np.linalg.norm(y1)
y2 = y2 - y2.dot(y1) * y1 
y2 = y2 / np.linalg.norm(y2)
# print( y1.T @ y2 ) # -3.33e-16
# print( np.linalg.norm(y1), np.linalg.norm(y2) )  # 1., 1.

# make the B matrix -> orthogonal eigendecomposition
Y = np.stack([y1, y2], axis=1).reshape(-1, 2)
B = Y.T @ A @ Y 
omega, U = np.linalg.eig(B)
W = Y @ U # shape: (n, n-1)

# put together eigenvalues and eigenvectors of A
Lambda = np.array([lam] + list(omega))
V = np.stack([v] + [W[:, i] for i in range(2)], axis=1)
V = V[:, np.argsort(-Lambda)]
Lambda = np.diag(-np.sort(-Lambda))

# print results
print('A =')
print(A)
print('\nB =')
print(B) # symmetric?
print('\nV =')
print(V) # symmetric?
print('\nV.T @ V =')
print(V.T @ V)
print('\nLambda (eigenvalues) =', Lambda.diagonal())
print('L1 error (A, V @ Lambda @ V.T) =', np.abs(A - V @ Lambda @ V.T).sum())
print('\nCompare with np.linalg.eig(A):')
print(np.linalg.eig(A)[0])
print(np.linalg.eig(A)[1])
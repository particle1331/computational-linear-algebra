import numpy as np


rng = np.random.RandomState(0)
A = rng.randn(2, 2)
u, s, vT = np.linalg.svd(A)

# check eigenvalues of sqrt(A^T A) = singular values A
print('eigvals of sqrt(A^T A):\t', np.sqrt(np.linalg.eig( A.T @ A )[0]))
print('singular values of A:\t', s)

# check that maximum singular value = maximum dilation of circle
N = 1000 # <--- no. of points on the unit circle
t = np.linspace(-1, 1, N)
unit_circle = np.stack([np.cos(2*np.pi*t), np.sin(2*np.pi*t)], axis=0)
outputs = A @ unit_circle
max_output_norm = np.linalg.norm(A @ unit_circle, axis=0).max()
print('max norm - max s.value:\t', abs(max_output_norm - s[0]))

# check that Av = \sigma u
print('|| Av - su ||.max():\t', np.abs(u @ np.diag(s) - A @ vT.T).max()) 
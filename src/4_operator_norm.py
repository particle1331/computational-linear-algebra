import numpy as np

t = np.linspace(-1, 1, 11123)
unit_circle = np.stack([np.cos(2*np.pi*t), np.sin(2*np.pi*t)], axis=0)
A = np.random.randn(2, 2)
op_norm = max([np.linalg.norm(A @ unit_circle[:, i]) for i in range(100)])
op_norm_np = np.linalg.norm(A, 2)
u, s, vT = np.linalg.svd(A)

print("approx:\t", op_norm)
print("numpy:\t", op_norm_np)
print("svd:\t", s[0])
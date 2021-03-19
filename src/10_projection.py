import numpy as np
import matplotlib.pyplot as plt


# initialize A with C(A) plane in R^3.
np.random.seed(18)
A = np.random.randn(3, 2)
c = np.cross(A[:, 0], A[:, 1])  # normal vector to C(A)

# compute vectors
b = np.array([0.9, 0.8, 0.2])    # test vector
x = np.linalg.solve( A.T @ A, A.T @ b )
Ax = A @ x  # projection of b on C(A)

# plot surface
fig = plt.figure(figsize=plt.figaspect(1))
ax = fig.gca(projection='3d')
xx, yy = np.meshgrid(np.linspace(-2, 2), np.linspace(-2, 2))
z = -( c[0] * xx + c[1] * yy ) / c[2]
ax.plot_surface(xx, yy, z, alpha=0.4)

# plot vectors
ax.plot([0,b[0]], 
        [0, b[1]], 
        [0, b[2]], 'r', label='b (test)')

ax.plot([0, Ax[0]], 
        [0, Ax[1]], 
        [0, Ax[2]], 'b', label='Ax (proj)')

ax.plot([Ax[0], b[0]], 
        [Ax[1], b[1]], 
        [Ax[2], b[2]], 'g', label='Ax - b (perp)')

print('(Ax - b) @ Ax =', (Ax - b) @ Ax )
plt.legend()
plt.tight_layout()
plt.savefig('../img/10_projection.png')
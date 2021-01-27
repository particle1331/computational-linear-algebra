import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


errors = []
for j in [2, 4, 5, 6]:
    N = int(10*j) # <--- no. of points on the unit circle
    t = np.linspace(-1, 1, N)
    unit_circle = np.stack([np.cos(2*np.pi*t), np.sin(2*np.pi*t)], axis=0)
    A = np.random.randn(2, 2)
    outputs = A @ unit_circle
    op_norm = np.linalg.norm(A @ unit_circle, axis=0).max()
    op_norm_np = np.linalg.norm(A, 2)
    errors.append(abs(op_norm - op_norm_np))

# plot errors
ax = plt.figure().gca()
ax.plot([10, 100, 1000, 10000], errors)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_ylabel('approx. error')
ax.set_xlabel('no. of unit circle points')

# save
plt.savefig('../img/4_operator_norm.png')
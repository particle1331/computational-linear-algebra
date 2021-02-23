from scipy.linalg import solve_triangular
import numpy as np 
import time
import matplotlib.pyplot as plt
np.random.seed(0)


inv_times = []
tri_times = []
pow_range = range(1, 12)
for j in pow_range:
    n = 2**j
    A = np.random.randn(n, n)
    
    # square matrix
    start_time = time.time()
    np.linalg.inv(A)
    inv_times.append(time.time() - start_time)

    # triangular
    A = np.triu(A)
    start_time = time.time()
    solve_triangular(A, np.eye(n), lower=False)
    tri_times.append(time.time() - start_time)


# plot
fig, ax = plt.subplots(1, 1, figsize=(6, 5))

plt.scatter(pow_range, inv_times, marker='o', edgecolors='k', label='np.linalg.inv')
plt.scatter(pow_range, tri_times, marker='o', edgecolors='k', label='sp.linalg.solve_triangular')
ax.plot(pow_range, inv_times, linestyle='--', linewidth=0.8)
ax.plot(pow_range, tri_times, linestyle='--', linewidth=0.8)

ax.set_xticks(pow_range)
ax.set_xticklabels([2**j for j in pow_range])
ax.set_ylabel('wall time (s)')
ax.set_xlabel(f'matrix size $n$')
fig.tight_layout()
plt.legend()

plt.savefig('../img/10_solve_triangular.png')
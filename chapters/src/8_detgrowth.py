import numpy as np 
import matplotlib.pyplot as plt


# sample dets of shifted matrix A with det A = 0
n = 20
sample_size = 10000
lambdas = np.linspace(0, 1, 30)
dets = []
for i in range(30):
    lam = lambdas[i]
    tmp = []
    for j in range(sample_size):
        A = np.random.randn(n, n)   # sample
        A[:, 0] = A[:, 1]           # make dependent!
        tmp.append(
            abs(np.linalg.det(A + lam * np.eye(n)))
        )
    dets.append(np.mean(tmp))

# plot
plt.scatter(lambdas, dets, marker='o', facecolors='none', edgecolors='k')
plt.plot(lambdas, dets, linestyle='--', c='red', linewidth=0.8)
plt.ylabel('avg. det size')
plt.xlabel(f'$\lambda$')
plt.savefig('../img/8_detgrowth.png')
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


if __name__ == '__main__':
    S = np.array([[1, 2], [2, 4]])
    T = np.array([[1, 2], [3, 4]])
    matrices = [S, T]

    fig, ax = plt.subplots(1, 2, figsize=(8, 6))
    for i in [0, 1]:
        A = matrices[i]
        # circle
        t = np.linspace(-1, 1, 100)

        # transform unit circle
        Ax = A[0, :] @ np.stack([np.cos(2*np.pi*t), np.sin(2*np.pi*t)], axis=0)
        Ay = A[1, :] @ np.stack([np.cos(2*np.pi*t), np.sin(2*np.pi*t)], axis=0)
        ax[i].scatter(np.cos(2*np.pi*t), np.sin(2*np.pi*t), s=0.7)
        ax[i].scatter(Ax, Ay, s=0.7)
        ax[i].axis('scaled')
        ax[i].set_ylim(-6, 6)
        ax[i].set_xlim(-4, 4)

        # obtain U and P from SVD
        u, s, vT = np.linalg.svd(A)
        Q = u @ vT
        P = vT.T @ np.diag(s) @ vT

        # get eigenvectors -> scale by singular values -> rotate -> plot
        eig_vals = s
        eig_vecs = vT.T
        ax[i].arrow(0, 0, eig_vals[0]*(Q @ eig_vecs[:, 0])[0], eig_vals[0]*(Q @ eig_vecs[:, 0])[1]) # rotate v1
        ax[i].arrow(0, 0, eig_vals[1]*(Q @ eig_vecs[:, 1])[0], eig_vals[1]*(Q @ eig_vecs[:, 1])[1]) # rotate v2
        ax[i].set_title(f'A={A.tolist()}')

    plt.savefig('../img/4_polar_decomposition.png')
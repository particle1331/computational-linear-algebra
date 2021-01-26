import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


def plot_circle_transformation(A, ax, use_svd=False):
    # circle
    t = np.linspace(-1, 1, 100)

    # transform unit circle
    Ax = A[0, :] @ np.stack([np.cos(2*np.pi*t), np.sin(2*np.pi*t)], axis=0)
    Ay = A[1, :] @ np.stack([np.cos(2*np.pi*t), np.sin(2*np.pi*t)], axis=0)
    ax.scatter(np.cos(2*np.pi*t), np.sin(2*np.pi*t), s=0.7)
    ax.scatter(Ax, Ay, s=0.7)
    ax.axis('scaled')
    ax.set_ylim(-6, 6)
    ax.set_xlim(-4, 4)

    if use_svd==False:
        # plot eigenvectors from polar decomposition
        U, P = linalg.polar(A, side='right')
    else:
        # obtain U and P from SVD
        u, s, vT = np.linalg.svd(A)
        U = u @ vT
        P = vT.T @ np.diag(s) @ vT
    
    # get eigenvectors -> scale by eigenvals -> rotate -> plot
    eig_vals = np.linalg.eig(P)[0]
    eig_vecs = np.linalg.eig(P)[1]
    ax.arrow(0, 0, eig_vals[0]*(U @ eig_vecs[:, 0])[0], eig_vals[0]*(U @ eig_vecs[:, 0])[1]) # rotate v1
    ax.arrow(0, 0, eig_vals[1]*(U @ eig_vecs[:, 1])[0], eig_vals[1]*(U @ eig_vecs[:, 1])[1]) # rotate v2
    return U, P


if __name__ == '__main__':
    use_svd = False
    T = np.array([[1, 2], [3, 4]]) # nonsingular
    S = np.array([[1, 2], [2, 4]]) # singular
    matrices = [T, S]
    
    # plot unit circle and its transformation
    fig, ax = plt.subplots(1, 2, figsize=(8, 6))
    for i in [0, 1]:
        A = matrices[i]
        U, P = plot_circle_transformation(A, ax[i], use_svd=use_svd)
        ax[i].set_title(f'A={A.tolist()}')

    if use_svd:
        plt.savefig('../img/4_polar_decomposition-svd.png')
    else:
        plt.savefig('../img/4_polar_decomposition.png')

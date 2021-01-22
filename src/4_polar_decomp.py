import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
 
# circle
t = np.linspace(-1, 1, 100)
T = np.array([[1, 2], [3, 4]]) # nonsingular
S = np.array([[1, 2], [2, 4]]) # singular
matrices = [T, S]

# plot singular and nonsingular transformation
fig, ax = plt.subplots(1, 2, figsize=(8, 6))

for i in [0, 1]:
    # get matrix
    A = matrices[i]

    # transform unit circle
    Ax = A[0, :] @ np.stack([np.cos(2*np.pi*t), np.sin(2*np.pi*t)], axis=0)
    Ay = A[1, :] @ np.stack([np.cos(2*np.pi*t), np.sin(2*np.pi*t)], axis=0)
    ax[i].scatter(np.cos(2*np.pi*t), np.sin(2*np.pi*t), s=0.7)
    ax[i].scatter(Ax, Ay, s=0.7)
    ax[i].axis('scaled')
    ax[i].set_ylim(-6, 6)
    ax[i].set_xlim(-4, 4)

    # plot eigenvectors from polar decomposition
    U, P = linalg.polar(A, side='right')
    # W, S, V = np.linalg.svd(A)
    # U = W @ V.T
    # P = V @ np.diag(S) @ V.T
    
    # get eigenvectors -> scale by eigenvals -> rotate -> plot
    eig_vals = np.linalg.eig(P)[0]
    eig_vecs = np.linalg.eig(P)[1]
    ax[i].arrow(0, 0, eig_vals[0]*(U @ eig_vecs[:, 0])[0], eig_vals[0]*(U @ eig_vecs[:, 0])[1]) # rotate v1
    ax[i].arrow(0, 0, eig_vals[1]*(U @ eig_vecs[:, 1])[0], eig_vals[1]*(U @ eig_vecs[:, 1])[1]) # rotate v2


ax[0].set_title('A=[[1, 2], [3, 4]]')
ax[1].set_title('A=[[1, 2], [2, 4]]')

# save
plt.savefig('../img/4_polar_decomposition.png')

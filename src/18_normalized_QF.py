import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d 
np.random.seed(0)


def QF(x, Q):
    '''Quadratic form of a matrix Q.'''
    return (x.T @ Q @ x) / (x.T @ x)


def plot_QF_surface(Q, fn=''):
    # symmetrize
    Q = 0.5*(Q.T + Q)
    
    fig = plt.figure(figsize=(6, 3))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)

    N = 400 # size of grid
    row = np.linspace(-5, 5, N)
    XX, YY = np.meshgrid(row, row)
    ZZ = np.zeros_like(XX)
    for i in range(N):
        for j in range(N):
            ZZ[i, j] = QF(np.array([XX[i, j], YY[i, j]]), Q)

    # plot 3D surface 
    ax1.plot_surface(XX, YY, ZZ, cmap='Reds') 
    ax1.set_xlabel(f'$w_0$')
    ax1.set_ylabel(f'$w_1$')

    # plot contour and principal axes
    ax2.contourf(XX, YY, ZZ, cmap='Reds', levels=30)
    ax2.set_xlabel(f'$w_0$')
    ax2.set_ylabel(f'$w_1$')

    # compute eigenvectors
    eigvals, eigvecs = np.linalg.eig(Q)
    idx = np.argsort(-eigvals) # sort decreasing
    eigvecs = eigvecs[:, idx]
    eigvals = eigvals[idx]

    # plot
    eigvec0 = ax2.arrow(0, 0, eigvals[0]*eigvecs[0, 0], eigvals[0]*eigvecs[1, 0], color='green', label=f'$\lambda_0 u_0$')
    eigvec1 = ax2.arrow(0, 0, eigvals[1]*eigvecs[0, 1], eigvals[1]*eigvecs[1, 1], color='blue',  label=f'$\lambda_1 u_1$')
    ax2.legend([eigvec0, eigvec1], [f'$\lambda_0 u_0$', f'$\lambda_1 u_1$'])
    ax2.set_title(f'$\lambda_0$={eigvals[0]:.2f}, $\lambda_1$={eigvals[1]:.2f}', fontsize=10)


    fig.tight_layout()
    plt.savefig(fn)
    # plt.show() # can move 3d plot around



if __name__ == '__main__':
    Q = np.array([
        [0, 1],
        [2, 0]
    ])
    plot_QF_surface(Q, fn='../img/18_normalized_indefiniteQF.png')
    
    Q = np.array([
        [1, 2],
        [1, 2]
    ])
    plot_QF_surface(Q, fn='../img/18_normalized_semidefiniteQF.png')
    
    Q = np.array([
        [4, 4],
        [4, 9]
    ])
    plot_QF_surface(Q, fn='../img/18_normalized_definiteQF.png')
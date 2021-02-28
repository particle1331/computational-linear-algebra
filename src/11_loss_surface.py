import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d 
np.random.seed(0)


def loss(w, X, y):
    '''linear least square loss'''
    return ((X @ w - y)**2).mean()


def plot_loss_surface(independent):
    # generate noisy sample data
    n = 1000
    X = np.zeros((n, 2))
    X[:, 1] = np.random.uniform(low=-1, high=1, size=n)
    if independent:
        X[:, 0] = 1
    else: 
        X[:, 0] = 2*X[:, 1]

    w_true = np.array([-1, 3])
    y = (X @ w_true) + 0.01 * np.random.randn(n)      # signal: y = -1 + 3x + noise


    # plot loss surface
    fig, ax = plt.subplots(2, 1, figsize=(3, 3))
    N = 200 # size of grid
    row = np.linspace(-5, 5, N)
    XX, YY = np.meshgrid(row, row)
    ZZ = np.zeros_like(XX)
    for i in range(N):
        for j in range(N):
            ZZ[i, j] = loss(np.array([XX[i, j], YY[i, j]]), X, y)


    # plot 3D surface
    ax = plt.axes(projection ='3d') 
    ax.plot_surface(XX, YY, ZZ, cmap='Reds') 
    ax.set_xlabel(f'$w_0$')
    ax.set_ylabel(f'$w_1$')
    fig.tight_layout()

    # save
    if independent:
        plt.savefig('../img/11_loss_independent.png')
    else:
        plt.savefig('../img/11_loss_dependent.png')


if __name__ == '__main__':
    plot_loss_surface(independent=True)
    plot_loss_surface(independent=False)
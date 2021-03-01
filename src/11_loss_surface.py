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
    fig = plt.figure(figsize=(6, 3))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)

    N = 400 # size of grid
    row = np.linspace(-5, 5, N)
    XX, YY = np.meshgrid(row, row)
    ZZ = np.zeros_like(XX)
    for i in range(N):
        for j in range(N):
            ZZ[i, j] = loss(np.array([XX[i, j], YY[i, j]]), X, y)


    # plot 3D surface 
    ax1.plot_surface(XX, YY, ZZ, cmap='Reds') 
    ax1.set_xlabel(f'$w_0$')
    ax1.set_ylabel(f'$w_1$')

    # plot contour and optimal weights
    ax2.contourf(XX, YY, ZZ, cmap='Reds', levels=30)
    ax2.set_xlabel(f'$w_0$')
    ax2.set_ylabel(f'$w_1$')
    u, s, vT = np.linalg.svd(X)
    r = np.count_nonzero((s > 1e-8).astype(int)) # rank of X
    d = X.shape[1]
    
    sample_size = 20
    optimal_weights = np.zeros(shape=[sample_size, d])
    optimal_weights[:, 0] = (np.linalg.pinv(X) @ y.reshape(-1, 1))[0]
    optimal_weights[:, 1] = (np.linalg.pinv(X) @ y.reshape(-1, 1))[1]
    
    alpha = np.linspace(-5, 5, sample_size)
    for j in range(r, d): # r+1, ..., d -> r, ..., d-1
        for i in range(sample_size):
            optimal_weights[i, :] = alpha[i] * vT[j, :] # see discussion

    ax2.scatter(optimal_weights[:, 0], optimal_weights[:, 1], 
                marker='o', facecolors='k', s=1)

    fig.tight_layout()
    plt.show() # can move 3d plot around

    # # save
    # if independent:
    #     plt.savefig('../img/11_loss_independent.png')
    # else:
    #     plt.savefig('../img/11_loss_dependent.png')



if __name__ == '__main__':
    plot_loss_surface(independent=True)
    plot_loss_surface(independent=False)
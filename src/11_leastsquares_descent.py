import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
np.random.seed(0)


def loss(w):
    '''linear least square loss'''
    return ((X @ w - y)**2).mean()

def grad(w):
    '''gradient step for the lls loss function'''
    # 2*((X @ w - y) * X[:, k]).mean() for each k -> broadcast
    dw = 2*((X @ w - y).reshape(-1, 1) * X).mean(axis=0)
    return dw / np.linalg.norm(dw)


# generate data
n = 1000
X = np.zeros((1000, 2))
X[:, 1]= np.random.uniform(low=-1, high=1, size=n)
X[:, 0] = 1
y = -1 + 3*X[:, 1] + 0.01*np.random.randn(1000) # signal: y = 3x - 1 + noise
assert loss(np.array([-1, 3])) < 0.001          # in theory, J(-1, 3) = 0


# initialize weights; perform iterative descent
w = np.array([-4, -4], dtype=np.float64)
epochs = 30
eta = 0.6
gradient_history = np.zeros((epochs, 2)) 
loss_history = []
for _ in range(epochs):
    w -= eta * grad(w)
    gradient_history[_, :] = w 
    loss_history.append(loss(w))


# contour plot
fig, ax = plt.subplots(2, 1, figsize=(6, 9))
N = 200 # size of grid
row = np.linspace(-5, 5, N)
XX, YY = np.meshgrid(row, row)
ZZ = np.zeros_like(XX)
for i in range(N):
    for j in range(N):
        ZZ[i, j] = loss(np.array([XX[i, j], YY[i, j]]))


# plot steps on gradient
ax[1].contourf(XX, YY, ZZ, cmap='Reds')
ax[1].scatter(gradient_history[:, 0], gradient_history[:, 1], marker='X', facecolors='k', edgecolors='k')
ax[1].set_xlabel(f'$w_0$')
ax[1].set_ylabel(f'$w_1$')


# plot loss history
ax[0].plot(loss_history, linestyle='--', c='red', linewidth=0.8)
ax[0].scatter(list(range(epochs)), loss_history, marker='o', facecolors='None', edgecolors='k')
ax[0].set_ylabel(f'loss')
ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[0].set_xlabel(f'epoch')


# save
fig.tight_layout()
plt.savefig('../img/11_leastsquares_descent.png')
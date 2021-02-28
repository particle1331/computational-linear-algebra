import numpy as np
import matplotlib.pyplot as plt
import matplotlib
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


# generate noisy sample data
n = 1000
X = np.zeros((n, 2))
X[:, 1] = np.random.uniform(low=-1, high=1, size=n)
X[:, 0] = 1
w_true = np.array([-1, 3])
y = (X @ w_true) + 0.01 * np.random.randn(n)      # signal: y = -1 + 3x + noise
assert loss(w_true) < 0.001                       # without noise, J(-1, 3) = 0


# gradient descent
w = np.array([-4, -4], dtype=np.float64)
epochs = 18
eta = 0.6
w_hist = np.zeros(shape=[epochs, 2]) 
loss_hist = []
for _ in range(epochs):
    w -= eta * grad(w)
    w_hist[_, :] = w 
    loss_hist.append(loss(w))


# contour plot
fig, ax = plt.subplots(2, 1, figsize=(4, 6))
N = 200 # size of grid
row = np.linspace(-5, 5, N)
XX, YY = np.meshgrid(row, row)
ZZ = np.zeros_like(XX)
for i in range(N):
    for j in range(N):
        ZZ[i, j] = loss(np.array([XX[i, j], YY[i, j]]))


# plot steps on gradient
ax[1].contourf(XX, YY, ZZ, cmap='Reds')
ax[1].scatter(w_hist[:, 0], w_hist[:, 1], marker='2', facecolors='k')
ax[1].set_xlabel(f'$w_0$')
ax[1].set_ylabel(f'$w_1$')


# plot loss history
ax[0].plot(loss_hist, linestyle='--', c='red', linewidth=0.8)
ax[0].scatter(list(range(epochs)), loss_hist, marker='o', facecolors='None', edgecolors='k')
ax[0].set_ylabel(f'loss')
ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[0].set_xlabel(f'epoch')


# save
fig.tight_layout()
plt.show()
# plt.savefig('../img/11_leastsquares_descent.png')

# check with pinv
w_best = w_hist[np.argmin(loss_hist)]   
print('MSE(y, X @ w_true) =', ((y - X @ w_true)**2).mean())
print('MSE(y, X @ w_best) =', ((y - X @ w_best)**2).mean())
print('MSE(y, X @ X_pinv @ y) =', ((y - X @ np.linalg.pinv(X) @ y)**2).mean())
print('w_true =', w_true)
print('w_best =', w_best)
print('X_pinv @ y =', np.linalg.pinv(X) @ y)
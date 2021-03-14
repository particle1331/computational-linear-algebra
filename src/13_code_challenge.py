import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
np.random.seed(0)

'''
Code challenge: Generate random matrix with condition no. 42
'''

def plot_svd_scree(A, save=False, filename=''):
    # Create 2x2 sub plots
    gs = gridspec.GridSpec(2, 4)
    plt.figure()

    ax = plt.subplot(gs[0, 0]) # row 0, col 0
    plt.imshow(A)
    plt.axis('off')
    plt.title(f'$A =$')

    ax = plt.subplot(gs[0, 1]) # row 0, col 1
    plt.imshow(U)
    plt.axis('off')
    plt.title(f'$U$')

    ax = plt.subplot(gs[0, 2]) # row 0, col 0
    plt.imshow(np.diag(s))
    plt.axis('off')
    plt.title(f'$\Sigma$')

    ax = plt.subplot(gs[0, 3]) # row 0, col 1
    plt.imshow(VT)
    plt.axis('off')
    plt.title(f'$V^\intercal$')

    ax = plt.subplot(gs[1, :]) # row 1, span all columns
    x = range(1, 20+1)
    plt.scatter(x, s, marker='o', facecolors='none', edgecolors='k')
    plt.plot(x, s, linestyle='--', c='red', linewidth=0.8)
    plt.ylabel(f'$\sigma$')
    plt.xlabel(f'k')
    plt.xticks(x)

    # save
    if save:
        plt.savefig('../img/' + filename)
        plt.show()


def generate_cond_42(n):
    # generate random matrix; get svd
    A = np.random.randn(n, n)
    U, s, VT = np.linalg.svd(A)

    # rescale: f(σ) = aσ + b, such that f(σ_0) = 42, and f(σ_n) = 1 
    a = -41.0 / (s[-1] - s[0])
    b = 42 - a * s[0]
    s = a * s + b

    # new random matrix with cond #. k = 42
    return U @ np.diag(s) @ VT


if __name__ == '__main__':
    A = generate_cond_42(20)
    U, s, VT = np.linalg.svd(A)
    
    # plot svd decomposition
    plot_svd_scree(A, save=True, filename='13_code_challenge.png')


    

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
np.random.seed(0)


def plot_svd_scree(A, save=False, filename=''):
    
    # Compute SVD
    U, s, VT = np.linalg.svd(A)

    # Create 2x2 subplots
    gs = gridspec.GridSpec(2, 4)
    plt.figure()
    plt.set_cmap('magma')

    ax = plt.subplot(gs[0, 0])
    plt.imshow(A)
    plt.axis('off')
    plt.title(f'$A =$')

    ax = plt.subplot(gs[0, 1])
    plt.imshow(U)
    plt.axis('off')
    plt.title(f'$U$')

    ax = plt.subplot(gs[0, 2])
    plt.imshow(np.diag(s))
    plt.axis('off')
    plt.title(f'$\Sigma$')

    ax = plt.subplot(gs[0, 3])
    plt.imshow(VT)
    plt.axis('off')
    plt.title(f'$V^\intercal$')

    ax = plt.subplot(gs[1, :]) # row 1, span all columns
    plt.scatter(range(len(s)), s, marker='o', facecolors='white', edgecolors='k', zorder=3)
    plt.plot(range(len(s)), s, linestyle='--', c='red', linewidth=0.8)
    plt.xlim(-0.5, int(0.5*len(s)) + 0.5)
    plt.ylabel(f'$\sigma$')
    plt.xlabel(f'k')
    plt.tight_layout()

    # save
    if save:
        plt.savefig('../img/' + filename)
        plt.show()
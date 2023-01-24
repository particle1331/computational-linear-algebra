import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
np.random.seed(42)

if __name__ == '__main__':
    # initialize
    norms = []
    A = np.array([[2, -1, 0], [1, 4, -1], [-1,-1,-3]])
    A = A / ( np.linalg.eig(A)[0][-1] )
    x = np.random.randn(3).reshape(-1, 1)
    norms.append(np.linalg.norm(x))

    # iterating
    N = 200
    for i in range(N):
        x = A @ x
        norms.append(np.linalg.norm(x))
        pass 

    # plotting
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot([norms[k]/norms[k-1] for k in range(1, N)])
    ax.set_ylabel('norm ratio')
    ax.set_xlabel('k')
    
    # save
    plt.savefig('../img/18_iterated_maps.png')

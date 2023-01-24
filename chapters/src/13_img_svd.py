import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils import plot_svd_scree


# load image and extract one channel -> grayscale
image = Image.open('../img/dog.jpg')
image = np.array(image)[:, :, 0]

# get SVD
U, s, VT = np.linalg.svd(image)
S = np.zeros([U.shape[1], VT.shape[0]])
np.fill_diagonal(S[:len(s), :len(s)], s)

# scree plot
x = range(len(s))
s = s / s[0]
plt.scatter(x[:30], s[:30], marker='o', facecolors='none', edgecolors='k')
plt.plot(x[:30], s[:30], linestyle='--', c='red', linewidth=0.8)
plt.ylabel(f'$\sigma$')
plt.xlabel(f'k')
plt.xlim(0, 30)
plt.tight_layout()
plt.savefig('../img/13_img_svd-scree.png')

# low-rank reconstruction
# the two images below add to the original image! recon + compl = image
f = plt.figure(figsize=(6, 10))
for j in range(3):
    k = [1, 10, 30][j]
    recon = U[:, :k] @ S[:k, :k] @ VT[:k, :]
    compl = U[:, k:] @ S[k:, k:] @ VT[k:, :]
    ax_recon = f.add_subplot(3, 2, 2*j+1)
    ax_compl = f.add_subplot(3, 2, 2*j+2)
    ax_recon.set_ylabel(f'k={k}')
    ax_recon.imshow(recon, cmap='gray')
    ax_compl.imshow(compl, cmap='gray')

plt.tight_layout()
plt.savefig('../img/13_img_svd-reconstruction.jpg')


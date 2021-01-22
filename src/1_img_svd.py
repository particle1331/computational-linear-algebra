import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# load image and extract one channel -> grayscale
image = Image.open('../img/bb.jpg')
image = np.array(image)[:, :, 0]

# get SVD
U, S, V = np.linalg.svd(image)
print(U.shape, S.shape, V.shape)

# plot spectrum
plt.plot(S/S[0], 's-')
plt.xlim((0, 40))
plt.xlabel("Component number")
plt.ylabel(r"Singular value $\sigma$")
plt.savefig('../img/1_img_svd-screeplot.png')

# low-rank reconstruction
# the two images below add to the original image! recon + compl = image
f = plt.figure(figsize=(6, 10))
for j in range(4):
    k = 2**j
    recon = U[:, :k] @ np.diag(S[:k]) @ V[:k, :]
    compl = U[:, k:] @ np.diag(S[k:]) @ V[k:, :]
    ax_recon = f.add_subplot(4, 2, 2*j+1)
    ax_compl = f.add_subplot(4, 2, 2*j+2)
    ax_recon.set_ylabel(f'k={k}')
    ax_recon.imshow(recon, cmap='gray')
    ax_compl.imshow(compl, cmap='gray')

plt.savefig('../img/1_img_svd-reconstruction.jpg')
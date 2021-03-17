import numpy as np 
import matplotlib.pyplot as plt
from utils import plot_svd_scree
np.random.seed(0)


# 50 looks like a good number
A = np.zeros([28, 28])

# lets make a KDE of 10 gaussians
x = np.arange(28)
y = np.arange(28)
xx, yy = np.meshgrid(x, y)

z  = np.exp((-(-xx -  3)**2 - (yy - 2)**2)/10.0)
h1 = np.random.choice(np.arange(28), 15)
h2 = np.random.choice(np.arange(28), 15)
h3 = np.random.choice(np.arange(2, 28), 15)
for (x, y, t) in zip(h1, h2, h3):
    z += np.exp((-(xx - x)**2 - (yy - y)**2)/(2*t))

# plot
plot_svd_scree(z, save=True, filename='13_kde.png')
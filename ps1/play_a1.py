"Playing with assignment 1"

import numpy as np
import matplotlib.pyplot as plt
from sheet1 import PCA

npz = np.load("data/banana.npz")

data, label = npz["data"], npz["label"]

# plot raw data
print("Plotting banana dataset (using only positive labels)")
positive_only = data[:, label[0] == 1]
plt.scatter(*positive_only)
plt.show()

# compute pca and plot
print("Plotting centered data and principal components")
pca = PCA(positive_only.T)
plt.scatter(*(pca.C.T), c="yellow")
plt.arrow(0,
            0,
            pca.U[0, 0] * pca.D[0],
            pca.U[0, 1] * pca.D[0],
            color="red",
            head_width=0.1,
            head_length=0.1)
plt.arrow(0,
            0,
            pca.U[1, 0] * pca.D[1],
            pca.U[1, 1] * pca.D[1],
            color="green",
            head_width=0.1,
            head_length=0.1)
plt.axis("scaled")
plt.show()

# denoise and plot
print("Denoising data onto first PC")
denoised = pca.denoise(positive_only.T, 1)
plt.scatter(*(denoised.T), c="violet")
plt.arrow(0,
            0,
            pca.U[0, 0] * pca.D[0],
            pca.U[0, 1] * pca.D[0],
            color="red",
            head_width=0.1,
            head_length=0.1)
plt.arrow(0,
            0,
            pca.U[1, 0] * pca.D[1],
            pca.U[1, 1] * pca.D[1],
            color="green",
            head_width=0.1,
            head_length=0.1)
plt.axis("scaled")
plt.show()

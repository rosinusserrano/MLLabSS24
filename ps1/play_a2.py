"Playing with assignment 2"

import numpy as np
import matplotlib.pyplot as plt
from sheet1 import gammaidx

data = np.random.multivariate_normal([0, 0], np.eye(2), (500, ))

print("Plotting raw data")
plt.scatter(*data.T)
plt.axis("scaled")
plt.show()

print("Plotting data with color indicating gamma index")
gamma_idxs = gammaidx(data, 30)
plt.scatter(*data.T, cmap="summer", c=gamma_idxs)
plt.axis("scaled")
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# load cleaned data
df = pd.read_csv("secom_cleaned.csv", header=None)

# find principal components
pca = PCA(n_components=3) # use 3 PCs as decided in part 1b
pcs = pca.fit_transform(df)
loadings = pca.components_

# recalculate outliers (copied from part 1c)
quadrants = [
  (pcs[:, 0] > 0) & (pcs[:, 1] > 0), # top right
  (pcs[:, 0] < 0) & (pcs[:, 1] > 0), # top left
  (pcs[:, 0] > 0) & (pcs[:, 1] < 0) # bottom right
]
distances = pcs[:, 0]**2 + pcs[:, 1]**2
outliers = []
for q in quadrants:
  points = np.where(q)[0]
  if len(points) > 0:
    farthest = points[np.argmax(distances[points])]
    outliers.append(farthest)

# Q-residual (squared prediction error)
pred = pcs @ loadings # reconstruct using 3 PCs
residual = df - pred
Q = np.sum(residual**2, axis=1)
plt.scatter(range(len(Q)), Q, s=10)
plt.axhline(np.quantile(Q, 0.99), color='r', linestyle='--')
for i in outliers:
  plt.scatter(i, Q[i], color='red')
plt.xlabel("Run #")
plt.ylabel("Q-residual")
plt.yscale("log")
plt.title("Q-residual Control Chart")
plt.savefig("1d_q_residual.png")
plt.close()

# Hotelling T^2
eigenvalues = pca.explained_variance_
T2 = np.sum(pcs**2 / eigenvalues, axis=1)
plt.scatter(range(len(T2)), T2, s=10)
for i in outliers:
  plt.scatter(i, T2[i], color='red')
plt.axhline(np.quantile(T2, 0.99), color='r', linestyle='--')
plt.xlabel("Run #")
plt.ylabel("Hotelling T^2")
plt.yscale("log")
plt.title("Hotelling T^2 Control Chart")
plt.savefig("1d_hotelling_t2.png")
plt.close()
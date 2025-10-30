import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# load cleaned data
df = pd.read_csv("secom_cleaned.csv", header=None)

# find principal components
pca = PCA()
pcs = pca.fit_transform(df)

# i. scores plot of PC2 vs PC1, with 3 faraway points highlighted
quadrants = [
  (pcs[:, 0] > 0) & (pcs[:, 1] > 0), # top right
  (pcs[:, 0] < 0) & (pcs[:, 1] > 0), # top left
  (pcs[:, 0] > 0) & (pcs[:, 1] < 0) # bottom right
]
distances = pcs[:, 0]**2 + pcs[:, 1]**2
plt.scatter(pcs[:, 0], pcs[:, 1])
outliers = []
for q in quadrants:
  points = np.where(q)[0]
  if len(points) > 0:
    farthest = points[np.argmax(distances[points])]
    outliers.append(farthest)
    plt.scatter(pcs[farthest, 0], pcs[farthest, 1], color='red')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PC2 vs PC1 (scores plot)')
plt.savefig('1c_scores.png')
plt.close()

# ii. loadings plot of PC1 vs PC2
loadings = pca.components_
plt.scatter(loadings[:, 0], loadings[:, 1])
for i in range(len(loadings)):
  plt.text(loadings[i, 0], loadings[i, 1], f'{i+1}')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PC2 vs PC1 (loadings plot)')
plt.savefig('1c_loadings.png')
plt.close()

# iii. variable contribution bar chart
loadings = pca.components_
for i, idx in enumerate(outliers):
  contributions = pcs[idx, 0]*loadings[:, 0] + pcs[idx, 1]*loadings[:, 1]
  plt.bar(range(1, len(contributions)+1), contributions)
  plt.xlabel('Feature')
  plt.ylabel('Contribution')
  plt.title(f'Variable Contributions for Outlier {i+1}')
  plt.savefig(f'1c_contribution_{i+1}.png')
  plt.close()
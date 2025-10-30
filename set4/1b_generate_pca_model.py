import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# load cleaned data
df = pd.read_csv("secom_cleaned.csv", header=None)

# find principal components
pca = PCA()
pcs = pca.fit_transform(df)

# plot PC2 vs PC1
plt.scatter(pcs[:, 0], pcs[:, 1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PC2 vs PC1 (scores plot)')
plt.savefig('1b_scores.png')
plt.close()

# scree plot
explained_variance_ratio = pca.explained_variance_ratio_
plt.plot(explained_variance_ratio)
plt.xlabel('Principal Component #')
plt.ylabel('Percent of Variance Captured')
plt.title('Scree Plot')
plt.savefig('1b_scree.png')
plt.close()

# print cumulative variance (to determine # of PCs)
cumulative_variance = explained_variance_ratio.cumsum()
for i, val in enumerate(cumulative_variance, start=1):
    print(f"PC{i}: {val:.3f}")
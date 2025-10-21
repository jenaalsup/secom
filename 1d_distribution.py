import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro, normaltest, probplot

df = pd.read_csv("secom_cleaned.csv", header=None)

for i, (_, feature) in enumerate(df.items(), start=1):
  # histogram
  plt.figure()
  plt.hist(feature.dropna().values, bins=35)
  plt.ylabel('Count')
  plt.savefig(f'hist_feature_{i}.png')
  plt.close()

  # q-q plot
  plt.figure()
  probplot(feature.dropna().values, dist="norm", plot=plt)
  plt.savefig(f'qq_feature_{i}.png')
  plt.close()

  # shapiro-wilk test
  W, p_shapiro = shapiro(feature.dropna().values)

  # d'agostino's k-squared test
  K2, p_k2 = normaltest(feature.dropna().values)

  print(f'Feature {i}: Shapiro W={W}, p={p_shapiro}; K2={K2}, p={p_k2}')
  
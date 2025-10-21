import pandas as pd
import matplotlib.pyplot as plt

# load cleaned data
df = pd.read_csv("secom_cleaned.csv", header=None)

# plot control charts for each feature
for i, (_, feature) in enumerate(df.items(), start=1):
  desired_val = feature.mean()
  k = 3 # standard according to NIST textbook
  std_val = feature.std()
  UCL = desired_val + k * std_val
  LCL = desired_val - k * std_val

  plt.figure()
  plt.plot(range(1, len(feature) + 1), feature.values)
  plt.axhline(desired_val)
  plt.axhline(UCL)
  plt.axhline(LCL)
  plt.ylabel(f'Feature {i}')
  plt.savefig(f'control_chart_feature_{i}.png')
  plt.close()


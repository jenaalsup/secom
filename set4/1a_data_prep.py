import pandas as pd
from sklearn.preprocessing import StandardScaler

# load data
df = pd.read_csv("secom.data", delim_whitespace=True, header=None)

# keep first 11 features only
df = df.iloc[:, :11]

# i. remove feature with mean = 100, and std = 0
means = df.mean()
stds = df.std()
setpoint = (means == 100) & (stds == 0)
df = df.loc[:, ~setpoint]

# ii. drop rows where any of the 11 features is NaN
df = df.dropna()
print("after NaN drop (1535):", len(df))

# iii. drop rows where any of the 11 features is an extreme outlier (z-score > 10)
z_scores = (df - df.mean()) / df.std()
outliers = z_scores.abs() > 10
df = df[~outliers.any(axis=1)]
print("after outlier drop (1533):", len(df))

# iv. use StandardScaler to autoscale the data for each feature
scaler = StandardScaler()
df[:] = scaler.fit_transform(df)

# verify results
print("remaining runs:", len(df))
print("First timestamp (10 features):", df.iloc[0].values)
df.to_csv("secom_cleaned.csv", index=False, header=False)

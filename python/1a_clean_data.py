import pandas as pd

# load data
df = pd.read_csv("secom.data", delim_whitespace=True, header=None)

# keep first 4 features only
df = df.iloc[:, :4]

# drop rows where any of the 4 features is NaN
df = df.dropna()

# verify results
print("remaining runs:", len(df))
print("First timestamp (4 features):", df.iloc[0].values)
df.to_csv("secom_cleaned.csv", index=False, header=False)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# reused code: PCA, Q, T2
df = pd.read_csv("secom_cleaned.csv", header=None)
pca = PCA(n_components=3)
pcs = pca.fit_transform(df)
loadings = pca.components_
pred = pcs @ loadings
residual = df - pred
Q = np.sum(residual**2, axis=1)
eigenvalues = pca.explained_variance_
T2 = np.sum(pcs**2 / eigenvalues, axis=1)

# find alarms
Q_alarm  = (Q  > np.quantile(Q, 0.99)).astype(int)
T2_alarm = (T2 > np.quantile(T2, 0.99)).astype(int)
alarm_matrix = np.column_stack([Q_alarm, T2_alarm]) 
alarm_count = alarm_matrix.sum(axis=1)

# plot alarm matrix
plt.bar(range(1, len(alarm_count)+1), alarm_count, width=1.0)
plt.title('MSPC Alarm Control Chart')
plt.xlabel('Run #')
plt.ylabel('# of alarms')
plt.savefig('1e_alarm_control_chart.png')
plt.close()

# influence plot
T2_percent = 100 * pca.explained_variance_ratio_.sum()
Q_percent = 100 - T2_percent
plt.scatter(T2, Q)
plt.axhline(np.quantile(Q, 0.99), color='r', linestyle='--')
plt.axvline(np.quantile(T2, 0.99), color='r', linestyle='--')
plt.xscale('log')
plt.yscale('log') 
plt.title('Influence Plot')
plt.xlabel(f'Hotelling T^2 (In-model): {T2_percent:.1f}%')
plt.ylabel(f'Q-residual (Out-of-model): {Q_percent:.1f}%')
plt.savefig('1e_influence_plot.png')
plt.close()

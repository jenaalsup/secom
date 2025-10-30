import pandas as pd
import matplotlib.pyplot as plt

def alarm_41s(feature, mean, std, num_consecutive=4, num_std=1):
  result = [0] * len(feature)
  lower = mean - num_std * std
  upper = mean + num_std * std
  curr_above = 0
  curr_below = 0
  for i, x in enumerate(feature):
    if x > upper:
      curr_above += 1
      curr_below = 0
    elif x < lower:
      curr_below += 1
      curr_above = 0
    else:
      curr_above = 0
      curr_below = 0
    if curr_above >= num_consecutive or curr_below >= num_consecutive:
      result[i] = 1
  return result

def alarm_r4s(feature, mean, std, num_std=4):
  result = [0] * len(feature)
  upper = mean + num_std * std
  lower = mean - num_std * std
  for i, x in enumerate(feature):
    if i == len(feature) - 1:
      break
    if x > upper:
      if feature[i + 1] < lower:
        result[i + 1] = 1
    elif x < lower:
      if feature[i + 1] > upper:
        result[i + 1] = 1
  return result

def alarm_10x(feature, mean, num_consecutive=10):
  result = [0] * len(feature)
  curr_above = 0
  curr_below = 0
  for i, x in enumerate(feature):
    if x > mean:
      curr_above += 1
      curr_below = 0
    elif x < mean:
      curr_below += 1
      curr_above = 0
    else:
      curr_above = 0
      curr_below = 0
    if curr_above >= num_consecutive or curr_below >= num_consecutive:
      result[i] = 1
  return result

def alarm_7t(feature, num_same_direction=7):
  result = [0] * len(feature)
  curr_up = 0
  curr_down = 0
  for i in range(1, len(feature)):
    if feature[i] > feature[i - 1]:
      curr_up += 1
      curr_down = 0
    elif feature[i] < feature[i - 1]:
      curr_down += 1
      curr_up = 0
    else:
      curr_up = 0
      curr_down = 0
    if curr_up >= num_same_direction - 1 or curr_down >= num_same_direction - 1:
      result[i] = 1
  return result


df = pd.read_csv("secom_cleaned.csv", header=None)
alarm_matrix = pd.Series(0, index=df.index)

for i, (_, feature) in enumerate(df.items(), start=1):
  mean = feature.mean()
  std = feature.std()
  flags_41s = alarm_41s(feature, mean, std)
  flags_r4s = alarm_r4s(feature, mean, std)
  flags_10x = alarm_10x(feature, mean)
  flags_7t = alarm_7t(feature)
  alarm_df = pd.DataFrame({
    'run': feature.index + 1,
    '41s': flags_41s,
    'R4s': flags_r4s,
    '10x': flags_10x,
    '7T': flags_7t,
  })
  counts = alarm_df[['41s','R4s','10x','7T']].sum(axis=1)
  alarm_matrix += counts.values

plt.figure()
plt.plot(range(1, len(alarm_matrix) + 1), alarm_matrix.values)
plt.ylabel('# of alarms')
plt.savefig('alarm_control_chart.png')
plt.close()

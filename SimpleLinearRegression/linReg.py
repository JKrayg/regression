# Jake Krayger
# Simple Linear Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(1, 10.5, 20)
y = 2 * x + 1 + np.random.normal(0, 1, size=len(x))
df = pd.DataFrame({'Height': x, 'Weight': y})
df_hw = df[df.columns[[0, 1]]]

n = len(df_hw)

height = df_hw["Height"]
weight = df_hw["Weight"]

hw = np.array(list(zip(height, weight)))

# scale by feature
# def scale(data):
#     maxByFeat = [max([j[z] for j in [i for i in data]]) for z in range(len(data[0]))]
#     scaled_data = []

#     for m in data:
#         temp = []
#         for t in range(len(maxByFeat)):
#             temp.append(m[t] / maxByFeat[t])
#         scaled_data.append(np.array(temp))

#     return scaled_data

# sclHW = np.array(scale(hw))

# sum x
sumHeight = height.sum()
# sum x^2
sumHeightSq = sum([x**2 for x in height])
# sum y
sumWeight = weight.sum()
# sum xy
sumHW = sum([x * y for x, y in hw])

m = (n * sumHW - sumHeight * sumWeight) / (n * sumHeightSq - sumHeight**2)
b = (sumWeight - m * sumHeight) / n
y = m * height + b


pred = m * 300 + b

ssRes = sum((y - (m * x + b))**2 for x, y in hw)
ssTot = sum((y - weight.mean())**2 for y in weight)
r2 = 1 - (ssRes / ssTot)
print("R^2:", r2)

plt.scatter(hw[:, 0], hw[:, 1], c = "yellow", alpha = 0.5, edgecolors = "black")
plt.xlabel("Height")
plt.ylabel("Weight")
plt.plot(height, y)
plt.show()
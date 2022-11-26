import pandas as pd
import numpy as np
import utils
import visualization
from sklearn import linear_model
import matplotlib.pyplot as plt

# 1. Loading data
df = pd.read_csv("turboaz.csv", usecols=["Yurush", "Buraxilish ili", "Qiymet"])
df["Yurush"] = df["Yurush"].map(
    lambda yurush: int(utils.removeUnit(yurush, "km")))
df["Qiymet"] = df["Qiymet"].map(lambda qiymet: utils.convertPrice(qiymet))

# 2. Visualization
# visualization.plot(df)

# 3. Implementation of Linear Regression from scratch
# Normalize data using Z score normalization
x1 = df["Yurush"]
x2 = df["Buraxilish ili"]
y = df["Qiymet"]

x1 = np.array((x1 - x1.mean()) / x1.std())
x2 = np.array((x2 - x2.mean()) / x2.std())
x = np.c_[np.ones(x1.shape[0]), x1, x2]

y = np.array((y - y.mean()) / y.std())

# Implement gradient descent algorithm to minimize cost function
weights = np.random.rand(3)
weightList, costList = utils.gradientDescent(x, y, weights)

print(weightList[-1])
print(np.dot([1, 240000, 2000], weightList[-1]))
print(weightList[-1][0] + weightList[-1][1] * 240000 + weightList[-1]
      [2] * 2000)

# 4. Linear Regression using library
# reg = linear_model.LinearRegression()
# reg.fit(df[["Yurush", "Buraxilish ili"]], df["Qiymet"])

# print(reg.predict([[240000, 2000]]))

# plt.title("Cost function")
# plt.xlabel("Number of iterations")
# plt.ylabel("Cost")
# plt.plot(costList)
# plt.show()

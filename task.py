import pandas as pd
import numpy as np
import utils
import visualization
from sklearn import linear_model

# 1. Loading data
df = pd.read_csv("turboaz.csv", usecols=["Yurush", "Buraxilish ili", "Qiymet"])
df["Yurush"] = df["Yurush"].map(
    lambda yurush: int(utils.removeUnit(yurush, "km")))
df["Qiymet"] = df["Qiymet"].map(lambda qiymet: utils.convertPrice(qiymet))

# 2. Visualization
# visualization.plot(df)

# 3. Implementation of Linear Regression from scratch
# Normalize data using Z score normalization
X = df[["Yurush", "Buraxilish ili"]]
Y = df["Qiymet"]

Y = np.array((Y - Y.mean()) / Y.std())
X = X.apply(lambda rec: (rec - rec.mean()) / rec.std(), axis=0)

# Implement gradient descent algorithm to minimize cost function
p = utils.gradientDescent(X["Yurush"], X["Buraxilish ili"], Y)

print(p[-1])
print(p[-1][0] + p[-1][1] * 240000 + p[-1][2] * 2000)

# 4. Linear Regression using library
# reg = linear_model.LinearRegression()
# reg.fit(df[["Yurush", "Buraxilish ili"]], df["Qiymet"])
# print(reg.coef_)

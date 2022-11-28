import pandas as pd
import numpy as np
import utils
import visualization
from sklearn import linear_model
import matplotlib.pyplot as plt

# 1. Loading data
df = pd.read_csv("turboaz.csv", usecols=["Yurush", "Buraxilish ili", "Qiymet"])
df["Yurush"] = df["Yurush"].map(
    lambda yurush: int(utils.remove_unit(yurush, "km")))
df["Qiymet"] = df["Qiymet"].map(lambda qiymet: utils.convert_price(qiymet))

# 2. Visualization
visualization.plot_data(df)

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
w = np.random.rand(3)
weights, costs = utils.gradient_descent(x, y, w)

# Plot graph of cost function
visualization.plot_cost_function(costs)

# Plot points of Y (Qiymet) vs X1 (Yurush) and draw a line of predictions.
predicts = weights[-1][0] + x1.dot(weights[-1][1])
plt.scatter(x1, y)
plt.plot(x1, predicts, color="red")
plt.xlabel("Yurush")
plt.ylabel("Qiymet")
plt.show()

# Plot points of Y (Qiymet) vs X2 (Buraxilish ili) and draw a line of predictions.
predicts = weights[-1][0] + x2.dot(weights[-1][2])
plt.scatter(x2, y)
plt.plot(x2, predicts, color="red")
plt.xlabel("Buraxilish ili")
plt.ylabel("Qiymet")
plt.show()

# Plot 3D graph of points of Y (Qiymet), X1, X2, and predicted Y (Qiymet).
predicts = x.dot(weights[-1])
ax = plt.axes(projection="3d")
ax.set_xlabel("Yurush")
ax.set_ylabel("Buraxilish ili")
ax.set_zlabel("Qiymet")
ax.scatter3D(x1, x2, y)
ax.scatter3D(x1, x2, predicts, color="red")
plt.show()

# Testing
test_data = np.array([[240000, 2000], [415558, 1996]])

# Normalize test data
test_data_x1 = np.array(
    (test_data[:, 0] - df["Yurush"].mean()) / df["Yurush"].std())
test_data_x2 = np.array(
    (test_data[:, 1] - df["Buraxilish ili"].mean()) / df["Buraxilish ili"].std())
test_data_x = np.c_[np.ones(test_data_x1.shape[0]), test_data_x1, test_data_x2]

results = []

# Denormalize and print the results
for price in np.nditer(test_data_x.dot(weights[-1])):
    results.append(price * df["Qiymet"].std() + df["Qiymet"].mean())

print(results)
# Example output: [15823.421869751804, 5450.465380251384]

# 4. Linear Regression using library
reg = linear_model.LinearRegression()
reg.fit(df[["Yurush", "Buraxilish ili"]], df["Qiymet"])
pred = reg.predict(test_data)

print(pred)
# Output: [15820.54127243  5453.69414862]

# Solve linear regression by normal equation
W = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
print(W)

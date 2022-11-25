import numpy as np


# Remove units from milage and price
def removeUnit(value, unit):
    return value.replace(unit, "").replace(" ", "")


# Remove units and convert prices
def convertPrice(priceAsString):
    dollarIndex = priceAsString.find("$")

    price = int(removeUnit(priceAsString, "$" if dollarIndex > 0 else "AZN"))

    if dollarIndex > 0:
        price *= 1.7

    return price


# Generate random bias and weights
def generateWeights(n):
    weights = np.random.rand(n)

    return weights


# Calculate X (predictions)
def calculatePredictions(x1, x2, w0, w1, w2):
    n = len(x1)
    X = np.zeros(n)

    for i in range(n):
        X[i] = (w1 * x1[i]) + (w2 * x2[i]) + w0

    return X


# Calculate cost function
def costFunction(x, y):
    cost = np.sum((x - y) ** 2) / (2 * len(x))

    return cost


# Calculate gradients
def gradients(x1, x2, y, w0, w1, w2):
    n = len(x1)
    dj_dw0 = 0
    dj_dw1 = 0
    dj_dw2 = 0

    for i in range(n):
        h = (w1 * x1[i]) + (w2 * x2[i]) + w0
        dj_dw0_i = h - y[i]
        dj_dw1_i = (h - y[i]) * x1[i]
        dj_dw2_i = (h - y[i]) * x2[i]

        dj_dw0 += dj_dw0_i
        dj_dw1 += dj_dw1_i
        dj_dw2 += dj_dw2_i

    dj_dw0 /= n
    dj_dw1 /= n
    dj_dw2 /= n

    return dj_dw0, dj_dw1, dj_dw2


# Implement gradient descent algorithm to minimize cost function
def gradientDescent(x1, x2, y, alpha=0.001, iterations=10000):
    n = len(x1)
    alpha = alpha
    iterations = iterations
    w0 = 0
    w1 = 0
    w2 = 0

    p = []

    for i in range(n):
        dj_dw0, dj_dw1, dj_dw2 = gradients(x1, x2, y, w0, w1, w2)

        # cost = cost + ((h - y[i]) ** 2)
        w0 = w0 - (alpha * dj_dw0)
        w1 = w1 - (alpha * dj_dw1)
        w2 = w2 - (alpha * dj_dw2)

        p.append([w0, w1, w2])

    return p

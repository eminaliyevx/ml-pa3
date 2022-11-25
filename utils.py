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


# Calculate cost function
def costFunction(x, y):
    cost = np.sum((x - y) ** 2) / (2 * len(x))

    return cost


# Implement gradient descent algorithm to minimize cost function
def gradientDescent(x, y, weights, alpha=0.001, iterations=10000):
    n = len(y)
    weightList = [weights]
    costList = []

    for i in range(iterations):
        prediction = np.dot(x, weights)
        error = prediction - y
        cost = 1 / (2 * n) * np.dot(error.T, error)
        derivative = (1 / n) * alpha * np.dot(x.T, error)
        weights = weights - derivative

        weightList.append(weights)
        costList.append(cost)

    return weightList, costList

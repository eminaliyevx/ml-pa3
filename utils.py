import numpy as np


# Remove units from milage and price
def remove_unit(value, unit):
    return value.replace(unit, "").replace(" ", "")


# Remove units and convert prices
def convert_price(priceAsString):
    dollarIndex = priceAsString.find("$")

    price = int(remove_unit(priceAsString, "$" if dollarIndex > 0 else "AZN"))

    if dollarIndex > 0:
        price *= 1.7

    return price


# Calculate cost function
def cost_function(x, y):
    cost = np.sum(((x - y) ** 2) / (2 * len(y)))

    return cost


# Implement gradient descent algorithm to minimize cost function
def gradient_descent(x, y, w, alpha=0.001, iterations=10000):
    n = len(y)
    weights = [w]
    costs = []

    for _ in range(iterations):
        predict = x.dot(w)
        gradient = x.T.dot(predict - y)
        w = w - (1 / n) * alpha * gradient
        cost = cost_function(predict, y)

        weights.append(w)
        costs.append(cost)

    return weights, costs

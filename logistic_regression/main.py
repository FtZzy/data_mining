#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

DATA_SIZE = 10


def generate_data(random=False):
    """Generate fixed data."""
    if random:
        return (np.array(range(DATA_SIZE)),
                np.random.rand(DATA_SIZE))
    else:
        return (np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12]))


def generate_random_data(data_size=DATA_SIZE):
    """Generate data."""


def estimate_coefficient(data):
    """Estimate the coefficient ax+b."""
    x = data[0]
    y = data[1]
    n = np.size(x)
    x_avg = np.mean(x)
    y_avg = np.mean(y)

    cross_deviation = np.sum(x * y) - n * y_avg * x_avg
    x_deviation = np.sum(x**2) - n * x_avg**2

    a = cross_deviation / x_deviation
    b = y_avg - a * x_avg
    return a, b


def display_regression_line(data, a, b):
    """Plot the points and the regression line."""
    x, y = data
    y_pred = a * x + b

    plt.plot(x, y, 'mo')
    plt.plot(x, y_pred, 'g-')
    plt.show()
    plt.close()


if __name__ == "__main__":
    data = generate_data(random=True)
    a, b = estimate_coefficient(data)
    print("Regression line: y = %sx + %s" % (a, b))
    display_regression_line(data, a, b)

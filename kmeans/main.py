#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy import sqrt, mean
from numpy.random import randint

import matplotlib.pyplot as plt

SIZE_DATA = 20
NB_CLUSTER = 3

X_MIN = -10
X_MAX = 10
Y_MIN = -10
Y_MAX = 10


def generate_random_data(size, x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX):
    """Generate a list of tuples (x,y)."""
    result = []
    for _i in range(size):
        result.append((randint(x_min, x_max), randint(y_min, y_max)))

    return result


def initialize_centers(data, k):
    """Generate the initial mean points."""
    x_data_min = min(p[0] for p in data)
    x_data_max = max(p[0] for p in data)
    y_data_min = min(p[1] for p in data)
    y_data_max = max(p[1] for p in data)

    return generate_random_data(
        k,
        x_data_min,
        x_data_max,
        y_data_min,
        y_data_max
    )


def euclidean_distance(a, b):
    """Compute the euclidean distance for 2 dimensions."""
    return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def assign_points(data, centers):
    """Assign the data points to the closest centers."""
    result = {c: [] for c in centers}
    for point in data:
        min_distance = float("inf")
        for c in centers:
            dist = euclidean_distance(point, c)
            if dist < min_distance:
                min_distance = dist
                min_center = c
        result[min_center].append(point)

    return result


def get_average(points):
    """Give the average of a list of points."""
    x = mean([p[0] for p in points])
    y = mean([p[1] for p in points])
    return x, y


def update_centers(assign):
    """Update the centers of the data."""
    result = []
    for a in assign:
        avg = get_average(assign[a])
        result.append(avg)
    return result


def display_clusters(assign):
    """Graphical display of the results."""
    for c in assign:
        plt.plot(c[0], c[1], "r*")
        plt.plot(
            [p[0] for p in assign[c]],
            [p[1] for p in assign[c]],
            "o"
        )
    plt.show()
    plt.close()


if __name__ == "__main__":
    data = generate_random_data(SIZE_DATA)
    centers = initialize_centers(data, NB_CLUSTER)
    assign = assign_points(data, centers)
    last_assign = None
    while assign != last_assign:
        last_assign = assign
        centers = update_centers(assign)
        assign = assign_points(data, centers)

    display_clusters(assign)

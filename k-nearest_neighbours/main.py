#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

NB_CLUSTER = 3


def generate_dataset():
    """Generate dataset."""
    return [[1.22148151, 6.04078543, 0],
            [1.31654309, 4.92098314, 0],
            [1.49610869, 1.13769523, 0],
            [2.24987882, 6.71659632, 0],
            [2.83858305, 6.53739512, 0],
            [5.51024623, 3.68040164, 1],
            [6.71324726, 1.44088423, 1],
            [4.72910635, 8.12165411, 2],
            [7.12581829, 7.59238996, 2],
            [7.37158982, 8.08647509, 2]]


def display_data(data, point):
    """Display the data."""
    plt.plot(point[0], point[1], "yd")
    for p in data:
        if p[2] == 0:
            plt.plot(p[0], p[1], "r*")
        elif p[2] == 1:
            plt.plot(p[0], p[1], "go")
        else:
            plt.plot(p[0], p[1], "bs")
    plt.show()
    plt.close()


def euclidean_distance(a, b):
    """Compute the euclidean distance for 2 dimensions."""
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def get_neighbours(data, point, nb_neighbours):
    """Get the nearest neighbours of a point."""
    distances = []
    for row in data:
        dist = euclidean_distance(row, point)
        distances.append((row, dist))
    distances.sort(key=lambda t: t[1])
    return distances[:nb_neighbours]


def predict_point_classification(data, point, nb_neighbours):
    """Predict the classification for a point."""
    neighbours = get_neighbours(data, point, nb_neighbours)
    classes = [n[0][2] for n in neighbours]
    return max(set(classes), key=classes.count)


if __name__ == "__main__":
    data = generate_dataset()
    point = np.random.rand(2) * 10
    prediction = predict_point_classification(data, point, nb_neighbours=3)
    display_data(data, point)
    print("Prediction for ", point, " is: ", prediction)

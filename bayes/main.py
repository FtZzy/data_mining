#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def generate_data():
    """Generate the data for our studie."""
    return {"male": [[182, 81.6, 30],
                     [180, 86.2, 28],
                     [170, 77.1, 30],
                     [180, 74.8, 25]],
            "female": [[152, 45.4, 15],
                       [168, 68.0, 20],
                       [165, 59.3, 18],
                       [175, 68.4, 23]]}


def summarize_class_data(data):
    """Compute the mean and the std of values by classes."""
    mean = {}
    std = {}
    for label in data:
        n = len(data[label][0])
        mean[label] = []
        std[label] = []
        for i in range(n):
            mean[label].append(np.mean([v[i] for v in data[label]]))
            std[label].append(np.std([v[i] for v in data[label]]))
        mean[label].append(n + 1)
        # std[label].append(n + 1)
    return mean, std


def gaussian_probability(x, mean, std):
    """Compute the Gaussian probability."""
    e = np.exp(-((x - mean)**2 / (2 * std**2)))
    result = (1 / (np.sqrt(2 * np.pi) * std)) * e
    return result


def compute_probabilities(test, mean, std):
    """Compute the probabilities for test values."""
    result = {}
    nb_person = sum([mean[label][-1] for label in mean])
    for label in mean:
        result[label] = mean[label][-1] / float(nb_person)
        for i in range(len(test)):
            result[label] *= gaussian_probability(test[i],
                                                  mean[label][i],
                                                  std[label][i])

    return result


if __name__ == "__main__":
    """Try determine the sex with the height, the weight and the foot size."""
    data = generate_data()
    mean, std = summarize_class_data(data)
    test = (np.random.randint(150, 190),  # height
            np.random.randint(40, 90),  # weight
            np.random.randint(10, 35))  # foot size
    prob = compute_probabilities(test, mean, std)
    print("Probabilité for %s: %s" % (test, prob))

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import operator

from copy import deepcopy
from numpy import log
from numpy.random import randint


def generate_data():
    """Generate training data."""
    data = [[0, 0, 0, 'play'],
            [0, 0, 1, 'play'],
            [0, 1, 0, 'report'],
            [0, 1, 1, 'play'],
            [1, 0, 0, 'play'],
            [1, 0, 1, 'play'],
            [1, 1, 0, 'report'],
            [1, 1, 1, 'report']]
    labels = ['sunny', 'hot', 'wind']
    return data, labels


def shannon_entropy(data):
    """Compute the Shannon entropy of a dataset."""
    counter = {}
    result = 0
    for vect in data:
        decision = vect[-1]
        if decision not in counter:
            counter[decision] = 0
        counter[decision] += 1

    for decision in counter:
        p = counter[decision] / float(len(data))
        result -= p * log(p) / log(2)

    return result


def split_data(data, col, value):
    """Split the dataset for an axis and a value."""
    result = []
    for row in data:
        if row[col] == value:
            row_without_col = row[:col]
            row_without_col.extend(row[col + 1:])
            result.append(row_without_col)
    return result


def define_feature(data, nb_feature):
    """Choose the best feature to split."""
    result = -1
    best_entropy = -1
    data_entropy = shannon_entropy(data)
    for i in range(nb_feature):
        values = set(row[i] for row in data)
        value_entropy = 0

        for v in values:
            splitted_data = split_data(data, i, v)
            p = len(splitted_data) / float(len(data))
            value_entropy += p * shannon_entropy(splitted_data)

        delta = data_entropy - value_entropy
        if (delta > best_entropy):
            best_entropy = delta
            result = i

    return result


def get_most_represented(classes):
    """Get the most represented value."""
    counter = {}
    for c in classes:
        if c not in counter:
            counter[c] = 0
        counter[c] += 1
    return sorted(counter.iteritems(),
                  key=operator.itemgetter(1),
                  reverse=True)


def create_tree(data, labels):
    """Create the decision tree."""
    decisions = [d[-1] for d in data]

    if decisions.count(decisions[0]) == len(decisions):
        return decisions[0]  # Only the same decision
    elif len(decisions[0]) == 1:
        return get_most_represented(decisions)  # Last value

    feature = define_feature(data, len(labels))
    label = labels[feature]
    result = {label: {}}
    del(labels[feature])

    values = set(d[feature] for d in data)
    for value in values:
        result[label][value] = \
            create_tree(split_data(data, feature, value), labels)

    return result


def predict_classification(tree, labels, test):
    """Predict the classification for a test."""
    if isinstance(tree, dict):
        node, node_choices = tree.popitem()
        feature = node_choices[test[labels.index(node)]]
        result = predict_classification(feature, labels, test)
    else:
        result = tree
    return result


if __name__ == "__main__":
    """Predict the possibility to play in depending on the weather
    (sunny, wetness, wind, ...)"""
    data, labels = generate_data()
    decision_tree = create_tree(data, deepcopy(labels))
    print("Decision tree: ", decision_tree)
    test_value = [randint(2), randint(2), randint(2)]
    print("Value tested: ", test_value)
    classification = predict_classification(decision_tree, labels, test_value)
    print("Classification: ", classification)

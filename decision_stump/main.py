#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy.random import rand


def generate_random_data(n=10):
    """Generate random data."""
    return rand(10)


def get_decision(data, decision_value):
    """Get the decision depending the decision value."""
    result = map(lambda x : x > 0.6, data)
    return list(result)


if __name__ == "__main__":
    """One level decision tree.
    Here, it choose 'yes' if the value is upper than 0.6 and 'no' else."""
    data = generate_random_data()
    decision_value = 0.6
    result = get_decision(data, decision_value)
    print(data)
    print("Decision value: ", decision_value)
    print(result)

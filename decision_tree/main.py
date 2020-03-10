#!/usr/bin/env python
# -*- coding: utf-8 -*-


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


if __name__ == "__main__":
    """Predict the possibility to play in depending on the weathe
    (sunny, wetness, wind, ...)"""
    data, labels = generate_data()

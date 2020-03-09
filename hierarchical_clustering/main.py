#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy.random import rand
from scipy.cluster.hierarchy import dendrogram, linkage

import matplotlib.pyplot as plt

SIZE_DATA = 20
DIM_DATA = 2


if __name__ == "__main__":
    data = rand(SIZE_DATA, DIM_DATA)

    plt.figure()
    linked = linkage(data)
    dendrogram(linked, show_leaf_counts=True)
    plt.show()

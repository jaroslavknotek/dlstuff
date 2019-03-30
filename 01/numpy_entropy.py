#!/usr/bin/env python3
# jaroslav knotek: 43b3e7da-366d-11e8-9de3-00505601122b

import numpy as np


def compute_entropy(x):
    return -np.sum(np.log(x) * x)


def compute_cross_entropy(p, q):
    return -np.sum(np.log(p) * q)


def get_model_probabilities():
    # Load model distribution, each line `word \t probability`.
    with open("numpy_entropy_model.txt", "r") as model:
        model = sorted([line.rstrip("\n").split("\t") for line in model])
        prob_touples = [(symbol, float(prob_str)) for symbol, prob_str in model]
        # TODO: process the line, aggregating using Python data structures
        return dict(prob_touples)


def get_data_histogram():
    with open("numpy_entropy_data.txt", "r") as data:
        data_lines = [line.rstrip("\n") for line in data]
    hist = {}
    for val in data_lines:
        hist[val] = hist.get(val, 0) + 1
    return hist


if __name__ == "__main__":
    probabilities = get_model_probabilities()

    data_histogram = get_data_histogram()

    data_occurrence = np.array([data_histogram[key] for key in sorted(data_histogram.keys())])

    # TODO: Create a NumPy array containing the data distribution.
    data_distribution = data_occurrence / np.sum(data_occurrence)
    # TODO: Create a NumPy array containing the model distribution.
    model_distribution = [probabilities.get(key, 0) for key in sorted(data_histogram.keys())]

    # TODO: Compute and print the entropy H(data distribution)
    entropy = compute_entropy(data_distribution)

    # TODO: Compute and print cross-entropy H(data distribution, model distribution)
    cross_entropy = compute_cross_entropy(model_distribution, data_distribution)

    # TODO: KL-divergence D_KL(data distribution, model_distribution)
    kl_divergence = cross_entropy - entropy

    print("{:.2f}".format(entropy))
    print("{:.2f}".format(cross_entropy))
    print("{:.2f}".format(kl_divergence))

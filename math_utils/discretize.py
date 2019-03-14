import numpy as np


def bucket_discretize(min_value, max_value, current_value, buckets=10):
    bins = np.linspace(min_value, max_value, buckets)
    idx = np.digitize(current_value, bins)
    return idx

import numpy as np
from sklearn.cluster import KMeans


class DiscretizeParam(object):
    feature_name = None
    discretize_function = None
    buckets_amount = None
    def __init__(self, feature_name, discretize_function, buckets_amount):
        self.feature_name = feature_name
        self.discretize_function = discretize_function
        self.buckets_amount = buckets_amount
    
    def __repr__(self):
        return "DP<{}, {}, {}>".format(self.feature_name, self.discretize_function.__name__, self.buckets_amount)

def bucket_discretize(values, current_value, buckets):
    min_value = min(values)
    max_value = max(values)
    bins = np.linspace(min_value, max_value, buckets)
    idx = np.digitize(current_value, bins)
    return idx

def frequency_discretize(values, current_value, buckets):
    split = np.array_split(np.sort(values), buckets)
    cutoffs = [x[-1] for x in split]
    cutoffs = cutoffs[:-1]
    idx = np.digitize(current_value, cutoffs, right=True)
    return idx

def kbins_discretize(values, current_value, buckets):
    values2D = np.array([[v, 0] for v in values])
    kmeans = KMeans(n_clusters=buckets, random_state=0).fit(values2D)
    curr_val2D = np.array([[current_value, 0]])
    val = kmeans.predict(curr_val2D)
    return val[0]

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

class Discretizer(object):
    bucket_models = {}
    frequency_models = {}
    kmean_models = {}

def bucket_discretize(dataset_name, feature_name, values, current_value, buckets):
    key = (dataset_name, feature_name, buckets)
    if not key in Discretizer.bucket_models.keys():
        min_value = min(values)
        max_value = max(values)
        Discretizer.bucket_models[key] = np.linspace(min_value, max_value, buckets)    
    bins = Discretizer.bucket_models[key]
    idx = np.digitize(current_value, bins)
    return idx

def frequency_discretize(dataset_name, feature_name, values, current_value, buckets):
    key = (dataset_name, feature_name, buckets)
    if not key in Discretizer.frequency_models.keys():
        split = np.array_split(np.sort(values), buckets)
        cutoffs = [x[-1] for x in split]
        cutoffs = cutoffs[:-1]
        Discretizer.frequency_models[key] = cutoffs
    cutoffs = Discretizer.frequency_models[key]
    idx = np.digitize(current_value, cutoffs, right=True)
    return idx

def kbins_discretize(dataset_name, feature_name, values, current_value, buckets):
    key = (dataset_name, feature_name, buckets)
    if not key in Discretizer.kmean_models.keys():
        values2D = np.array([[v, 0] for v in values])
        Discretizer.kmean_models[key] = KMeans(n_clusters=buckets, random_state=0).fit(values2D) 
    kmeans = Discretizer.kmean_models[key]
    curr_val2D = np.array([[current_value, 0]])
    val = kmeans.predict(curr_val2D)
    return val[0]
